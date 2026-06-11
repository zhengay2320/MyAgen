from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from rs_service.core.manifest import write_json, write_manifest
from rs_service.core.raster import read_raster, write_raster
from rs_service.pipelines.base import flag, prepare_output_dir

try:  # pragma: no cover - pillow is present in normal installs
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


BAND_NAMES = {"blue", "green", "red", "nir", "swir1", "swir2"}
INDEX_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "ndvi": ("nir", "red"),
    "ndwi": ("green", "nir"),
    "mndwi": ("green", "swir1"),
    "ndbi": ("swir1", "nir"),
    "savi": ("nir", "red"),
    "evi": ("nir", "red", "blue"),
}


def run_spectral_indices(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    indices: list[str] | None = None,
    band_mapping: dict[str, int] | None = None,
    band_map: dict[str, int] | None = None,
    thresholds: dict[str, Any] | None = None,
    tile_size: int = 512,
    overlap: int = 64,
) -> dict[str, Any]:
    """Calculate common spectral indices and write rasters, previews, stats, and report."""
    out_dir = prepare_output_dir(output_dir, task="spectral_indices")
    data, profile, info = read_raster(input_path)
    mapping = _resolve_band_map(info.count, band_mapping=band_mapping, band_map=band_map)
    requested = [item.lower() for item in (indices or ["ndvi"])]
    _validate_requested_indices(requested, data.shape[0], mapping)
    nodata = profile.get("nodata", info.nodata)
    bands = _load_bands(data, mapping, nodata)
    outputs: dict[str, str] = {}
    stats: dict[str, Any] = {
        "type": "spectral_indices",
        "requested_indices": requested,
        "computed_indices": [],
        "band_map": mapping,
        "pixel_area_m2": _pixel_area(info.transform),
        "indices": {},
    }
    quality_flags: list[dict[str, Any]] = []
    if info.count < 4 and not (band_mapping or band_map):
        quality_flags.append(flag("input_band_count_low", "Default spectral band mapping expects at least 4 bands.", "warning"))

    index_profile = dict(profile)
    index_profile.update(count=1, dtype="float32", nodata=np.nan)
    for name in requested:
        raw, denominator = _compute_index(name, bands)
        zero_count = int(np.count_nonzero(np.isfinite(denominator) & (np.abs(denominator) < 1e-6)))
        if zero_count:
            quality_flags.append(flag("division_by_zero_handled", f"{name.upper()} had {zero_count} near-zero denominator pixels.", "warning", index=name))
        index = raw.astype(np.float32)
        index[~np.isfinite(index)] = np.nan
        range_warning = _range_warning(name, index)
        if range_warning:
            quality_flags.append(range_warning)
        out_path = out_dir / f"{name}.tif"
        preview_path = out_dir / f"{name}_preview.png"
        out_info = write_raster(out_path, index, index_profile, dtype="float32", nodata=np.nan)
        outputs[f"{name}_geotiff"] = str(out_path)
        outputs[f"{name}_preview"] = _write_index_preview(preview_path, index)
        index_stats = _summarize(index, thresholds=(thresholds or {}).get(name), pixel_area=stats["pixel_area_m2"])
        stats["indices"][name] = index_stats
        stats[name] = index_stats
        stats["computed_indices"].append(name)
        if out_info.fallback_container:
            quality_flags.append(flag("fallback_raster_io", "Rasterio unavailable; fallback raster container was used.", "info"))

    stats_path = write_json(out_dir / "stats.json", stats)
    report_path = _write_report(out_dir / "report.md", input_path, stats, quality_flags, outputs)
    outputs["stats_json"] = stats_path
    outputs["report"] = report_path
    outputs["report_md"] = report_path
    return write_manifest(
        task="spectral_indices",
        output_dir=out_dir,
        inputs={"image": str(input_path), "raster": info.to_dict()},
        outputs=outputs,
        parameters={
            "indices": requested,
            "band_mapping": mapping,
            "band_map": mapping,
            "thresholds": thresholds or {},
            "tile_size": tile_size,
            "overlap": overlap,
        },
        stats=stats,
        quality_flags=quality_flags,
        model={"id": "spectral-index-calculator", "backend": "numpy", "framework": "numpy"},
    )


def _resolve_band_map(
    band_count: int,
    *,
    band_mapping: dict[str, int] | None,
    band_map: dict[str, int] | None,
) -> dict[str, int]:
    """Resolve user band mapping with 4-band defaults when possible."""
    provided = band_map or band_mapping
    if provided:
        normalized = {str(key).lower(): int(value) for key, value in provided.items()}
        unknown = sorted(set(normalized) - BAND_NAMES)
        if unknown:
            raise ValueError(f"Unsupported band_map keys: {', '.join(unknown)}")
        return normalized
    mapping = {"blue": 1, "green": 2, "red": 3, "nir": 4}
    if band_count >= 5:
        mapping["swir1"] = 5
    if band_count >= 6:
        mapping["swir2"] = 6
    return mapping


def _validate_requested_indices(requested: list[str], band_count: int, mapping: dict[str, int]) -> None:
    """Validate requested indices and their required bands."""
    unsupported = sorted(set(requested) - set(INDEX_REQUIREMENTS))
    if unsupported:
        raise ValueError(f"Unsupported spectral indices: {', '.join(unsupported)}")
    errors: list[str] = []
    for index in requested:
        missing = []
        for band_name in INDEX_REQUIREMENTS[index]:
            band_index = mapping.get(band_name)
            if band_index is None:
                missing.append(band_name)
            elif band_index < 1 or band_index > band_count:
                missing.append(f"{band_name}=band{band_index} outside 1..{band_count}")
        if missing:
            errors.append(f"{index}: missing/invalid {', '.join(missing)}")
    if errors:
        raise ValueError("Cannot compute requested spectral indices because required bands are unavailable: " + "; ".join(errors))


def _load_bands(data: np.ndarray, mapping: dict[str, int], nodata: float | int | None) -> dict[str, np.ndarray]:
    """Load mapped bands as float arrays with nodata converted to nan."""
    bands: dict[str, np.ndarray] = {}
    for name, index in mapping.items():
        if 1 <= index <= data.shape[0]:
            band = data[index - 1].astype(np.float32)
            if nodata is not None and np.isfinite(float(nodata)):
                band = band.copy()
                band[band == float(nodata)] = np.nan
            bands[name] = band
    return bands


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide with nan for near-zero denominators."""
    return numerator / np.where(np.abs(denominator) < 1e-6, np.nan, denominator)


def _compute_index(name: str, bands: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Compute one spectral index and return index plus denominator."""
    if name == "ndvi":
        denominator = bands["nir"] + bands["red"]
        return _safe_divide(bands["nir"] - bands["red"], denominator), denominator
    if name == "ndwi":
        denominator = bands["green"] + bands["nir"]
        return _safe_divide(bands["green"] - bands["nir"], denominator), denominator
    if name == "mndwi":
        denominator = bands["green"] + bands["swir1"]
        return _safe_divide(bands["green"] - bands["swir1"], denominator), denominator
    if name == "ndbi":
        denominator = bands["swir1"] + bands["nir"]
        return _safe_divide(bands["swir1"] - bands["nir"], denominator), denominator
    if name == "savi":
        soil_l = 0.5
        denominator = bands["nir"] + bands["red"] + soil_l
        return _safe_divide(bands["nir"] - bands["red"], denominator) * (1.0 + soil_l), denominator
    if name == "evi":
        denominator = bands["nir"] + 6.0 * bands["red"] - 7.5 * bands["blue"] + 1.0
        return 2.5 * _safe_divide(bands["nir"] - bands["red"], denominator), denominator
    raise ValueError(f"Unsupported spectral index: {name}")


def _summarize(index: np.ndarray, thresholds: Any, pixel_area: float) -> dict[str, Any]:
    """Summarize an index raster with percentiles and optional threshold areas."""
    finite = index[np.isfinite(index)]
    if finite.size == 0:
        summary: dict[str, Any] = {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "p5": 0.0, "p50": 0.0, "p95": 0.0, "valid_pixels": 0}
    else:
        summary = {
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "p5": float(np.percentile(finite, 5)),
            "p50": float(np.percentile(finite, 50)),
            "p95": float(np.percentile(finite, 95)),
            "valid_pixels": int(finite.size),
        }
    summary["threshold_areas"] = _threshold_areas(index, thresholds, pixel_area)
    return summary


def _threshold_areas(index: np.ndarray, thresholds: Any, pixel_area: float) -> dict[str, Any]:
    """Calculate optional threshold classification areas."""
    if thresholds is None:
        return {}
    rules: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
    if isinstance(thresholds, (int, float)):
        value = float(thresholds)
        rules[f">{value}"] = lambda arr, threshold=value: arr > threshold
    elif isinstance(thresholds, dict):
        for label, expr in thresholds.items():
            rules[str(label)] = _threshold_rule(expr)
    elif isinstance(thresholds, list):
        for expr in thresholds:
            rules[str(expr)] = _threshold_rule(expr)
    result: dict[str, Any] = {}
    for label, rule in rules.items():
        selected = rule(index) & np.isfinite(index)
        pixels = int(selected.sum())
        result[label] = {
            "pixel_count": pixels,
            "area_m2": pixels * pixel_area,
            "area_km2": pixels * pixel_area / 1_000_000.0,
        }
    return result


def _threshold_rule(expr: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Build a numpy threshold rule from a number or expression string."""
    if isinstance(expr, (int, float)):
        value = float(expr)
        return lambda arr, threshold=value: arr > threshold
    text = str(expr).strip()
    for op in (">=", "<=", ">", "<"):
        if text.startswith(op):
            value = float(text[len(op) :].strip())
            if op == ">=":
                return lambda arr, threshold=value: arr >= threshold
            if op == "<=":
                return lambda arr, threshold=value: arr <= threshold
            if op == ">":
                return lambda arr, threshold=value: arr > threshold
            return lambda arr, threshold=value: arr < threshold
    value = float(text)
    return lambda arr, threshold=value: arr > threshold


def _range_warning(name: str, index: np.ndarray) -> dict[str, Any] | None:
    """Return a quality warning when index values leave a broad reasonable range."""
    finite = index[np.isfinite(index)]
    if finite.size == 0:
        return flag("empty_index", f"{name.upper()} has no finite valid pixels.", "warning", index=name)
    min_value = float(np.min(finite))
    max_value = float(np.max(finite))
    lower, upper = (-1.5, 1.5) if name == "evi" else (-1.05, 1.05)
    if min_value < lower or max_value > upper:
        return flag(
            "index_value_out_of_range",
            f"{name.upper()} values are outside the expected range: min={min_value:.3f}, max={max_value:.3f}.",
            "warning",
            index=name,
            min=min_value,
            max=max_value,
        )
    return None


def _write_index_preview(path: Path, index: np.ndarray) -> str:
    """Write a color preview for an index raster."""
    if Image is None:
        path.write_text("Pillow is not installed; preview unavailable.", encoding="utf-8")
        return str(path)
    finite = index[np.isfinite(index)]
    if finite.size == 0:
        scaled = np.zeros(index.shape, dtype=np.uint8)
    else:
        clipped = np.clip(index, -1.0, 1.0)
        scaled = np.nan_to_num((clipped + 1.0) * 127.5, nan=0.0).astype(np.uint8)
    rgb = np.zeros((*scaled.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.where(scaled < 128, 255 - scaled * 2, 0).clip(0, 255)
    rgb[..., 1] = np.where(scaled >= 128, (scaled - 128) * 2, scaled * 2).clip(0, 255)
    rgb[..., 2] = np.where(scaled < 128, scaled * 2, 255 - (scaled - 128) * 2).clip(0, 255)
    Image.fromarray(rgb, mode="RGB").save(path)
    return str(path)


def _write_report(
    path: Path,
    input_path: str | Path,
    stats: dict[str, Any],
    quality_flags: list[dict[str, Any]],
    outputs: dict[str, str],
) -> str:
    """Write a deterministic Chinese markdown report for spectral indices."""
    lines = [
        "# 光谱指数分析报告",
        "",
        "## 数据概况",
        f"- 输入影像: `{input_path}`",
        f"- 像元面积: {stats['pixel_area_m2']:.6f} m2",
        f"- 波段映射: `{stats['band_map']}`",
        "",
        "## 指数统计",
    ]
    for name, summary in stats["indices"].items():
        lines.extend(
            [
                f"### {name.upper()}",
                f"- min/max: {summary['min']:.6f} / {summary['max']:.6f}",
                f"- mean/std: {summary['mean']:.6f} / {summary['std']:.6f}",
                f"- p5/p50/p95: {summary['p5']:.6f} / {summary['p50']:.6f} / {summary['p95']:.6f}",
            ]
        )
        for label, area in summary.get("threshold_areas", {}).items():
            lines.append(f"- 阈值 {label}: {area['area_m2']:.2f} m2 ({area['area_km2']:.6f} km2)")
    lines.extend(["", "## 质量检查"])
    if quality_flags:
        lines.extend(f"- [{item['severity']}] {item['code']}: {item['message']}" for item in quality_flags)
    else:
        lines.append("- 未发现明显质量风险。")
    lines.extend(["", "## 输出文件"])
    lines.extend(f"- {key}: `{value}`" for key, value in sorted(outputs.items()))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _pixel_area(transform: tuple[float, float, float, float, float, float]) -> float:
    """Calculate pixel area from affine transform."""
    return abs(float(transform[0]) * float(transform[4]) - float(transform[1]) * float(transform[3]))
