from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rs_service.core.manifest import write_manifest
from rs_service.core.raster import read_raster, write_raster
from rs_service.pipelines.base import flag, prepare_output_dir


DEFAULT_BANDS = {
    "blue": 1,
    "green": 2,
    "red": 3,
    "nir": 4,
    "swir1": 5,
}


def _band(data: np.ndarray, mapping: dict[str, int], name: str) -> np.ndarray | None:
    index = mapping.get(name)
    if index is None or index < 1 or index > data.shape[0]:
        return None
    return data[index - 1].astype(np.float32)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / np.where(np.abs(denominator) < 1e-6, np.nan, denominator)


def _summarize(index: np.ndarray) -> dict[str, float]:
    finite = index[np.isfinite(index)]
    if finite.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def run_spectral_indices(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    indices: list[str] | None = None,
    band_mapping: dict[str, int] | None = None,
    tile_size: int = 512,
    overlap: int = 64,
) -> dict[str, Any]:
    # Indices are full-raster array operations, but tile_size/overlap are accepted and recorded for API parity.
    out_dir = prepare_output_dir(output_dir, task="spectral_indices")
    data, profile, info = read_raster(input_path)
    mapping = dict(DEFAULT_BANDS)
    mapping.update(band_mapping or {})
    requested = [item.lower() for item in (indices or ["ndvi", "ndwi", "ndbi", "evi"])]
    outputs: dict[str, str] = {}
    stats: dict[str, Any] = {}
    quality_flags = []

    red = _band(data, mapping, "red")
    nir = _band(data, mapping, "nir")
    green = _band(data, mapping, "green")
    blue = _band(data, mapping, "blue")
    swir1 = _band(data, mapping, "swir1")
    computed: dict[str, np.ndarray] = {}

    if "ndvi" in requested and red is not None and nir is not None:
        computed["ndvi"] = _safe_divide(nir - red, nir + red)
    if "ndwi" in requested and green is not None and nir is not None:
        computed["ndwi"] = _safe_divide(green - nir, green + nir)
    if "ndbi" in requested and swir1 is not None and nir is not None:
        computed["ndbi"] = _safe_divide(swir1 - nir, swir1 + nir)
    if "evi" in requested and blue is not None and red is not None and nir is not None:
        computed["evi"] = 2.5 * _safe_divide(nir - red, nir + 6.0 * red - 7.5 * blue + 1.0)

    missing = sorted(set(requested) - set(computed))
    if missing:
        quality_flags.append(flag("missing_bands", f"Could not compute indices: {', '.join(missing)}.", "warning"))

    index_profile = dict(profile)
    index_profile.update(count=1, dtype="float32", nodata=np.nan)
    for name, array in computed.items():
        index = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        out_path = out_dir / f"{name}.tif"
        out_info = write_raster(out_path, index, index_profile, dtype="float32", nodata=np.nan)
        outputs[f"{name}_geotiff"] = str(out_path)
        stats[name] = _summarize(index)
        if out_info.fallback_container:
            quality_flags.append(flag("fallback_raster_io", "Rasterio unavailable; fallback raster container was used.", "info"))

    stats["requested_indices"] = requested
    stats["computed_indices"] = sorted(computed)
    return write_manifest(
        task="spectral_indices",
        output_dir=out_dir,
        inputs={"image": str(input_path), "raster": info.to_dict()},
        outputs=outputs,
        parameters={"indices": requested, "band_mapping": mapping, "tile_size": tile_size, "overlap": overlap},
        stats=stats,
        quality_flags=quality_flags,
        model={"id": "spectral-index-calculator", "backend": "numpy", "framework": "numpy"},
    )
