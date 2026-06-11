from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rs_service.adapters.base import ModelBackendUnavailable, SuperResolutionPrediction
from rs_service.analysis.sr_metrics import reference_metrics
from rs_service.core.manifest import write_json, write_manifest
from rs_service.core.raster import read_raster, update_transform_for_super_resolution, write_raster
from rs_service.core.stitching import stitch_sr_tiles
from rs_service.core.tiling import Tile, iter_tiles
from rs_service.pipelines.base import flag, prepare_output_dir
from rs_service.registry import get_adapter

try:  # pragma: no cover - pillow is optional in minimal local runtime
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


def run_super_resolution(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    tile_size: int = 512,
    overlap: int = 64,
    scale: int = 2,
    model_id: str | None = None,
    reference_path: str | Path | None = None,
) -> dict[str, Any]:
    out_dir = prepare_output_dir(output_dir, task="super_resolution")
    data, profile, info = read_raster(input_path)
    adapter = get_adapter("super_resolution", model_id=model_id, scale=scale)
    out_height = info.height * scale
    out_width = info.width * scale
    sr_tiles: list[np.ndarray] = []
    tiles: list[Tile] = []
    for tile in iter_tiles(data, tile_size=tile_size, overlap=overlap):
        prediction = _predict_sr_tile(adapter, tile, info, scale)
        sr_tiles.append(prediction.image)
        tiles.append(tile)
    if not sr_tiles:
        raise RuntimeError("No tiles were generated for super resolution.")
    sr = stitch_sr_tiles(sr_tiles, tiles, (info.count, out_height, out_width), scale=scale)
    if np.issubdtype(data.dtype, np.integer):
        sr = np.clip(np.round(sr), np.iinfo(data.dtype).min, np.iinfo(data.dtype).max).astype(data.dtype)
    else:
        sr = sr.astype(data.dtype)
    sr_profile = dict(profile)
    sr_profile.update(
        height=out_height,
        width=out_width,
        count=info.count,
        dtype=str(sr.dtype),
        transform=update_transform_for_super_resolution(info.transform, scale),
    )
    sr_path = out_dir / "sr.tif"
    sr_info = write_raster(sr_path, sr, sr_profile, dtype=str(sr.dtype))
    preview_path = _write_sr_preview(out_dir / "preview.png", sr)
    stats = {
        "type": "super_resolution",
        "tile_count": len(tiles),
        "scale": scale,
        "input_width": info.width,
        "input_height": info.height,
        "output_width": sr_info.width,
        "output_height": sr_info.height,
        "input_transform": info.transform,
        "output_transform": sr_info.transform,
        "input_resolution": [abs(info.transform[0]), abs(info.transform[4])],
        "output_resolution": [abs(sr_info.transform[0]), abs(sr_info.transform[4])],
        "shape_matches_scale": sr_info.width == out_width and sr_info.height == out_height,
        "transform_matches_scale": sr_info.transform[0] == info.transform[0] / scale and sr_info.transform[4] == info.transform[4] / scale,
        "reference_path": str(reference_path) if reference_path else None,
        "reference_metrics_available": False,
    }
    quality_flags = []
    if sr_info.transform[0] != info.transform[0] / scale or sr_info.transform[4] != info.transform[4] / scale:
        quality_flags.append(flag("transform_scale_check_failed", "Output transform was not scaled as expected.", "error"))
    if sr_info.fallback_container:
        quality_flags.append(flag("fallback_raster_io", "Rasterio unavailable; fallback raster container was used.", "info"))
    metrics: dict[str, Any] = {}
    if reference_path:
        reference, _, reference_info = read_raster(reference_path)
        aligned_reference = _align_reference_to_sr(reference, out_height, out_width, sr.shape[0])
        metrics, metric_warnings = reference_metrics(aligned_reference, sr)
        stats["reference_metrics_available"] = True
        stats["reference_shape"] = [reference_info.height, reference_info.width]
        for warning in metric_warnings:
            quality_flags.append(flag("optional_metric_skipped", warning, "info"))
    else:
        quality_flags.append(flag("no_reference_image", "No reference image provided; PSNR/SSIM were not calculated.", "info"))
    stats_path = write_json(out_dir / "stats.json", stats)
    outputs = {
        "super_resolved_geotiff": str(sr_path),
        "sr_geotiff": str(sr_path),
        "preview": preview_path,
        "stats_json": stats_path,
    }
    if reference_path:
        outputs["reference"] = str(reference_path)

    return write_manifest(
        task="super_resolution",
        output_dir=out_dir,
        inputs={"image": str(input_path), "raster": info.to_dict()},
        outputs=outputs,
        parameters={"tile_size": tile_size, "overlap": overlap, "scale": scale, "reference_path": str(reference_path) if reference_path else None},
        stats=stats,
        metrics=metrics,
        quality_flags=quality_flags,
        conclusion="无参考图，无法定量评价重建精度。" if not reference_path else None,
        model=adapter.metadata.to_dict(),
    )


def _predict_sr_tile(adapter: Any, tile: Tile, raster_info: Any, scale: int) -> SuperResolutionPrediction:
    """Run an SR adapter and normalize backend errors."""
    context = {"tile": tile, "raster": raster_info}
    try:
        if hasattr(adapter, "predict_tile"):
            result = adapter.predict_tile(tile.data, {"tile_id": tile.tile_id, "x_off": tile.x0, "y_off": tile.y0}, scale=scale)
            if isinstance(result, SuperResolutionPrediction):
                return result
        image = adapter.upscale(tile.data, context=context)
        return SuperResolutionPrediction(image=image, scale=scale)
    except (ModelBackendUnavailable, FileNotFoundError, ImportError, RuntimeError) as exc:
        raise RuntimeError(str(exc)) from exc
    except Exception as exc:
        raise RuntimeError(f"Super-resolution tile inference failed for tile_id={tile.tile_id}: {exc}") from exc


def _align_reference_to_sr(reference: np.ndarray, height: int, width: int, bands: int) -> np.ndarray:
    """MVP nearest-neighbor alignment of reference to SR shape."""
    data = np.asarray(reference)
    y_idx = np.linspace(0, data.shape[1] - 1, height).round().astype(np.int64)
    x_idx = np.linspace(0, data.shape[2] - 1, width).round().astype(np.int64)
    aligned = data[:, y_idx][:, :, x_idx]
    if aligned.shape[0] > bands:
        return aligned[:bands]
    if aligned.shape[0] < bands:
        pad = np.repeat(aligned[-1:], bands - aligned.shape[0], axis=0)
        return np.concatenate([aligned, pad], axis=0)
    return aligned


def _write_sr_preview(path: Path, sr: np.ndarray) -> str:
    """Write an RGB preview for SR output."""
    if Image is None:
        path.write_text("Pillow is not installed; preview unavailable.", encoding="utf-8")
        return str(path)
    rgb = _preview_rgb(sr)
    Image.fromarray(rgb, mode="RGB").save(path)
    return str(path)


def _preview_rgb(array: np.ndarray) -> np.ndarray:
    """Convert CHW raster data to uint8 RGB preview."""
    data = np.asarray(array)
    if data.shape[0] == 1:
        rgb = np.repeat(data[:1], 3, axis=0)
    elif data.shape[0] >= 3:
        rgb = data[:3]
    else:
        rgb = np.concatenate([data, np.repeat(data[-1:], 3 - data.shape[0], axis=0)], axis=0)
    rgb = np.moveaxis(rgb, 0, -1).astype(np.float32)
    if rgb.max() > rgb.min():
        rgb = (rgb - rgb.min()) * 255.0 / (rgb.max() - rgb.min())
    return rgb.clip(0, 255).astype(np.uint8)
