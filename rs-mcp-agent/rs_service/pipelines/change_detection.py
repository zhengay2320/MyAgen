from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rs_service.core.manifest import write_manifest
from rs_service.core.raster import read_raster, write_raster
from rs_service.core.tiling import add_tile_prediction, empty_accumulator, finalize_accumulator, iter_tiles
from rs_service.pipelines.base import flag, prepare_output_dir
from rs_service.registry import get_adapter


def run_change_detection(
    before_path: str | Path,
    after_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    tile_size: int = 512,
    overlap: int = 64,
    model_id: str | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    out_dir = prepare_output_dir(output_dir, task="change_detection")
    before, profile, before_info = read_raster(before_path)
    after, _, after_info = read_raster(after_path)
    if before.shape != after.shape:
        raise ValueError(f"before and after rasters must have identical shape, got {before.shape} and {after.shape}")
    adapter = get_adapter("change_detection", model_id=model_id)
    accumulator, weights = empty_accumulator(before_info.height, before_info.width, channels=1)
    tile_count = 0
    for tile in iter_tiles(before, tile_size=tile_size, overlap=overlap):
        tile_count += 1
        after_tile = after[:, tile.y0 : tile.y1, tile.x0 : tile.x1]
        probability = adapter.predict_proba(tile.data, after_tile, context={"tile": tile, "raster": before_info})
        add_tile_prediction(accumulator, weights, probability, tile.x0, tile.y0)
    probability_map = finalize_accumulator(accumulator, weights)[0]
    mask = (probability_map >= threshold).astype(np.uint8)
    probability_path = out_dir / "change_probability.npy"
    np.save(probability_path, probability_map.astype(np.float32))
    mask_profile = dict(profile)
    mask_profile.update(count=1, dtype="uint8", nodata=0)
    mask_info = write_raster(out_dir / "change_mask.tif", mask, mask_profile, dtype="uint8", nodata=0)
    changed_pixels = int(mask.sum())
    total_pixels = int(mask.size)
    stats = {
        "tile_count": tile_count,
        "threshold": threshold,
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "changed_fraction": changed_pixels / total_pixels if total_pixels else 0.0,
        "probability_min": float(np.min(probability_map)),
        "probability_max": float(np.max(probability_map)),
    }
    quality_flags = []
    if before_info.transform != after_info.transform or before_info.crs != after_info.crs:
        quality_flags.append(flag("georeference_mismatch", "Before and after rasters have different georeferencing.", "warning"))
    if mask_info.fallback_container:
        quality_flags.append(flag("fallback_raster_io", "Rasterio unavailable; fallback raster container was used.", "info"))

    return write_manifest(
        task="change_detection",
        output_dir=out_dir,
        inputs={
            "before": str(before_path),
            "after": str(after_path),
            "before_raster": before_info.to_dict(),
            "after_raster": after_info.to_dict(),
        },
        outputs={"mask_geotiff": str(out_dir / "change_mask.tif"), "probability_npy": str(probability_path)},
        parameters={"tile_size": tile_size, "overlap": overlap, "threshold": threshold},
        stats=stats,
        quality_flags=quality_flags,
        model=adapter.metadata.to_dict(),
    )
