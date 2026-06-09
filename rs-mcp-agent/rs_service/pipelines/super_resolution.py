from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rs_service.core.manifest import write_manifest
from rs_service.core.raster import read_raster, update_transform_for_super_resolution, write_raster
from rs_service.core.tiling import add_tile_prediction, empty_accumulator, finalize_accumulator, iter_tiles
from rs_service.pipelines.base import flag, prepare_output_dir
from rs_service.registry import get_adapter


def run_super_resolution(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    tile_size: int = 512,
    overlap: int = 64,
    scale: int = 2,
    model_id: str | None = None,
) -> dict[str, Any]:
    out_dir = prepare_output_dir(output_dir, task="super_resolution")
    data, profile, info = read_raster(input_path)
    adapter = get_adapter("super_resolution", model_id=model_id, scale=scale)
    out_height = info.height * scale
    out_width = info.width * scale
    accumulator, weights = empty_accumulator(out_height, out_width, channels=info.count)
    tile_count = 0
    for tile in iter_tiles(data, tile_size=tile_size, overlap=overlap):
        tile_count += 1
        sr_tile = adapter.upscale(tile.data, context={"tile": tile, "raster": info})
        add_tile_prediction(accumulator, weights, sr_tile, tile.x0 * scale, tile.y0 * scale)
    sr = finalize_accumulator(accumulator, weights)
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
    sr_info = write_raster(out_dir / "super_resolved.tif", sr, sr_profile, dtype=str(sr.dtype))
    stats = {
        "tile_count": tile_count,
        "scale": scale,
        "input_width": info.width,
        "input_height": info.height,
        "output_width": sr_info.width,
        "output_height": sr_info.height,
        "input_transform": info.transform,
        "output_transform": sr_info.transform,
    }
    quality_flags = []
    if sr_info.transform[0] != info.transform[0] / scale or sr_info.transform[4] != info.transform[4] / scale:
        quality_flags.append(flag("transform_scale_check_failed", "Output transform was not scaled as expected.", "error"))
    if sr_info.fallback_container:
        quality_flags.append(flag("fallback_raster_io", "Rasterio unavailable; fallback raster container was used.", "info"))

    return write_manifest(
        task="super_resolution",
        output_dir=out_dir,
        inputs={"image": str(input_path), "raster": info.to_dict()},
        outputs={"super_resolved_geotiff": str(out_dir / "super_resolved.tif")},
        parameters={"tile_size": tile_size, "overlap": overlap, "scale": scale},
        stats=stats,
        quality_flags=quality_flags,
        model=adapter.metadata.to_dict(),
    )
