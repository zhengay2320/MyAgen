from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rs_service.core.manifest import write_manifest
from rs_service.core.raster import read_raster, write_raster
from rs_service.core.tiling import add_tile_prediction, empty_accumulator, finalize_accumulator, iter_tiles
from rs_service.pipelines.base import flag, prepare_output_dir
from rs_service.registry import get_adapter


def run_semantic_segmentation(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    tile_size: int = 512,
    overlap: int = 64,
    model_id: str | None = None,
) -> dict[str, Any]:
    out_dir = prepare_output_dir(output_dir, task="semantic_segmentation")
    data, profile, info = read_raster(input_path)
    adapter = get_adapter("semantic_segmentation", model_id=model_id)
    first_tile = next(iter_tiles(data, tile_size=tile_size, overlap=overlap))
    first_proba = adapter.predict_proba(first_tile.data, context={"tile": first_tile, "raster": info})
    class_count = int(first_proba.shape[0])
    accumulator, weights = empty_accumulator(info.height, info.width, channels=class_count)
    add_tile_prediction(accumulator, weights, first_proba, first_tile.x0, first_tile.y0)
    tile_count = 1

    for tile in iter_tiles(data, tile_size=tile_size, overlap=overlap):
        if tile.tile_id == first_tile.tile_id:
            continue
        tile_count += 1
        proba = adapter.predict_proba(tile.data, context={"tile": tile, "raster": info})
        add_tile_prediction(accumulator, weights, proba, tile.x0, tile.y0)

    probability = finalize_accumulator(accumulator, weights)
    mask = np.argmax(probability, axis=0).astype(np.uint8)
    probability_path = out_dir / "probabilities.npy"
    np.save(probability_path, probability.astype(np.float32))
    mask_profile = dict(profile)
    mask_profile.update(count=1, dtype="uint8", nodata=0)
    mask_info = write_raster(out_dir / "mask.tif", mask, mask_profile, dtype="uint8", nodata=0)
    labels, counts = np.unique(mask, return_counts=True)
    stats = {
        "tile_count": tile_count,
        "class_count": class_count,
        "class_pixels": {str(int(label)): int(count) for label, count in zip(labels, counts)},
        "probability_min": float(np.min(probability)),
        "probability_max": float(np.max(probability)),
    }
    quality_flags = []
    if mask_info.fallback_container:
        quality_flags.append(flag("fallback_raster_io", "Rasterio unavailable; fallback raster container was used.", "info"))

    return write_manifest(
        task="semantic_segmentation",
        output_dir=out_dir,
        inputs={"image": str(input_path), "raster": info.to_dict()},
        outputs={"mask_geotiff": str(out_dir / "mask.tif"), "probability_npy": str(probability_path)},
        parameters={"tile_size": tile_size, "overlap": overlap},
        stats=stats,
        quality_flags=quality_flags,
        model=adapter.metadata.to_dict(),
    )
