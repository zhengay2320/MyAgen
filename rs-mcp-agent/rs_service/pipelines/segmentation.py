from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rs_service.adapters.base import ModelBackendUnavailable, SegmentationPrediction
from rs_service.core.geometry import box_to_pixel_polygon, pixel_polygon_to_geojson_coords
from rs_service.core.manifest import write_json, write_manifest
from rs_service.core.raster import read_raster, write_raster
from rs_service.core.stitching import stitch_segmentation_tiles
from rs_service.core.tiling import Tile, iter_tiles
from rs_service.core.vector import write_geojson, write_gpkg
from rs_service.pipelines.base import flag, prepare_output_dir
from rs_service.registry import get_adapter

try:  # pragma: no cover - pillow is part of base deps but tests can run minimal
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


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
    tile_predictions: list[np.ndarray] = []
    tiles: list[Tile] = []
    class_names: dict[int, str] = {}
    class_count = 0

    for tile in iter_tiles(data, tile_size=tile_size, overlap=overlap):
        prediction = _predict_segmentation_tile(adapter, tile, info)
        class_names.update(prediction.class_names)
        proba = _prediction_to_probability(prediction)
        class_count = max(class_count, int(proba.shape[0]))
        tile_predictions = [_pad_channels(item, class_count) for item in tile_predictions]
        tile_predictions.append(_pad_channels(proba, class_count))
        tiles.append(tile)

    if not tile_predictions:
        raise RuntimeError("No tiles were generated for semantic segmentation.")
    probability = stitch_segmentation_tiles(tile_predictions, tiles, (class_count, info.height, info.width))
    mask = np.argmax(probability, axis=0).astype(np.uint8)
    probability_path = out_dir / "probabilities.npy"
    np.save(probability_path, probability.astype(np.float32))
    mask_profile = dict(profile)
    mask_profile.update(count=1, dtype="uint8", nodata=0)
    mask_info = write_raster(out_dir / "mask.tif", mask, mask_profile, dtype="uint8", nodata=0)
    preview_path = _write_preview(out_dir / "preview.png", mask)
    features = _mask_class_features(mask, info.transform, class_names)
    geojson_path = write_geojson(out_dir / "segments.geojson", features, crs=info.crs)
    gpkg_path = write_gpkg(out_dir / "segments.gpkg", features, crs=info.crs, layer="segments")
    labels, counts = np.unique(mask, return_counts=True)
    stats = {
        "tile_count": len(tiles),
        "class_count": class_count,
        "class_pixels": {str(int(label)): int(count) for label, count in zip(labels, counts)},
        "class_names": {str(key): value for key, value in class_names.items()},
        "probability_min": float(np.min(probability)),
        "probability_max": float(np.max(probability)),
    }
    stats_path = write_json(out_dir / "stats.json", stats)
    quality_flags = []
    if mask_info.fallback_container:
        quality_flags.append(flag("fallback_raster_io", "Rasterio unavailable; fallback raster container was used.", "info"))

    return write_manifest(
        task="semantic_segmentation",
        output_dir=out_dir,
        inputs={"image": str(input_path), "raster": info.to_dict()},
        outputs={
            "mask_geotiff": str(out_dir / "mask.tif"),
            "probability_npy": str(probability_path),
            "geojson": geojson_path,
            "gpkg": gpkg_path,
            "preview": preview_path,
            "stats_json": stats_path,
        },
        parameters={"tile_size": tile_size, "overlap": overlap},
        stats=stats,
        quality_flags=quality_flags,
        model=adapter.metadata.to_dict(),
    )


def _predict_segmentation_tile(adapter: Any, tile: Tile, raster_info: Any) -> SegmentationPrediction:
    """Run a segmentation adapter and normalize errors with tile context."""
    context = {"tile": tile, "raster": raster_info}
    try:
        if hasattr(adapter, "predict_tile"):
            result = adapter.predict_tile(tile.data, {"tile_id": tile.tile_id, "x_off": tile.x0, "y_off": tile.y0})
            if isinstance(result, SegmentationPrediction):
                return result
        proba = adapter.predict_proba(tile.data, context=context)
        mask = np.argmax(proba, axis=0).astype(np.uint8)
        return SegmentationPrediction(mask=mask, probabilities=proba)
    except (ModelBackendUnavailable, FileNotFoundError, ImportError, RuntimeError) as exc:
        raise RuntimeError(str(exc)) from exc
    except Exception as exc:
        raise RuntimeError(f"Semantic segmentation tile inference failed for tile_id={tile.tile_id}: {exc}") from exc


def _prediction_to_probability(prediction: SegmentationPrediction) -> np.ndarray:
    """Return CxHxW probabilities, converting hard masks to one-hot votes."""
    if prediction.probabilities is not None:
        proba = np.asarray(prediction.probabilities, dtype=np.float32)
        if proba.ndim != 3:
            raise ValueError(f"Expected CxHxW probabilities, got {proba.shape}.")
        return proba
    class_count = max(int(prediction.mask.max()) + 1, max(prediction.class_names.keys(), default=0) + 1)
    output = np.zeros((class_count, prediction.mask.shape[0], prediction.mask.shape[1]), dtype=np.float32)
    for class_id in range(class_count):
        output[class_id] = (prediction.mask == class_id).astype(np.float32)
    return output


def _pad_channels(array: np.ndarray, channels: int) -> np.ndarray:
    """Pad a CxHxW array to the requested number of channels."""
    if array.shape[0] >= channels:
        return array
    padding = np.zeros((channels - array.shape[0], array.shape[1], array.shape[2]), dtype=array.dtype)
    return np.concatenate([array, padding], axis=0)


def _mask_class_features(
    mask: np.ndarray,
    transform: tuple[float, float, float, float, float, float],
    class_names: dict[int, str],
) -> list[dict[str, Any]]:
    """Create simple per-class extent polygons from a semantic mask."""
    features: list[dict[str, Any]] = []
    for class_id in sorted(int(value) for value in np.unique(mask) if int(value) != 0):
        ys, xs = np.where(mask == class_id)
        if xs.size == 0:
            continue
        box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        polygon = box_to_pixel_polygon(box)
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [pixel_polygon_to_geojson_coords(polygon, transform)],
                },
                "properties": {
                    "class_id": class_id,
                    "class_name": class_names.get(class_id, f"class_{class_id}"),
                    "pixel_count": int(xs.size),
                    "bbox_pixel": box,
                },
            }
        )
    return features


def _write_preview(path: Path, mask: np.ndarray) -> str:
    """Write a small RGB preview for the segmentation mask."""
    if Image is None:
        path.write_text("Pillow is not installed; preview unavailable.", encoding="utf-8")
        return str(path)
    palette = np.asarray(
        [
            [0, 0, 0],
            [60, 180, 75],
            [230, 25, 75],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
        ],
        dtype=np.uint8,
    )
    rgb = palette[mask.astype(np.int64) % len(palette)]
    Image.fromarray(rgb, mode="RGB").save(path)
    return str(path)
