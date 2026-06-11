from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rs_service.adapters.base import ChangePrediction, ModelBackendUnavailable
from rs_service.analysis.change_summary import summarize_change_mask
from rs_service.core.alignment import align_raster_to_reference, check_pair_alignment, estimate_simple_alignment_warning
from rs_service.core.geometry import box_to_pixel_polygon, pixel_polygon_to_geojson_coords
from rs_service.core.manifest import write_json, write_manifest
from rs_service.core.raster import read_raster, write_raster
from rs_service.core.stitching import stitch_segmentation_tiles
from rs_service.core.tiling import Tile, iter_tiles
from rs_service.core.vector import write_geojson, write_gpkg
from rs_service.pipelines.base import flag, prepare_output_dir
from rs_service.registry import get_adapter

try:  # pragma: no cover - pillow is optional in minimal local runtime
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


def run_change_detection(
    before_path: str | Path,
    after_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    tile_size: int = 512,
    overlap: int = 64,
    model_id: str | None = None,
    threshold: float = 0.5,
    auto_align: bool = False,
) -> dict[str, Any]:
    out_dir = prepare_output_dir(output_dir, task="change_detection")
    before, profile, before_info = read_raster(before_path)
    after, _, after_info = read_raster(after_path)
    alignment = check_pair_alignment(before_info, after_info)
    alignment_warnings: list[str] = []
    aligned_after_path: str | None = None
    if not alignment["aligned"]:
        if not auto_align:
            raise ValueError(
                "before and after rasters are not aligned. "
                f"issues={alignment['issues']}. Set auto_align=True to resample image_t2 to image_t1 grid."
            )
        aligned_after_path = str(out_dir / "aligned_after.tif")
        after, _, after_info, alignment_warnings = align_raster_to_reference(after_path, before_path, aligned_after_path)
    else:
        alignment_warnings = estimate_simple_alignment_warning(before_info, after_info)
    if before.shape != after.shape:
        raise ValueError(f"before and after rasters must have identical shape after alignment, got {before.shape} and {after.shape}")
    adapter = get_adapter("change_detection", model_id=model_id)
    probability_tiles: list[np.ndarray] = []
    tiles: list[Tile] = []
    for tile in iter_tiles(before, tile_size=tile_size, overlap=overlap):
        after_tile = after[:, tile.y0 : tile.y1, tile.x0 : tile.x1]
        prediction = _predict_change_tile(adapter, tile, after_tile, before_info, threshold)
        probability_tiles.append(prediction.probability[np.newaxis, :, :])
        tiles.append(tile)
    if not probability_tiles:
        raise RuntimeError("No tiles were generated for change detection.")
    probability_map = stitch_segmentation_tiles(probability_tiles, tiles, (1, before_info.height, before_info.width))[0]
    mask = (probability_map >= threshold).astype(np.uint8)
    probability_path = out_dir / "change_probability.npy"
    np.save(probability_path, probability_map.astype(np.float32))
    mask_profile = dict(profile)
    mask_profile.update(count=1, dtype="uint8", nodata=0)
    mask_info = write_raster(out_dir / "change_mask.tif", mask, mask_profile, dtype="uint8", nodata=0)
    features = _change_features(mask, before_info.transform)
    geojson_path = write_geojson(out_dir / "changes.geojson", features, crs=before_info.crs)
    gpkg_path = write_gpkg(out_dir / "changes.gpkg", features, crs=before_info.crs, layer="changes")
    preview_path = _write_change_preview(out_dir / "preview.png", before, mask)
    pixel_area = abs(before_info.transform[0] * before_info.transform[4] - before_info.transform[1] * before_info.transform[3])
    stats = summarize_change_mask(out_dir / "change_mask.tif", pixel_area, alignment_warnings=alignment_warnings)
    stats.update(
        {
            "tile_count": len(tiles),
            "threshold": threshold,
            "probability_min": float(np.min(probability_map)),
            "probability_max": float(np.max(probability_map)),
            "alignment": alignment,
            "auto_align": auto_align,
        }
    )
    stats["changed_fraction"] = stats["changed_area_ratio"]
    stats_path = write_json(out_dir / "stats.json", stats)
    quality_flags = []
    for warning in alignment_warnings:
        quality_flags.append(flag(warning, f"Pair alignment warning: {warning}.", "warning"))
    if not alignment["aligned"] and auto_align:
        quality_flags.append(flag("auto_align_used", "image_t2 was resampled to image_t1 grid with MVP nearest-neighbor alignment.", "warning"))
    if mask_info.fallback_container:
        quality_flags.append(flag("fallback_raster_io", "Rasterio unavailable; fallback raster container was used.", "info"))

    outputs = {
        "mask_geotiff": str(out_dir / "change_mask.tif"),
        "probability_npy": str(probability_path),
        "geojson": geojson_path,
        "gpkg": gpkg_path,
        "preview": preview_path,
        "stats_json": stats_path,
    }
    if aligned_after_path:
        outputs["aligned_after"] = aligned_after_path

    return write_manifest(
        task="change_detection",
        output_dir=out_dir,
        inputs={
            "before": str(before_path),
            "after": str(after_path),
            "before_raster": before_info.to_dict(),
            "after_raster": after_info.to_dict(),
        },
        outputs=outputs,
        parameters={
            "tile_size": tile_size,
            "overlap": overlap,
            "threshold": threshold,
            "auto_align": auto_align,
        },
        stats=stats,
        quality_flags=quality_flags,
        model=adapter.metadata.to_dict(),
    )


def _predict_change_tile(
    adapter: Any,
    tile: Tile,
    after_tile: np.ndarray,
    raster_info: Any,
    threshold: float,
) -> ChangePrediction:
    """Run adapter prediction for one synchronized tile pair."""
    try:
        if hasattr(adapter, "predict_tile"):
            result = adapter.predict_tile(
                tile.data,
                {"tile_id": tile.tile_id, "x_off": tile.x0, "y_off": tile.y0},
                tile_t2=after_tile,
                after_tile=after_tile,
                threshold=threshold,
            )
            if isinstance(result, ChangePrediction):
                return result
        probability = adapter.predict_proba(tile.data, after_tile, context={"tile": tile, "raster": raster_info})
        return ChangePrediction(mask=(probability >= threshold).astype(np.uint8), probability=probability.astype(np.float32), threshold=threshold)
    except (ModelBackendUnavailable, FileNotFoundError, ImportError, RuntimeError) as exc:
        raise RuntimeError(str(exc)) from exc
    except Exception as exc:
        raise RuntimeError(f"Change detection tile inference failed for tile_id={tile.tile_id}: {exc}") from exc


def _change_features(
    mask: np.ndarray,
    transform: tuple[float, float, float, float, float, float],
) -> list[dict[str, Any]]:
    """Vectorize change mask components as simple bbox polygons."""
    features: list[dict[str, Any]] = []
    for index, component in enumerate(_component_boxes(mask > 0), start=1):
        x1, y1, x2, y2, pixel_count = component
        polygon = box_to_pixel_polygon([x1, y1, x2, y2])
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [pixel_polygon_to_geojson_coords(polygon, transform)],
                },
                "properties": {
                    "change_id": index,
                    "class_id": 1,
                    "class_name": "changed",
                    "pixel_count": pixel_count,
                    "bbox_pixel": [x1, y1, x2, y2],
                },
            }
        )
    return features


def _component_boxes(mask: np.ndarray) -> list[tuple[float, float, float, float, int]]:
    """Return connected component bbox and size using 4-neighbor fill."""
    binary = np.asarray(mask, dtype=bool)
    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    boxes: list[tuple[float, float, float, float, int]] = []
    for y in range(height):
        for x in range(width):
            if not binary[y, x] or visited[y, x]:
                continue
            stack = [(x, y)]
            visited[y, x] = True
            xs: list[int] = []
            ys: list[int] = []
            while stack:
                cx, cy = stack.pop()
                xs.append(cx)
                ys.append(cy)
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < width and 0 <= ny < height and binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((nx, ny))
            boxes.append((float(min(xs)), float(min(ys)), float(max(xs) + 1), float(max(ys) + 1), len(xs)))
    return boxes


def _write_change_preview(path: Path, before: np.ndarray, mask: np.ndarray) -> str:
    """Write an RGB preview with changes highlighted in red."""
    if Image is None:
        path.write_text("Pillow is not installed; preview unavailable.", encoding="utf-8")
        return str(path)
    base = _preview_rgb(before)
    overlay = base.copy()
    overlay[mask > 0] = np.asarray([255, 0, 0], dtype=np.uint8)
    rgb = (base.astype(np.float32) * 0.65 + overlay.astype(np.float32) * 0.35).clip(0, 255).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)
    return str(path)


def _preview_rgb(array: np.ndarray) -> np.ndarray:
    """Convert CHW raster data to uint8 RGB for preview."""
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
