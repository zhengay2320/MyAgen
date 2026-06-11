from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rs_service.adapters.base import ModelBackendUnavailable
from rs_service.core.geometry import (
    box_to_pixel_polygon,
    clip_box,
    nms,
    pixel_polygon_to_geojson_coords,
    polygon_bounds,
    rotated_box_to_pixel_polygon,
    translate_box,
)
from rs_service.core.manifest import write_json, write_manifest
from rs_service.core.raster import read_raster, write_raster
from rs_service.core.tiling import iter_tiles
from rs_service.core.vector import write_geojson, write_gpkg
from rs_service.pipelines.base import flag, prepare_output_dir
from rs_service.registry import get_adapter


def _feature_from_polygon(
    pixel_polygon: list[tuple[float, float]],
    transform: tuple[float, float, float, float, float, float],
    properties: dict[str, Any],
) -> dict[str, Any]:
    geo_coords = pixel_polygon_to_geojson_coords(pixel_polygon, transform)
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [geo_coords]},
        "properties": properties,
    }


def _record_stats(records: list[dict[str, Any]], tile_count: int) -> dict[str, Any]:
    scores = [float(record.get("score", 0.0)) for record in records]
    labels = [str(record.get("label", "target")) for record in records]
    per_class_counts = {label: labels.count(label) for label in sorted(set(labels))}
    low_confidence_count = sum(1 for score in scores if score < 0.5)
    angles = [float(record["angle_degrees"]) for record in records if record.get("angle_degrees") is not None]
    return {
        "count": len(records),
        "tile_count": tile_count,
        "per_class_counts": per_class_counts,
        "confidence_mean": round(float(np.mean(scores)), 6) if scores else 0.0,
        "confidence_median": round(float(np.median(scores)), 6) if scores else 0.0,
        "low_confidence_count": low_confidence_count,
        "low_confidence_ratio": round(low_confidence_count / len(scores), 6) if scores else 0.0,
        "edge_object_count": sum(1 for record in records if record.get("touches_image_edge")),
        "edge_object_ratio": round(
            sum(1 for record in records if record.get("touches_image_edge")) / len(records), 6
        )
        if records
        else 0.0,
        "mean_score": round(sum(scores) / len(scores), 6) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "min_score": min(scores) if scores else 0.0,
        "angle_degrees_mean": round(float(np.mean(angles)), 6) if angles else None,
        "angle_degrees_median": round(float(np.median(angles)), 6) if angles else None,
        "angle_degrees_min": round(float(np.min(angles)), 6) if angles else None,
        "angle_degrees_max": round(float(np.max(angles)), 6) if angles else None,
    }


def run_object_detection(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    tile_size: int = 512,
    overlap: int = 64,
    model_id: str | None = None,
    score_threshold: float = 0.0,
    nms_threshold: float = 0.5,
) -> dict[str, Any]:
    return _run_detection(
        task="object_detection",
        input_path=input_path,
        output_dir=output_dir,
        tile_size=tile_size,
        overlap=overlap,
        model_id=model_id,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        oriented=False,
        instance=False,
    )


def run_oriented_detection(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    tile_size: int = 512,
    overlap: int = 64,
    model_id: str | None = None,
    score_threshold: float = 0.0,
) -> dict[str, Any]:
    return _run_detection(
        task="oriented_detection",
        input_path=input_path,
        output_dir=output_dir,
        tile_size=tile_size,
        overlap=overlap,
        model_id=model_id,
        score_threshold=score_threshold,
        nms_threshold=0.0,
        oriented=True,
        instance=False,
    )


def run_instance_segmentation(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    tile_size: int = 512,
    overlap: int = 64,
    model_id: str | None = None,
    score_threshold: float = 0.0,
    nms_threshold: float = 0.5,
) -> dict[str, Any]:
    return _run_detection(
        task="instance_segmentation",
        input_path=input_path,
        output_dir=output_dir,
        tile_size=tile_size,
        overlap=overlap,
        model_id=model_id,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        oriented=False,
        instance=True,
    )


def _run_detection(
    *,
    task: str,
    input_path: str | Path,
    output_dir: str | Path | None,
    tile_size: int,
    overlap: int,
    model_id: str | None,
    score_threshold: float,
    nms_threshold: float,
    oriented: bool,
    instance: bool,
) -> dict[str, Any]:
    out_dir = prepare_output_dir(output_dir, task=task)
    data, _, info = read_raster(input_path)
    adapter = get_adapter(
        task,
        model_id=model_id,
        confidence_threshold=max(score_threshold, 0.0),
        iou_threshold=nms_threshold,
    )
    records: list[dict[str, Any]] = []
    tile_count = 0
    adapter_kwargs = {
        "confidence_threshold": max(score_threshold, 0.0),
        "iou_threshold": nms_threshold,
    }

    for tile in iter_tiles(data, tile_size=tile_size, overlap=overlap):
        tile_count += 1
        predictions = _predict_tile(adapter, tile.data, {"tile": tile, "raster": info, "adapter_kwargs": adapter_kwargs})
        for index, prediction in enumerate(predictions):
            score = float(prediction.get("score", 0.0))
            if score < score_threshold:
                continue
            label = str(prediction.get("label", "target"))
            class_id = int(prediction.get("class_id", prediction.get("attributes", {}).get("class_id", 0)) or 0)
            if oriented:
                rotated = prediction.get("rotated_box") or {}
                if prediction.get("polygon"):
                    pixel_polygon = [
                        (float(x) + tile.x0, float(y) + tile.y0)
                        for x, y in prediction["polygon"]
                    ]
                    cx = float(rotated.get("cx", (polygon_bounds(pixel_polygon)[0] + polygon_bounds(pixel_polygon)[2]) / 2.0)) + (
                        tile.x0 if "cx" in rotated else 0.0
                    )
                    cy = float(rotated.get("cy", (polygon_bounds(pixel_polygon)[1] + polygon_bounds(pixel_polygon)[3]) / 2.0)) + (
                        tile.y0 if "cy" in rotated else 0.0
                    )
                    width_pixel = float(rotated.get("width", polygon_bounds(pixel_polygon)[2] - polygon_bounds(pixel_polygon)[0]))
                    height_pixel = float(rotated.get("height", polygon_bounds(pixel_polygon)[3] - polygon_bounds(pixel_polygon)[1]))
                    angle_degrees = float(rotated.get("angle_degrees", 0.0))
                else:
                    rotated = prediction["rotated_box"]
                    cx = float(rotated["cx"]) + tile.x0
                    cy = float(rotated["cy"]) + tile.y0
                    width_pixel = float(rotated["width"])
                    height_pixel = float(rotated["height"])
                    angle_degrees = float(rotated["angle_degrees"])
                    pixel_polygon = rotated_box_to_pixel_polygon(
                        cx,
                        cy,
                        width_pixel,
                        height_pixel,
                        angle_degrees,
                    )
                bbox_pixel = polygon_bounds(pixel_polygon)
                properties = {
                    "id": f"{tile.tile_id}_{index}",
                    "task": task,
                    "label": label,
                    "class_id": class_id,
                    "class_name": label,
                    "score": score,
                    "tile_id": tile.tile_id,
                    "cx_pixel": cx,
                    "cy_pixel": cy,
                    "width_pixel": width_pixel,
                    "height_pixel": height_pixel,
                    "angle_degrees": angle_degrees,
                    "bbox_pixel": bbox_pixel,
                }
            else:
                global_box = clip_box(translate_box(prediction["bbox"], tile.x0, tile.y0), info.width, info.height)
                if global_box[2] <= global_box[0] or global_box[3] <= global_box[1]:
                    continue
                if instance and prediction.get("mask_polygon"):
                    pixel_polygon = [
                        (float(x) + tile.x0, float(y) + tile.y0)
                        for x, y in prediction["mask_polygon"]
                    ]
                else:
                    pixel_polygon = box_to_pixel_polygon(global_box)
                properties = {
                    "id": f"{tile.tile_id}_{index}",
                    "instance_id": len(records) + 1 if instance else None,
                    "task": task,
                    "label": label,
                    "class_id": class_id,
                    "class_name": label,
                    "score": score,
                    "tile_id": tile.tile_id,
                    "bbox_pixel": global_box,
                    "area_pixels": _box_area(global_box),
                    "touches_image_edge": _touches_image_edge(global_box, info.width, info.height),
                }
                bbox_pixel = global_box

            feature = _feature_from_polygon(pixel_polygon, info.transform, properties)
            record = dict(properties)
            record["bbox_pixel"] = bbox_pixel
            record["geometry"] = feature["geometry"]
            records.append(record)

    if not oriented and nms_threshold > 0:
        records = nms(records, threshold=nms_threshold)

    features = [
        {
            "type": "Feature",
            "geometry": record["geometry"],
            "properties": {key: value for key, value in record.items() if key != "geometry"},
        }
        for record in records
    ]
    basename = {
        "object_detection": "detections",
        "oriented_detection": "oriented_detections",
        "instance_segmentation": "instances",
    }[task]
    json_path = write_json(out_dir / f"{basename}.json", {"records": records})
    geojson_path = write_geojson(out_dir / f"{basename}.geojson", features, crs=info.crs)
    gpkg_path = write_gpkg(out_dir / f"{basename}.gpkg", features, crs=info.crs, layer=basename)
    outputs = {"json": json_path, "geojson": geojson_path, "gpkg": gpkg_path}
    if instance:
        instance_mask = _instance_mask_from_records(records, info.height, info.width)
        mask_profile = {"crs": info.crs, "transform": info.transform, "driver": "GTiff"}
        mask_info = write_raster(out_dir / "instance_mask.tif", instance_mask, mask_profile, dtype="uint16", nodata=0)
        outputs["instance_mask_geotiff"] = str(out_dir / "instance_mask.tif")
        if mask_info.fallback_container:
            quality_flags_extra = flag("fallback_raster_io", "Rasterio unavailable; fallback raster container was used.", "info")
        else:
            quality_flags_extra = None
    else:
        quality_flags_extra = None
    stats = _record_stats(records, tile_count)
    quality_flags = []
    if not records:
        quality_flags.append(flag("empty_result", "No detections survived thresholding.", "warning"))
    if info.fallback_container:
        quality_flags.append(flag("fallback_raster_io", "Rasterio unavailable; fallback raster container was used.", "info"))
    if quality_flags_extra is not None:
        quality_flags.append(quality_flags_extra)

    manifest = write_manifest(
        task=task,
        output_dir=out_dir,
        inputs={"image": str(input_path), "raster": info.to_dict()},
        outputs=outputs,
        parameters={
            "tile_size": tile_size,
            "overlap": overlap,
            "score_threshold": score_threshold,
            "confidence_threshold": max(score_threshold, 0.0),
            "nms_threshold": nms_threshold,
        },
        stats=stats,
        quality_flags=quality_flags,
        model=adapter.metadata.to_dict(),
    )
    return manifest


def _predict_tile(adapter: Any, tile_data: Any, context: dict[str, Any]) -> list[dict[str, Any]]:
    """Run adapter prediction and convert optional backend failures to readable errors."""
    try:
        if hasattr(adapter, "predict"):
            return list(adapter.predict(tile_data, context=context))
        tile = context.get("tile")
        tile_info = {
            "tile_id": getattr(tile, "tile_id", None),
            "x_off": getattr(tile, "x0", 0),
            "y_off": getattr(tile, "y0", 0),
        }
        return [item.to_dict() for item in adapter.predict_tile(tile_data, tile_info, **context.get("adapter_kwargs", {}))]
    except (ModelBackendUnavailable, FileNotFoundError, ImportError, RuntimeError) as exc:
        raise RuntimeError(str(exc)) from exc


def _touches_image_edge(box: list[float], width: int, height: int, margin: float = 2.0) -> bool:
    """Return whether a box touches the full-image border."""
    x1, y1, x2, y2 = [float(v) for v in box]
    return x1 <= margin or y1 <= margin or x2 >= float(width) - margin or y2 >= float(height) - margin


def _box_area(box: list[float]) -> float:
    """Return bbox area in pixels."""
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _instance_mask_from_records(records: list[dict[str, Any]], height: int, width: int) -> np.ndarray:
    """Create a simple instance-id raster from full-image bbox records."""
    mask = np.zeros((height, width), dtype=np.uint16)
    for index, record in enumerate(records, start=1):
        box = record.get("bbox_pixel")
        if not box:
            continue
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
        mask[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)] = min(index, np.iinfo(np.uint16).max)
    return mask
