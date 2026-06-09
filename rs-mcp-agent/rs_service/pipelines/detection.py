from __future__ import annotations

from pathlib import Path
from typing import Any

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
from rs_service.core.raster import read_raster
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
    return {
        "count": len(records),
        "tile_count": tile_count,
        "mean_score": round(sum(scores) / len(scores), 6) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "min_score": min(scores) if scores else 0.0,
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
    adapter = get_adapter(task, model_id=model_id)
    records: list[dict[str, Any]] = []
    tile_count = 0

    for tile in iter_tiles(data, tile_size=tile_size, overlap=overlap):
        tile_count += 1
        predictions = adapter.predict(tile.data, context={"tile": tile, "raster": info})
        for index, prediction in enumerate(predictions):
            score = float(prediction.get("score", 0.0))
            if score < score_threshold:
                continue
            label = str(prediction.get("label", "target"))
            if oriented:
                rotated = prediction["rotated_box"]
                cx = float(rotated["cx"]) + tile.x0
                cy = float(rotated["cy"]) + tile.y0
                pixel_polygon = rotated_box_to_pixel_polygon(
                    cx,
                    cy,
                    float(rotated["width"]),
                    float(rotated["height"]),
                    float(rotated["angle_degrees"]),
                )
                bbox_pixel = polygon_bounds(pixel_polygon)
                properties = {
                    "id": f"{tile.tile_id}_{index}",
                    "task": task,
                    "label": label,
                    "score": score,
                    "tile_id": tile.tile_id,
                    "cx_pixel": cx,
                    "cy_pixel": cy,
                    "width_pixel": float(rotated["width"]),
                    "height_pixel": float(rotated["height"]),
                    "angle_degrees": float(rotated["angle_degrees"]),
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
                    "task": task,
                    "label": label,
                    "score": score,
                    "tile_id": tile.tile_id,
                    "bbox_pixel": global_box,
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
    stats = _record_stats(records, tile_count)
    quality_flags = []
    if not records:
        quality_flags.append(flag("empty_result", "No detections survived thresholding.", "warning"))
    if info.fallback_container:
        quality_flags.append(flag("fallback_raster_io", "Rasterio unavailable; fallback raster container was used.", "info"))

    manifest = write_manifest(
        task=task,
        output_dir=out_dir,
        inputs={"image": str(input_path), "raster": info.to_dict()},
        outputs={"json": json_path, "geojson": geojson_path, "gpkg": gpkg_path},
        parameters={
            "tile_size": tile_size,
            "overlap": overlap,
            "score_threshold": score_threshold,
            "nms_threshold": nms_threshold,
        },
        stats=stats,
        quality_flags=quality_flags,
        model=adapter.metadata.to_dict(),
    )
    return manifest
