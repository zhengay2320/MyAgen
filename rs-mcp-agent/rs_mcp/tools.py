from __future__ import annotations

from typing import Any, Callable

from rs_service import services


def inspect_raster(path: str) -> dict[str, Any]:
    return services.inspect_raster(path)


def preflight_plan(path: str, tile_size: int = 512, overlap: int = 64) -> dict[str, Any]:
    return services.preflight_plan(path, tile_size=tile_size, overlap=overlap)


def list_models() -> dict[str, Any]:
    return services.list_models()


def run_object_detection(
    image_path: str,
    output_dir: str | None = None,
    tile_size: int = 512,
    overlap: int = 64,
    model_id: str | None = None,
    score_threshold: float = 0.0,
) -> dict[str, Any]:
    return services.run_object_detection(image_path, output_dir, tile_size, overlap, model_id, score_threshold)


def run_oriented_detection(
    image_path: str,
    output_dir: str | None = None,
    tile_size: int = 512,
    overlap: int = 64,
    model_id: str | None = None,
    score_threshold: float = 0.0,
) -> dict[str, Any]:
    return services.run_oriented_detection(image_path, output_dir, tile_size, overlap, model_id, score_threshold)


def run_semantic_segmentation(
    image_path: str,
    output_dir: str | None = None,
    tile_size: int = 512,
    overlap: int = 64,
    model_id: str | None = None,
) -> dict[str, Any]:
    return services.run_semantic_segmentation(image_path, output_dir, tile_size, overlap, model_id)


def run_instance_segmentation(
    image_path: str,
    output_dir: str | None = None,
    tile_size: int = 512,
    overlap: int = 64,
    model_id: str | None = None,
    score_threshold: float = 0.0,
) -> dict[str, Any]:
    return services.run_instance_segmentation(image_path, output_dir, tile_size, overlap, model_id, score_threshold)


def run_change_detection(
    before_path: str,
    after_path: str,
    output_dir: str | None = None,
    tile_size: int = 512,
    overlap: int = 64,
    model_id: str | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    return services.run_change_detection(before_path, after_path, output_dir, tile_size, overlap, model_id, threshold)


def run_super_resolution(
    image_path: str,
    output_dir: str | None = None,
    tile_size: int = 512,
    overlap: int = 64,
    scale: int = 2,
    model_id: str | None = None,
) -> dict[str, Any]:
    return services.run_super_resolution(image_path, output_dir, tile_size, overlap, scale, model_id)


def run_spectral_indices(
    image_path: str,
    output_dir: str | None = None,
    indices: list[str] | None = None,
    band_mapping: dict[str, int] | None = None,
    tile_size: int = 512,
    overlap: int = 64,
) -> dict[str, Any]:
    return services.run_spectral_indices(image_path, output_dir, indices, band_mapping, tile_size, overlap)


def calculate_statistics(
    input_path: str | None = None,
    output_dir: str | None = None,
    manifest_path: str | None = None,
    zones_path: str | None = None,
) -> dict[str, Any]:
    return services.calculate_statistics(input_path, output_dir, manifest_path, zones_path)


def quality_check_result(
    input_path: str | None = None,
    output_dir: str | None = None,
    manifest_path: str | None = None,
) -> dict[str, Any]:
    return services.quality_check_result(input_path, output_dir, manifest_path)


def generate_report(
    manifest_path: str,
    output_dir: str | None = None,
    title: str = "Remote Sensing Processing Report",
) -> dict[str, Any]:
    return services.generate_report(manifest_path, output_dir, title)


def get_job_status(job_id: str) -> dict[str, Any]:
    return services.get_job_status(job_id)


def get_result_manifest(job_id: str | None = None, manifest_path: str | None = None) -> dict[str, Any]:
    return services.get_result_manifest(job_id, manifest_path)


TOOL_FUNCTIONS: dict[str, Callable[..., dict[str, Any]]] = {
    "inspect_raster": inspect_raster,
    "preflight_plan": preflight_plan,
    "list_models": list_models,
    "run_object_detection": run_object_detection,
    "run_oriented_detection": run_oriented_detection,
    "run_semantic_segmentation": run_semantic_segmentation,
    "run_instance_segmentation": run_instance_segmentation,
    "run_change_detection": run_change_detection,
    "run_super_resolution": run_super_resolution,
    "run_spectral_indices": run_spectral_indices,
    "calculate_statistics": calculate_statistics,
    "quality_check_result": quality_check_result,
    "generate_report": generate_report,
    "get_job_status": get_job_status,
    "get_result_manifest": get_result_manifest,
}


def tool_names() -> list[str]:
    return sorted(TOOL_FUNCTIONS)


def _schema(properties: dict[str, dict[str, Any]], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required or []}


def tool_definitions() -> list[dict[str, Any]]:
    string = {"type": "string"}
    integer = {"type": "integer"}
    number = {"type": "number"}
    array_string = {"type": "array", "items": {"type": "string"}}
    object_schema = {"type": "object"}
    return [
        {"name": "inspect_raster", "description": "Inspect raster metadata, CRS, transform, bounds, and pixel size.", "inputSchema": _schema({"path": string}, ["path"])},
        {"name": "preflight_plan", "description": "Plan tiled processing for a raster before running inference.", "inputSchema": _schema({"path": string, "tile_size": integer, "overlap": integer}, ["path"])},
        {"name": "list_models", "description": "List fake stage-1 models and installed framework status.", "inputSchema": _schema({})},
        {"name": "run_object_detection", "description": "Run tiled object detection and export JSON, GeoJSON, GPKG, and manifest.", "inputSchema": _schema({"image_path": string, "output_dir": string, "tile_size": integer, "overlap": integer, "model_id": string, "score_threshold": number}, ["image_path"])},
        {"name": "run_oriented_detection", "description": "Run tiled rotated-box detection and export geospatial vectors.", "inputSchema": _schema({"image_path": string, "output_dir": string, "tile_size": integer, "overlap": integer, "model_id": string, "score_threshold": number}, ["image_path"])},
        {"name": "run_semantic_segmentation", "description": "Run tiled semantic segmentation with probability stitching and mask GeoTIFF output.", "inputSchema": _schema({"image_path": string, "output_dir": string, "tile_size": integer, "overlap": integer, "model_id": string}, ["image_path"])},
        {"name": "run_instance_segmentation", "description": "Run tiled instance segmentation and export geospatial polygons.", "inputSchema": _schema({"image_path": string, "output_dir": string, "tile_size": integer, "overlap": integer, "model_id": string, "score_threshold": number}, ["image_path"])},
        {"name": "run_change_detection", "description": "Run tiled bi-temporal change detection with probability stitching.", "inputSchema": _schema({"before_path": string, "after_path": string, "output_dir": string, "tile_size": integer, "overlap": integer, "model_id": string, "threshold": number}, ["before_path", "after_path"])},
        {"name": "run_super_resolution", "description": "Run tiled super resolution and update output GeoTIFF transform.", "inputSchema": _schema({"image_path": string, "output_dir": string, "tile_size": integer, "overlap": integer, "scale": integer, "model_id": string}, ["image_path"])},
        {"name": "run_spectral_indices", "description": "Calculate common spectral indices such as NDVI, NDWI, NDBI, and EVI.", "inputSchema": _schema({"image_path": string, "output_dir": string, "indices": array_string, "band_mapping": object_schema, "tile_size": integer, "overlap": integer}, ["image_path"])},
        {"name": "calculate_statistics", "description": "Generate stats.json and a statistics manifest for rasters or vector outputs.", "inputSchema": _schema({"input_path": string, "output_dir": string, "manifest_path": string, "zones_path": string})},
        {"name": "quality_check_result", "description": "Run rule-based output quality checks and write quality.json.", "inputSchema": _schema({"input_path": string, "output_dir": string, "manifest_path": string})},
        {"name": "generate_report", "description": "Generate report.md from a result manifest.", "inputSchema": _schema({"manifest_path": string, "output_dir": string, "title": string}, ["manifest_path"])},
        {"name": "get_job_status", "description": "Get an in-memory or workspace job status.", "inputSchema": _schema({"job_id": string}, ["job_id"])},
        {"name": "get_result_manifest", "description": "Load a result manifest by job_id or manifest_path.", "inputSchema": _schema({"job_id": string, "manifest_path": string})},
    ]
