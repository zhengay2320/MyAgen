from __future__ import annotations

from pathlib import Path
from typing import Any

from rs_service.core.manifest import read_json
from rs_service.core.raster import inspect_raster as inspect_raster_core
from rs_service.core.tiling import preflight_plan as build_tiling_preflight_plan
from rs_service.job_store import job_store
from rs_service.pipelines.change_detection import run_change_detection as pipeline_change_detection
from rs_service.pipelines.detection import (
    run_instance_segmentation as pipeline_instance_segmentation,
    run_object_detection as pipeline_object_detection,
    run_oriented_detection as pipeline_oriented_detection,
)
from rs_service.pipelines.quality import quality_check_result as pipeline_quality_check
from rs_service.pipelines.report import generate_report as pipeline_generate_report
from rs_service.pipelines.segmentation import run_semantic_segmentation as pipeline_semantic_segmentation
from rs_service.pipelines.spectral_indices import run_spectral_indices as pipeline_spectral_indices
from rs_service.pipelines.statistics import calculate_statistics as pipeline_statistics
from rs_service.pipelines.super_resolution import run_super_resolution as pipeline_super_resolution
from rs_service.registry import TASK_MODEL_IDS, list_models as registry_list_models


def _resolve_model_id(task: str, model_id: str | None) -> str | None:
    return model_id or TASK_MODEL_IDS.get(task)


def inspect_raster(path: str) -> dict[str, Any]:
    return inspect_raster_core(path)


def preflight_plan(path: str, tile_size: int | None = None, overlap: int | None = None, task: str = "detection") -> dict[str, Any]:
    info = inspect_raster_core(path)
    plan = build_tiling_preflight_plan(
        width=info["width"],
        height=info["height"],
        task=task,
        tile_size=tile_size,
        overlap=overlap,
    )
    return {
        "input": path,
        "raster": info,
        "task": task,
        "tile_size": plan["tile_size"],
        "overlap": plan["overlap"],
        "tile_count": plan["tile_count"],
        "tiles": plan["tiles"],
        "windows": [
            {
                "x0": tile["x_off"],
                "y0": tile["y_off"],
                "x1": tile["x_off"] + tile["width"],
                "y1": tile["y_off"] + tile["height"],
            }
            for tile in plan["tiles"][:20]
        ],
        "windows_truncated": len(plan["tiles"]) > 20,
        "outputs": {
            "raster": ["GeoTIFF"],
            "vector": ["GeoJSON", "GPKG"],
            "metadata": ["manifest.json", "stats.json", "quality.json", "report.md"],
        },
    }


def list_models() -> dict[str, Any]:
    return registry_list_models()


def run_object_detection(
    image_path: str,
    output_dir: str | None = None,
    tile_size: int = 512,
    overlap: int = 64,
    model_id: str | None = None,
    score_threshold: float = 0.0,
    nms_threshold: float = 0.5,
) -> dict[str, Any]:
    resolved_model_id = _resolve_model_id("object_detection", model_id)
    parameters = {
        "tile_size": tile_size,
        "overlap": overlap,
        "model_id": resolved_model_id,
        "score_threshold": score_threshold,
        "nms_threshold": nms_threshold,
        "requested_output_dir": output_dir,
    }
    return job_store.submit_sync(
        "object_detection",
        lambda job_output_dir: pipeline_object_detection(
            image_path,
            output_dir=job_output_dir,
            tile_size=tile_size,
            overlap=overlap,
            model_id=resolved_model_id,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
        ),
        model_id=resolved_model_id,
        input_files=[image_path],
        parameters=parameters,
    )


def run_oriented_detection(
    image_path: str,
    output_dir: str | None = None,
    tile_size: int = 512,
    overlap: int = 64,
    model_id: str | None = None,
    score_threshold: float = 0.0,
) -> dict[str, Any]:
    resolved_model_id = _resolve_model_id("oriented_detection", model_id)
    parameters = {
        "tile_size": tile_size,
        "overlap": overlap,
        "model_id": resolved_model_id,
        "score_threshold": score_threshold,
        "requested_output_dir": output_dir,
    }
    return job_store.submit_sync(
        "oriented_detection",
        lambda job_output_dir: pipeline_oriented_detection(
            image_path,
            output_dir=job_output_dir,
            tile_size=tile_size,
            overlap=overlap,
            model_id=resolved_model_id,
            score_threshold=score_threshold,
        ),
        model_id=resolved_model_id,
        input_files=[image_path],
        parameters=parameters,
    )


def run_semantic_segmentation(
    image_path: str,
    output_dir: str | None = None,
    tile_size: int = 512,
    overlap: int = 64,
    model_id: str | None = None,
) -> dict[str, Any]:
    resolved_model_id = _resolve_model_id("semantic_segmentation", model_id)
    parameters = {
        "tile_size": tile_size,
        "overlap": overlap,
        "model_id": resolved_model_id,
        "requested_output_dir": output_dir,
    }
    return job_store.submit_sync(
        "semantic_segmentation",
        lambda job_output_dir: pipeline_semantic_segmentation(
            image_path,
            output_dir=job_output_dir,
            tile_size=tile_size,
            overlap=overlap,
            model_id=resolved_model_id,
        ),
        model_id=resolved_model_id,
        input_files=[image_path],
        parameters=parameters,
    )


def run_instance_segmentation(
    image_path: str,
    output_dir: str | None = None,
    tile_size: int = 512,
    overlap: int = 64,
    model_id: str | None = None,
    score_threshold: float = 0.0,
    nms_threshold: float = 0.5,
) -> dict[str, Any]:
    resolved_model_id = _resolve_model_id("instance_segmentation", model_id)
    parameters = {
        "tile_size": tile_size,
        "overlap": overlap,
        "model_id": resolved_model_id,
        "score_threshold": score_threshold,
        "nms_threshold": nms_threshold,
        "requested_output_dir": output_dir,
    }
    return job_store.submit_sync(
        "instance_segmentation",
        lambda job_output_dir: pipeline_instance_segmentation(
            image_path,
            output_dir=job_output_dir,
            tile_size=tile_size,
            overlap=overlap,
            model_id=resolved_model_id,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
        ),
        model_id=resolved_model_id,
        input_files=[image_path],
        parameters=parameters,
    )


def run_change_detection(
    before_path: str,
    after_path: str,
    output_dir: str | None = None,
    tile_size: int = 512,
    overlap: int = 64,
    model_id: str | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    resolved_model_id = _resolve_model_id("change_detection", model_id)
    parameters = {
        "tile_size": tile_size,
        "overlap": overlap,
        "model_id": resolved_model_id,
        "threshold": threshold,
        "requested_output_dir": output_dir,
    }
    return job_store.submit_sync(
        "change_detection",
        lambda job_output_dir: pipeline_change_detection(
            before_path,
            after_path,
            output_dir=job_output_dir,
            tile_size=tile_size,
            overlap=overlap,
            model_id=resolved_model_id,
            threshold=threshold,
        ),
        model_id=resolved_model_id,
        input_files=[before_path, after_path],
        parameters=parameters,
    )


def run_super_resolution(
    image_path: str,
    output_dir: str | None = None,
    tile_size: int = 512,
    overlap: int = 64,
    scale: int = 2,
    model_id: str | None = None,
) -> dict[str, Any]:
    resolved_model_id = _resolve_model_id("super_resolution", model_id)
    parameters = {
        "tile_size": tile_size,
        "overlap": overlap,
        "scale": scale,
        "model_id": resolved_model_id,
        "requested_output_dir": output_dir,
    }
    return job_store.submit_sync(
        "super_resolution",
        lambda job_output_dir: pipeline_super_resolution(
            image_path,
            output_dir=job_output_dir,
            tile_size=tile_size,
            overlap=overlap,
            scale=scale,
            model_id=resolved_model_id,
        ),
        model_id=resolved_model_id,
        input_files=[image_path],
        parameters=parameters,
    )


def run_spectral_indices(
    image_path: str,
    output_dir: str | None = None,
    indices: list[str] | None = None,
    band_mapping: dict[str, int] | None = None,
    tile_size: int = 512,
    overlap: int = 64,
) -> dict[str, Any]:
    parameters = {
        "indices": indices,
        "band_mapping": band_mapping,
        "tile_size": tile_size,
        "overlap": overlap,
        "requested_output_dir": output_dir,
    }
    return job_store.submit_sync(
        "spectral_indices",
        lambda job_output_dir: pipeline_spectral_indices(
            image_path,
            output_dir=job_output_dir,
            indices=indices,
            band_mapping=band_mapping,
            tile_size=tile_size,
            overlap=overlap,
        ),
        model_id="spectral-index-calculator",
        input_files=[image_path],
        parameters=parameters,
    )


def calculate_statistics(
    input_path: str | None = None,
    output_dir: str | None = None,
    manifest_path: str | None = None,
    zones_path: str | None = None,
) -> dict[str, Any]:
    input_files = [path for path in [input_path, manifest_path, zones_path] if path]
    parameters = {"manifest_path": manifest_path, "zones_path": zones_path, "requested_output_dir": output_dir}
    return job_store.submit_sync(
        "statistics",
        lambda job_output_dir: pipeline_statistics(
            input_path,
            output_dir=job_output_dir,
            manifest_path=manifest_path,
            zones_path=zones_path,
        ),
        model_id="statistics",
        input_files=input_files,
        parameters=parameters,
    )


def quality_check_result(
    input_path: str | None = None,
    output_dir: str | None = None,
    manifest_path: str | None = None,
) -> dict[str, Any]:
    input_files = [path for path in [input_path, manifest_path] if path]
    parameters = {"manifest_path": manifest_path, "requested_output_dir": output_dir}
    return job_store.submit_sync(
        "quality_check",
        lambda job_output_dir: pipeline_quality_check(input_path, output_dir=job_output_dir, manifest_path=manifest_path),
        model_id="quality-checker",
        input_files=input_files,
        parameters=parameters,
    )


def generate_report(
    manifest_path: str,
    output_dir: str | None = None,
    title: str = "Remote Sensing Processing Report",
) -> dict[str, Any]:
    parameters = {"title": title, "requested_output_dir": output_dir}
    return job_store.submit_sync(
        "report",
        lambda job_output_dir: pipeline_generate_report(manifest_path, output_dir=job_output_dir, title=title),
        model_id="markdown-report-generator",
        input_files=[manifest_path],
        parameters=parameters,
    )


def get_job_status(job_id: str) -> dict[str, Any]:
    record = job_store.get_job(job_id)
    if record is None:
        return {"job_id": job_id, "status": "not_found"}
    return record.to_dict()


def list_jobs() -> list[dict[str, Any]]:
    return [record.to_dict() for record in job_store.list_jobs()]


def get_result_manifest(job_id: str | None = None, manifest_path: str | None = None) -> dict[str, Any]:
    if manifest_path:
        return read_json(manifest_path)
    if not job_id:
        raise ValueError("job_id or manifest_path is required")
    status = get_job_status(job_id)
    path = status.get("manifest_path") or str(Path("workspace") / "outputs" / job_id / "manifest.json")
    if not Path(path).exists():
        raise FileNotFoundError(f"No manifest found for job_id={job_id!r}")
    return read_json(path)
