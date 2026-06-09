from __future__ import annotations

from pathlib import Path
from typing import Any

from rs_service.core.manifest import read_json
from rs_service.core.raster import inspect_raster as inspect_raster_core
from rs_service.core.tiling import iter_windows
from rs_service.jobs import job_store
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
from rs_service.registry import list_models as registry_list_models


def inspect_raster(path: str) -> dict[str, Any]:
    return inspect_raster_core(path)


def preflight_plan(path: str, tile_size: int = 512, overlap: int = 64) -> dict[str, Any]:
    info = inspect_raster_core(path)
    windows = list(iter_windows(info["width"], info["height"], tile_size=tile_size, overlap=overlap))
    return {
        "input": path,
        "raster": info,
        "tile_size": tile_size,
        "overlap": overlap,
        "tile_count": len(windows),
        "windows": [
            {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
            for x0, y0, x1, y1 in windows[:20]
        ],
        "windows_truncated": len(windows) > 20,
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
) -> dict[str, Any]:
    return job_store.submit_sync(
        "object_detection",
        lambda: pipeline_object_detection(
            image_path,
            output_dir=output_dir,
            tile_size=tile_size,
            overlap=overlap,
            model_id=model_id,
            score_threshold=score_threshold,
        ),
    )


def run_oriented_detection(
    image_path: str,
    output_dir: str | None = None,
    tile_size: int = 512,
    overlap: int = 64,
    model_id: str | None = None,
    score_threshold: float = 0.0,
) -> dict[str, Any]:
    return job_store.submit_sync(
        "oriented_detection",
        lambda: pipeline_oriented_detection(
            image_path,
            output_dir=output_dir,
            tile_size=tile_size,
            overlap=overlap,
            model_id=model_id,
            score_threshold=score_threshold,
        ),
    )


def run_semantic_segmentation(
    image_path: str,
    output_dir: str | None = None,
    tile_size: int = 512,
    overlap: int = 64,
    model_id: str | None = None,
) -> dict[str, Any]:
    return job_store.submit_sync(
        "semantic_segmentation",
        lambda: pipeline_semantic_segmentation(
            image_path,
            output_dir=output_dir,
            tile_size=tile_size,
            overlap=overlap,
            model_id=model_id,
        ),
    )


def run_instance_segmentation(
    image_path: str,
    output_dir: str | None = None,
    tile_size: int = 512,
    overlap: int = 64,
    model_id: str | None = None,
    score_threshold: float = 0.0,
) -> dict[str, Any]:
    return job_store.submit_sync(
        "instance_segmentation",
        lambda: pipeline_instance_segmentation(
            image_path,
            output_dir=output_dir,
            tile_size=tile_size,
            overlap=overlap,
            model_id=model_id,
            score_threshold=score_threshold,
        ),
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
    return job_store.submit_sync(
        "change_detection",
        lambda: pipeline_change_detection(
            before_path,
            after_path,
            output_dir=output_dir,
            tile_size=tile_size,
            overlap=overlap,
            model_id=model_id,
            threshold=threshold,
        ),
    )


def run_super_resolution(
    image_path: str,
    output_dir: str | None = None,
    tile_size: int = 512,
    overlap: int = 64,
    scale: int = 2,
    model_id: str | None = None,
) -> dict[str, Any]:
    return job_store.submit_sync(
        "super_resolution",
        lambda: pipeline_super_resolution(
            image_path,
            output_dir=output_dir,
            tile_size=tile_size,
            overlap=overlap,
            scale=scale,
            model_id=model_id,
        ),
    )


def run_spectral_indices(
    image_path: str,
    output_dir: str | None = None,
    indices: list[str] | None = None,
    band_mapping: dict[str, int] | None = None,
    tile_size: int = 512,
    overlap: int = 64,
) -> dict[str, Any]:
    return job_store.submit_sync(
        "spectral_indices",
        lambda: pipeline_spectral_indices(
            image_path,
            output_dir=output_dir,
            indices=indices,
            band_mapping=band_mapping,
            tile_size=tile_size,
            overlap=overlap,
        ),
    )


def calculate_statistics(
    input_path: str | None = None,
    output_dir: str | None = None,
    manifest_path: str | None = None,
    zones_path: str | None = None,
) -> dict[str, Any]:
    return job_store.submit_sync(
        "statistics",
        lambda: pipeline_statistics(
            input_path,
            output_dir=output_dir,
            manifest_path=manifest_path,
            zones_path=zones_path,
        ),
    )


def quality_check_result(
    input_path: str | None = None,
    output_dir: str | None = None,
    manifest_path: str | None = None,
) -> dict[str, Any]:
    return job_store.submit_sync(
        "quality_check",
        lambda: pipeline_quality_check(input_path, output_dir=output_dir, manifest_path=manifest_path),
    )


def generate_report(
    manifest_path: str,
    output_dir: str | None = None,
    title: str = "Remote Sensing Processing Report",
) -> dict[str, Any]:
    return job_store.submit_sync(
        "report",
        lambda: pipeline_generate_report(manifest_path, output_dir=output_dir, title=title),
    )


def get_job_status(job_id: str) -> dict[str, Any]:
    return job_store.get(job_id)


def list_jobs() -> list[dict[str, Any]]:
    return job_store.list()


def get_result_manifest(job_id: str | None = None, manifest_path: str | None = None) -> dict[str, Any]:
    if manifest_path:
        return read_json(manifest_path)
    if not job_id:
        raise ValueError("job_id or manifest_path is required")
    status = job_store.get(job_id)
    path = status.get("manifest_path") or str(Path("workspace") / job_id / "manifest.json")
    if not Path(path).exists():
        raise FileNotFoundError(f"No manifest found for job_id={job_id!r}")
    return read_json(path)
