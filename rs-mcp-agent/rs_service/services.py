from __future__ import annotations

from pathlib import Path
from typing import Any

from rs_service.analysis.conclusion_engine import build_conclusion
from rs_service.analysis.quality_checks import run_quality_checks
from rs_service.analysis.report_builder import build_report_markdown
from rs_service.analysis.statistics import (
    change_statistics,
    detection_statistics,
    instance_statistics,
    segmentation_statistics,
    spectral_index_statistics,
    super_resolution_statistics,
)
from rs_service.core.manifest import read_json, write_json
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
    auto_align: bool = False,
) -> dict[str, Any]:
    resolved_model_id = _resolve_model_id("change_detection", model_id)
    parameters = {
        "tile_size": tile_size,
        "overlap": overlap,
        "model_id": resolved_model_id,
        "threshold": threshold,
        "auto_align": auto_align,
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
            auto_align=auto_align,
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
    reference_path: str | None = None,
) -> dict[str, Any]:
    resolved_model_id = _resolve_model_id("super_resolution", model_id)
    parameters = {
        "tile_size": tile_size,
        "overlap": overlap,
        "scale": scale,
        "model_id": resolved_model_id,
        "reference_path": reference_path,
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
            reference_path=reference_path,
        ),
        model_id=resolved_model_id,
        input_files=[path for path in [image_path, reference_path] if path],
        parameters=parameters,
    )


def run_spectral_indices(
    image_path: str,
    output_dir: str | None = None,
    indices: list[str] | None = None,
    band_mapping: dict[str, int] | None = None,
    band_map: dict[str, int] | None = None,
    thresholds: dict[str, Any] | None = None,
    tile_size: int = 512,
    overlap: int = 64,
) -> dict[str, Any]:
    resolved_band_map = band_map or band_mapping
    parameters = {
        "indices": indices,
        "band_mapping": resolved_band_map,
        "band_map": resolved_band_map,
        "thresholds": thresholds,
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
            band_mapping=resolved_band_map,
            band_map=resolved_band_map,
            thresholds=thresholds,
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


def analyze_job(job_id: str, output_dir: str | None = None, zones_path: str | None = None) -> dict[str, Any]:
    """Analyze a completed job and update its original manifest."""
    manifest = get_result_manifest(job_id=job_id)
    manifest_path = Path(manifest["manifest_path"])
    out_dir = Path(output_dir) if output_dir else manifest_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    statistics = _statistics_for_manifest(manifest)
    quality_flags = run_quality_checks(manifest, statistics)
    conclusion = build_conclusion(manifest, statistics, quality_flags)
    stats_path = write_json(out_dir / "stats.json", statistics)
    quality_path = write_json(out_dir / "quality.json", {"quality_flags": quality_flags})
    manifest["statistics"] = statistics
    manifest["stats"] = statistics
    manifest["quality_flags"] = quality_flags
    manifest["conclusion"] = conclusion
    manifest.setdefault("outputs", {})
    manifest["outputs"]["stats_json"] = stats_path
    manifest["outputs"]["quality_json"] = quality_path
    write_json(manifest_path, manifest)
    _update_job_record_from_manifest(manifest)
    return manifest


def generate_job_report(job_id: str, output_dir: str | None = None, title: str = "Remote Sensing Processing Report") -> dict[str, Any]:
    """Generate a Chinese markdown report and update the source manifest."""
    manifest = get_result_manifest(job_id=job_id)
    if not isinstance(manifest.get("conclusion"), dict) or not manifest.get("statistics"):
        manifest = analyze_job(job_id)
    manifest_path = Path(manifest["manifest_path"])
    out_dir = Path(output_dir) if output_dir else manifest_path.parent
    report_path = out_dir / "report.md"
    report = build_report_markdown(manifest, report_path)
    manifest["outputs"]["report"] = report
    manifest["outputs"]["report_md"] = report
    write_json(manifest_path, manifest)
    _update_job_record_from_manifest(manifest)
    return manifest


def _statistics_for_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Route a manifest to the matching analysis function."""
    task = manifest.get("task")
    outputs = manifest.get("outputs", {})
    pixel_area = _pixel_area_from_manifest(manifest)
    if task == "semantic_segmentation":
        return segmentation_statistics(_required_output(outputs, "mask_geotiff"), "configs/class_maps/landcover.yaml", pixel_area)
    if task in {"object_detection", "oriented_detection"}:
        return detection_statistics(_required_output(outputs, "geojson"))
    if task == "instance_segmentation":
        return instance_statistics(_required_output(outputs, "geojson"))
    if task == "change_detection":
        return change_statistics(_required_output(outputs, "mask_geotiff"), pixel_area)
    if task == "super_resolution":
        input_path = _first_input_file(manifest)
        return super_resolution_statistics(input_path, _required_output(outputs, "super_resolved_geotiff"), int(manifest.get("parameters", {}).get("scale", 2)))
    if task == "spectral_indices":
        index_outputs = [value for key, value in outputs.items() if key.endswith("_geotiff")]
        if not index_outputs:
            raise FileNotFoundError("No spectral index GeoTIFF output found in manifest.")
        stats_by_index = {Path(path).stem: spectral_index_statistics(path) for path in index_outputs}
        return {"type": "spectral_indices", "indices": stats_by_index}
    if task == "statistics":
        return manifest.get("statistics", manifest.get("stats", {}))
    raise ValueError(f"Unsupported analysis task: {task}")


def _required_output(outputs: dict[str, Any], key: str) -> str:
    """Return a required output path or raise a clear error."""
    value = outputs.get(key)
    if not value:
        raise FileNotFoundError(f"Manifest is missing required output: {key}")
    if not Path(str(value)).exists():
        raise FileNotFoundError(f"Missing output file for {key}: {value}")
    return str(value)


def _first_input_file(manifest: dict[str, Any]) -> str:
    """Return the first input file from a manifest."""
    files = manifest.get("input_files") or []
    if files:
        return str(files[0])
    inputs = manifest.get("inputs", {})
    for value in inputs.values():
        if isinstance(value, str):
            return value
    raise FileNotFoundError("Manifest has no input file path.")


def _pixel_area_from_manifest(manifest: dict[str, Any]) -> float:
    """Estimate pixel area from raster metadata in a manifest."""
    for value in manifest.get("inputs", {}).values():
        if isinstance(value, dict) and "transform" in value:
            transform = value["transform"]
            return abs(float(transform[0]) * float(transform[4]) - float(transform[1]) * float(transform[3]))
    return 1.0


def _update_job_record_from_manifest(manifest: dict[str, Any]) -> None:
    """Best-effort sync from manifest analysis fields back to the local job store."""
    try:
        job_store.update_job(
            manifest["job_id"],
            manifest_path=manifest.get("manifest_path"),
            outputs=manifest.get("outputs", {}),
            statistics=manifest.get("statistics", {}),
            quality_flags=manifest.get("quality_flags", []),
        )
    except Exception:
        return
