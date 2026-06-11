from __future__ import annotations

from typing import Any, Callable

from rs_mcp.client import RsServiceClient, unwrap_result


MODEL_ALIASES = {
    "fake_semantic_segmentation": "fake_segmentation",
    "fake_instance_segmentation": "fake_instance",
    "fake_change_detection": "fake_change",
}


def _client() -> RsServiceClient:
    """Create a service client from environment configuration."""
    return RsServiceClient()


def _clean_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop None values before sending JSON payloads to FastAPI."""
    return {key: _normalize_model_id(value) if key == "model_id" else value for key, value in payload.items() if value is not None}


def _normalize_model_id(model_id: Any) -> Any:
    """Translate user-facing fake model aliases to API registry IDs."""
    if isinstance(model_id, str):
        return MODEL_ALIASES.get(model_id, model_id)
    return model_id


def inspect_raster(image_path: str) -> dict[str, Any]:
    """Inspect a raster through the FastAPI backend."""
    return unwrap_result(_client().post("/inspect", {"path": image_path}))


def preflight_plan(image_path: str, task: str, model_id: str | None = None) -> dict[str, Any]:
    """Build a tiling preflight plan through the FastAPI backend."""
    return unwrap_result(_client().post("/preflight", _clean_payload({"image_path": image_path, "task": task, "model_id": model_id})))


def list_models() -> dict[str, Any]:
    """List models from the FastAPI backend."""
    return _client().get("/models")


def run_object_detection(
    image_path: str,
    model_id: str = "fake_detection",
    tile_size: int | None = None,
    overlap: int | None = None,
    confidence_threshold: float | None = None,
) -> dict[str, Any]:
    """Submit a fake or configured object detection job."""
    return _client().post(
        "/jobs/detection",
        _clean_payload(
            {
                "image_path": image_path,
                "model_id": model_id,
                "tile_size": tile_size,
                "overlap": overlap,
                "score_threshold": confidence_threshold,
            }
        ),
    )


def run_oriented_detection(
    image_path: str,
    model_id: str = "fake_oriented_detection",
    tile_size: int | None = None,
    overlap: int | None = None,
    confidence_threshold: float | None = None,
) -> dict[str, Any]:
    """Submit an oriented detection job."""
    return _client().post(
        "/jobs/oriented-detection",
        _clean_payload(
            {
                "image_path": image_path,
                "model_id": model_id,
                "tile_size": tile_size,
                "overlap": overlap,
                "score_threshold": confidence_threshold,
            }
        ),
    )


def run_semantic_segmentation(
    image_path: str,
    model_id: str = "fake_semantic_segmentation",
    tile_size: int | None = None,
    overlap: int | None = None,
) -> dict[str, Any]:
    """Submit a semantic segmentation job."""
    return _client().post(
        "/jobs/semantic-segmentation",
        _clean_payload({"image_path": image_path, "model_id": model_id, "tile_size": tile_size, "overlap": overlap}),
    )


def run_instance_segmentation(
    image_path: str,
    model_id: str = "fake_instance_segmentation",
    tile_size: int | None = None,
    overlap: int | None = None,
) -> dict[str, Any]:
    """Submit an instance segmentation job."""
    return _client().post(
        "/jobs/instance-segmentation",
        _clean_payload({"image_path": image_path, "model_id": model_id, "tile_size": tile_size, "overlap": overlap}),
    )


def run_change_detection(
    image_t1_path: str,
    image_t2_path: str,
    model_id: str = "fake_change",
    tile_size: int | None = None,
    overlap: int | None = None,
    auto_align: bool | None = None,
) -> dict[str, Any]:
    """Submit a bi-temporal change detection job."""
    return _client().post(
        "/jobs/change-detection",
        _clean_payload(
            {
                "before_path": image_t1_path,
                "after_path": image_t2_path,
                "model_id": model_id,
                "tile_size": tile_size,
                "overlap": overlap,
                "auto_align": auto_align,
            }
        ),
    )


def run_super_resolution(
    image_path: str,
    model_id: str = "fake_super_resolution",
    scale: int = 2,
    tile_size: int | None = None,
    overlap: int | None = None,
    reference_path: str | None = None,
) -> dict[str, Any]:
    """Submit a super-resolution job."""
    return _client().post(
        "/jobs/super-resolution",
        _clean_payload(
            {
                "image_path": image_path,
                "model_id": model_id,
                "scale": scale,
                "tile_size": tile_size,
                "overlap": overlap,
                "reference_path": reference_path,
            }
        ),
    )


def run_spectral_indices(image_path: str, indices: list[str] | None = None) -> dict[str, Any]:
    """Submit a spectral index calculation job."""
    return _client().post("/jobs/spectral-indices", _clean_payload({"image_path": image_path, "indices": indices or ["ndvi"]}))


def calculate_statistics(job_id: str) -> dict[str, Any]:
    """Run statistics analysis for an existing job."""
    return _client().post(f"/jobs/{job_id}/analyze", {})


def quality_check_result(job_id: str) -> dict[str, Any]:
    """Run quality checks for an existing job through the FastAPI backend."""
    return _client().post(f"/jobs/{job_id}/analyze", {})


def generate_report(job_id: str, output_format: str = "markdown") -> dict[str, Any]:
    """Generate a report for an existing job."""
    if output_format != "markdown":
        raise ValueError("Only markdown report output is supported in the MVP.")
    return _client().post(f"/jobs/{job_id}/report", {"title": "Remote Sensing Processing Report"})


def get_job_status(job_id: str) -> dict[str, Any]:
    """Get status for a FastAPI job."""
    return _client().get(f"/jobs/{job_id}")


def get_result_manifest(job_id: str) -> dict[str, Any]:
    """Fetch a job manifest from the FastAPI backend."""
    return _client().get(f"/jobs/{job_id}/manifest")


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
    """Return registered MCP tool names."""
    return sorted(TOOL_FUNCTIONS)


def _schema(properties: dict[str, dict[str, Any]], required: list[str] | None = None) -> dict[str, Any]:
    """Build a JSON schema object."""
    return {"type": "object", "properties": properties, "required": required or []}


def tool_definitions() -> list[dict[str, Any]]:
    """Return fallback JSON-RPC tool definitions."""
    string = {"type": "string"}
    integer = {"type": "integer"}
    number = {"type": "number"}
    array_string = {"type": "array", "items": {"type": "string"}}
    return [
        {"name": "inspect_raster", "description": "Inspect raster metadata through rs_service.", "inputSchema": _schema({"image_path": string}, ["image_path"])},
        {"name": "preflight_plan", "description": "Plan tiled processing before running a job.", "inputSchema": _schema({"image_path": string, "task": string, "model_id": string}, ["image_path", "task"])},
        {"name": "list_models", "description": "List models from rs_service.", "inputSchema": _schema({})},
        {"name": "run_object_detection", "description": "Submit an object detection job.", "inputSchema": _schema({"image_path": string, "model_id": string, "tile_size": integer, "overlap": integer, "confidence_threshold": number}, ["image_path"])},
        {"name": "run_oriented_detection", "description": "Submit an oriented detection job.", "inputSchema": _schema({"image_path": string, "model_id": string, "tile_size": integer, "overlap": integer, "confidence_threshold": number}, ["image_path"])},
        {"name": "run_semantic_segmentation", "description": "Submit a semantic segmentation job.", "inputSchema": _schema({"image_path": string, "model_id": string, "tile_size": integer, "overlap": integer}, ["image_path"])},
        {"name": "run_instance_segmentation", "description": "Submit an instance segmentation job.", "inputSchema": _schema({"image_path": string, "model_id": string, "tile_size": integer, "overlap": integer}, ["image_path"])},
        {"name": "run_change_detection", "description": "Submit a change detection job.", "inputSchema": _schema({"image_t1_path": string, "image_t2_path": string, "model_id": string, "tile_size": integer, "overlap": integer, "auto_align": {"type": "boolean"}}, ["image_t1_path", "image_t2_path"])},
        {"name": "run_super_resolution", "description": "Submit a super-resolution job.", "inputSchema": _schema({"image_path": string, "model_id": string, "scale": integer, "tile_size": integer, "overlap": integer, "reference_path": string}, ["image_path"])},
        {"name": "run_spectral_indices", "description": "Submit a spectral index job.", "inputSchema": _schema({"image_path": string, "indices": array_string}, ["image_path"])},
        {"name": "calculate_statistics", "description": "Analyze an existing job.", "inputSchema": _schema({"job_id": string}, ["job_id"])},
        {"name": "quality_check_result", "description": "Quality-check an existing job.", "inputSchema": _schema({"job_id": string}, ["job_id"])},
        {"name": "generate_report", "description": "Generate a markdown report for an existing job.", "inputSchema": _schema({"job_id": string, "output_format": string}, ["job_id"])},
        {"name": "get_job_status", "description": "Get job status.", "inputSchema": _schema({"job_id": string}, ["job_id"])},
        {"name": "get_result_manifest", "description": "Get job manifest.", "inputSchema": _schema({"job_id": string}, ["job_id"])},
    ]
