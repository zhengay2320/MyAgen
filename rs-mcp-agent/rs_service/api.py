from __future__ import annotations

from typing import Any

try:  # pragma: no cover - local environment may not have fastapi
    from fastapi import FastAPI, HTTPException
except Exception:  # pragma: no cover
    FastAPI = None
    HTTPException = Exception

from rs_service import __version__
from rs_service import services
from rs_service.schemas import (
    ChangeDetectionRequest,
    InspectRequest,
    ManifestRequest,
    PreflightRequest,
    QualityRequest,
    ReportRequest,
    SpectralIndicesRequest,
    StatisticsRequest,
    SuperResolutionRequest,
    TiledRunRequest,
)


def _handle(func, *args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        return {"ok": True, "result": func(*args, **kwargs)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _payload_dict(payload: Any) -> dict[str, Any]:
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    return payload.dict()


def create_app() -> Any:
    if FastAPI is None:  # pragma: no cover
        raise RuntimeError("FastAPI is not installed. Run: pip install -e .")
    app = FastAPI(title="rs-mcp-agent service", version=__version__)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "service": "rs_service", "version": __version__}

    @app.get("/models")
    def models() -> dict[str, Any]:
        return services.list_models()

    @app.get("/jobs")
    def jobs() -> list[dict[str, Any]]:
        return services.list_jobs()

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str) -> dict[str, Any]:
        return services.get_job_status(job_id)

    @app.post("/inspect")
    def inspect(payload: InspectRequest) -> dict[str, Any]:
        return _handle(services.inspect_raster, payload.path)

    @app.post("/preflight")
    def preflight(payload: PreflightRequest) -> dict[str, Any]:
        return _handle(services.preflight_plan, payload.image_path, payload.tile_size, payload.overlap)

    @app.post("/run/object-detection")
    def object_detection(payload: TiledRunRequest) -> dict[str, Any]:
        return _handle(services.run_object_detection, **_payload_dict(payload))

    @app.post("/run/oriented-detection")
    def oriented_detection(payload: TiledRunRequest) -> dict[str, Any]:
        return _handle(services.run_oriented_detection, **_payload_dict(payload))

    @app.post("/run/semantic-segmentation")
    def semantic_segmentation(payload: TiledRunRequest) -> dict[str, Any]:
        data = _payload_dict(payload)
        data.pop("score_threshold", None)
        return _handle(services.run_semantic_segmentation, **data)

    @app.post("/run/instance-segmentation")
    def instance_segmentation(payload: TiledRunRequest) -> dict[str, Any]:
        return _handle(services.run_instance_segmentation, **_payload_dict(payload))

    @app.post("/run/change-detection")
    def change_detection(payload: ChangeDetectionRequest) -> dict[str, Any]:
        return _handle(services.run_change_detection, **_payload_dict(payload))

    @app.post("/run/super-resolution")
    def super_resolution(payload: SuperResolutionRequest) -> dict[str, Any]:
        return _handle(services.run_super_resolution, **_payload_dict(payload))

    @app.post("/run/spectral-indices")
    def spectral_indices(payload: SpectralIndicesRequest) -> dict[str, Any]:
        return _handle(services.run_spectral_indices, **_payload_dict(payload))

    @app.post("/statistics")
    def statistics(payload: StatisticsRequest) -> dict[str, Any]:
        return _handle(services.calculate_statistics, **_payload_dict(payload))

    @app.post("/quality")
    def quality(payload: QualityRequest) -> dict[str, Any]:
        return _handle(services.quality_check_result, **_payload_dict(payload))

    @app.post("/report")
    def report(payload: ReportRequest) -> dict[str, Any]:
        return _handle(services.generate_report, **_payload_dict(payload))

    @app.post("/manifest")
    def manifest(payload: ManifestRequest) -> dict[str, Any]:
        return _handle(services.get_result_manifest, **_payload_dict(payload))

    return app


app = create_app() if FastAPI is not None else None


def main() -> None:
    if app is None:
        raise SystemExit("FastAPI/uvicorn are not installed. Run: pip install -e .")
    import uvicorn

    uvicorn.run("rs_service.api:app", host="127.0.0.1", port=8765, reload=False)


if __name__ == "__main__":
    main()
