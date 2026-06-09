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
    DetectionRequest,
    InspectRequest,
    InstanceSegmentationRequest,
    JobAnalyzeRequest,
    JobReportRequest,
    ManifestRequest,
    OrientedDetectionRequest,
    PreflightRequest,
    QualityRequest,
    ReportRequest,
    SemanticSegmentationRequest,
    SpectralIndicesRequest,
    StatisticsRequest,
    SuperResolutionRequest,
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


def _job_response(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "job_id": manifest.get("job_id"),
        "task": manifest.get("task"),
        "status": manifest.get("status", "success"),
        "manifest_path": manifest.get("manifest_path"),
        "outputs": manifest.get("outputs", {}),
        "result": manifest,
    }


def _submit_job(func, *args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        return _job_response(func(*args, **kwargs))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


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

    @app.get("/jobs/{job_id}/manifest")
    def job_manifest(job_id: str) -> dict[str, Any]:
        try:
            return services.get_result_manifest(job_id=job_id)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.post("/inspect")
    def inspect(payload: InspectRequest) -> dict[str, Any]:
        return _handle(services.inspect_raster, payload.path)

    @app.post("/preflight")
    def preflight(payload: PreflightRequest) -> dict[str, Any]:
        return _handle(services.preflight_plan, payload.image_path, payload.tile_size, payload.overlap, payload.task)

    @app.post("/run/object-detection")
    def object_detection(payload: DetectionRequest) -> dict[str, Any]:
        return _handle(services.run_object_detection, **_payload_dict(payload))

    @app.post("/jobs/detection")
    def job_detection(payload: DetectionRequest) -> dict[str, Any]:
        return _submit_job(services.run_object_detection, **_payload_dict(payload))

    @app.post("/run/oriented-detection")
    def oriented_detection(payload: OrientedDetectionRequest) -> dict[str, Any]:
        return _handle(services.run_oriented_detection, **_payload_dict(payload))

    @app.post("/jobs/oriented-detection")
    def job_oriented_detection(payload: OrientedDetectionRequest) -> dict[str, Any]:
        return _submit_job(services.run_oriented_detection, **_payload_dict(payload))

    @app.post("/run/semantic-segmentation")
    def semantic_segmentation(payload: SemanticSegmentationRequest) -> dict[str, Any]:
        data = _payload_dict(payload)
        data.pop("class_map", None)
        return _handle(services.run_semantic_segmentation, **data)

    @app.post("/jobs/semantic-segmentation")
    def job_semantic_segmentation(payload: SemanticSegmentationRequest) -> dict[str, Any]:
        data = _payload_dict(payload)
        data.pop("class_map", None)
        return _submit_job(services.run_semantic_segmentation, **data)

    @app.post("/run/instance-segmentation")
    def instance_segmentation(payload: InstanceSegmentationRequest) -> dict[str, Any]:
        return _handle(services.run_instance_segmentation, **_payload_dict(payload))

    @app.post("/jobs/instance-segmentation")
    def job_instance_segmentation(payload: InstanceSegmentationRequest) -> dict[str, Any]:
        return _submit_job(services.run_instance_segmentation, **_payload_dict(payload))

    @app.post("/run/change-detection")
    def change_detection(payload: ChangeDetectionRequest) -> dict[str, Any]:
        return _handle(services.run_change_detection, **_payload_dict(payload))

    @app.post("/jobs/change-detection")
    def job_change_detection(payload: ChangeDetectionRequest) -> dict[str, Any]:
        return _submit_job(services.run_change_detection, **_payload_dict(payload))

    @app.post("/run/super-resolution")
    def super_resolution(payload: SuperResolutionRequest) -> dict[str, Any]:
        return _handle(services.run_super_resolution, **_payload_dict(payload))

    @app.post("/jobs/super-resolution")
    def job_super_resolution(payload: SuperResolutionRequest) -> dict[str, Any]:
        return _submit_job(services.run_super_resolution, **_payload_dict(payload))

    @app.post("/run/spectral-indices")
    def spectral_indices(payload: SpectralIndicesRequest) -> dict[str, Any]:
        return _handle(services.run_spectral_indices, **_payload_dict(payload))

    @app.post("/jobs/spectral-indices")
    def job_spectral_indices(payload: SpectralIndicesRequest) -> dict[str, Any]:
        return _submit_job(services.run_spectral_indices, **_payload_dict(payload))

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

    @app.post("/jobs/{job_id}/analyze")
    def analyze_job(job_id: str, payload: JobAnalyzeRequest | None = None) -> dict[str, Any]:
        data = _payload_dict(payload) if payload is not None else {}
        source_manifest = services.get_result_manifest(job_id=job_id)
        return _submit_job(
            services.calculate_statistics,
            manifest_path=source_manifest["manifest_path"],
            output_dir=data.get("output_dir"),
            zones_path=data.get("zones_path"),
        )

    @app.post("/jobs/{job_id}/report")
    def report_job(job_id: str, payload: JobReportRequest | None = None) -> dict[str, Any]:
        data = _payload_dict(payload) if payload is not None else {}
        source_manifest = services.get_result_manifest(job_id=job_id)
        return _submit_job(
            services.generate_report,
            manifest_path=source_manifest["manifest_path"],
            output_dir=data.get("output_dir"),
            title=data.get("title", "Remote Sensing Processing Report"),
        )

    return app


app = create_app() if FastAPI is not None else None


def main() -> None:
    if app is None:
        raise SystemExit("FastAPI/uvicorn are not installed. Run: pip install -e .")
    import uvicorn
    from rs_service.settings import get_settings

    settings = get_settings()
    uvicorn.run("rs_service.api:app", host=settings.service_host, port=settings.service_port, reload=False)


if __name__ == "__main__":
    main()
