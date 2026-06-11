from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

JobStatusValue = Literal["queued", "running", "success", "failed", "not_found"]


class RasterInspectRequest(BaseModel):
    path: str


class RasterInspectResult(BaseModel):
    path: str
    width: int
    height: int
    count: int
    dtype: str
    crs: str | None = None
    transform: tuple[float, float, float, float, float, float]
    bounds: tuple[float, float, float, float]
    nodata: float | int | None = None
    driver: str = "GTiff"
    fallback_container: bool = False
    pixel_size: list[float] = Field(default_factory=list)
    area_per_pixel: float | None = None
    is_georeferenced: bool = False


class PreflightRequest(BaseModel):
    image_path: str
    task: str = "detection"
    tile_size: int | None = Field(default=None, gt=0)
    overlap: int | None = Field(default=None, ge=0)


class PreflightResult(BaseModel):
    input: str
    raster: dict[str, Any]
    task: str = "detection"
    tile_size: int
    overlap: int
    tile_count: int
    tiles: list[dict[str, Any]] = Field(default_factory=list)
    windows: list[dict[str, int]]
    windows_truncated: bool = False
    outputs: dict[str, list[str]] = Field(default_factory=dict)


class JobSubmitResponse(BaseModel):
    job_id: str
    task: str
    status: JobStatusValue
    manifest_path: str | None = None
    output_dir: str | None = None


class JobStatus(BaseModel):
    job_id: str
    task: str | None = None
    status: JobStatusValue
    created_at: str | None = None
    updated_at: str | None = None
    model_id: str | None = None
    input_files: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    output_dir: str | None = None
    manifest_path: str | None = None
    outputs: dict[str, Any] = Field(default_factory=dict)
    statistics: dict[str, Any] = Field(default_factory=dict)
    quality_flags: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class TiledTaskRequest(BaseModel):
    image_path: str
    output_dir: str | None = None
    tile_size: int = Field(default=512, gt=0)
    overlap: int = Field(default=64, ge=0)
    model_id: str | None = None


class DetectionRequest(TiledTaskRequest):
    score_threshold: float = Field(default=0.0, ge=0.0)
    nms_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class OrientedDetectionRequest(TiledTaskRequest):
    score_threshold: float = Field(default=0.0, ge=0.0)


class SemanticSegmentationRequest(TiledTaskRequest):
    class_map: str | None = None


class InstanceSegmentationRequest(TiledTaskRequest):
    score_threshold: float = Field(default=0.0, ge=0.0)
    nms_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class ChangeDetectionRequest(BaseModel):
    before_path: str
    after_path: str
    output_dir: str | None = None
    tile_size: int = Field(default=512, gt=0)
    overlap: int = Field(default=64, ge=0)
    model_id: str | None = None
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    auto_align: bool = False


class SuperResolutionRequest(TiledTaskRequest):
    scale: int = Field(default=2, gt=0)
    reference_path: str | None = None


class SpectralIndexRequest(BaseModel):
    image_path: str
    output_dir: str | None = None
    indices: list[str] | None = None
    band_mapping: dict[str, int] | None = None
    band_map: dict[str, int] | None = None
    thresholds: dict[str, Any] | None = None
    tile_size: int = Field(default=512, gt=0)
    overlap: int = Field(default=64, ge=0)


class StatisticsRequest(BaseModel):
    input_path: str | None = None
    output_dir: str | None = None
    manifest_path: str | None = None
    zones_path: str | None = None


class QualityCheckRequest(BaseModel):
    input_path: str | None = None
    output_dir: str | None = None
    manifest_path: str | None = None


class ReportRequest(BaseModel):
    manifest_path: str
    output_dir: str | None = None
    title: str = "Remote Sensing Processing Report"


class JobAnalyzeRequest(BaseModel):
    output_dir: str | None = None
    zones_path: str | None = None


class JobReportRequest(BaseModel):
    output_dir: str | None = None
    title: str = "Remote Sensing Processing Report"


class Manifest(BaseModel):
    job_id: str
    task: str
    status: str
    model_id: str
    input_files: list[str]
    parameters: dict[str, Any]
    outputs: dict[str, Any]
    statistics: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    quality_flags: list[dict[str, Any]] = Field(default_factory=list)
    conclusion: str = ""
    errors: list[str] = Field(default_factory=list)
    schema_version: str = "0.1"
    manifest_path: str | None = None
    created_at: str | None = None


class ManifestRequest(BaseModel):
    job_id: str | None = None
    manifest_path: str | None = None


class ApiResponse(BaseModel):
    ok: bool = True
    result: dict[str, Any] | list[dict[str, Any]]


# Backward-compatible names used by the existing API/tests.
InspectRequest = RasterInspectRequest
TiledRunRequest = DetectionRequest
SpectralIndicesRequest = SpectralIndexRequest
QualityRequest = QualityCheckRequest
