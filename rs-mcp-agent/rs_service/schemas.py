from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class InspectRequest(BaseModel):
    path: str


class TiledRunRequest(BaseModel):
    image_path: str
    output_dir: str | None = None
    tile_size: int = Field(default=512, gt=0)
    overlap: int = Field(default=64, ge=0)
    model_id: str | None = None
    score_threshold: float = 0.0


class PreflightRequest(BaseModel):
    image_path: str
    tile_size: int = Field(default=512, gt=0)
    overlap: int = Field(default=64, ge=0)


class ChangeDetectionRequest(BaseModel):
    before_path: str
    after_path: str
    output_dir: str | None = None
    tile_size: int = Field(default=512, gt=0)
    overlap: int = Field(default=64, ge=0)
    model_id: str | None = None
    threshold: float = 0.5


class SuperResolutionRequest(BaseModel):
    image_path: str
    output_dir: str | None = None
    tile_size: int = Field(default=512, gt=0)
    overlap: int = Field(default=64, ge=0)
    scale: int = Field(default=2, gt=0)
    model_id: str | None = None


class SpectralIndicesRequest(BaseModel):
    image_path: str
    output_dir: str | None = None
    indices: list[str] | None = None
    band_mapping: dict[str, int] | None = None
    tile_size: int = Field(default=512, gt=0)
    overlap: int = Field(default=64, ge=0)


class StatisticsRequest(BaseModel):
    input_path: str | None = None
    output_dir: str | None = None
    manifest_path: str | None = None
    zones_path: str | None = None


class QualityRequest(BaseModel):
    input_path: str | None = None
    output_dir: str | None = None
    manifest_path: str | None = None


class ReportRequest(BaseModel):
    manifest_path: str
    output_dir: str | None = None
    title: str = "Remote Sensing Processing Report"


class ManifestRequest(BaseModel):
    job_id: str | None = None
    manifest_path: str | None = None


class ApiResponse(BaseModel):
    ok: bool = True
    result: dict[str, Any] | list[dict[str, Any]]
