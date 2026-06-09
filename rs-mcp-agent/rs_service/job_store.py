from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from rs_service.core.manifest import new_job_id, read_json, write_json
from rs_service.settings import get_settings

JobState = Literal["queued", "running", "success", "failed"]


class JobRecord(BaseModel):
    job_id: str
    task: str
    status: JobState
    created_at: str
    updated_at: str
    model_id: str | None = None
    input_files: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    output_dir: str | None = None
    manifest_path: str | None = None
    outputs: dict[str, Any] = Field(default_factory=dict)
    statistics: dict[str, Any] = Field(default_factory=dict)
    quality_flags: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class LocalJobStore:
    def __init__(self, jobs_dir: str | Path | None = None, outputs_dir: str | Path | None = None) -> None:
        settings = get_settings()
        self.jobs_dir = Path(jobs_dir) if jobs_dir else settings.jobs_dir
        self.outputs_dir = Path(outputs_dir) if outputs_dir else settings.outputs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def _job_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def create_job(
        self,
        task: str,
        *,
        model_id: str | None = None,
        input_files: list[str] | None = None,
        parameters: dict[str, Any] | None = None,
        job_id: str | None = None,
    ) -> JobRecord:
        resolved_job_id = job_id or new_job_id(task)
        now = utc_now()
        output_dir = self.outputs_dir / resolved_job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        record = JobRecord(
            job_id=resolved_job_id,
            task=task,
            status="queued",
            created_at=now,
            updated_at=now,
            model_id=model_id,
            input_files=input_files or [],
            parameters=parameters or {},
            output_dir=str(output_dir),
        )
        self._write(record)
        return record

    def update_job(self, job_id: str, **updates: Any) -> JobRecord:
        record = self.get_job(job_id)
        if record is None:
            raise FileNotFoundError(f"Job not found: {job_id}")
        data = record.to_dict()
        data.update(updates)
        data["updated_at"] = utc_now()
        updated = JobRecord(**data)
        self._write(updated)
        return updated

    def get_job(self, job_id: str) -> JobRecord | None:
        path = self._job_path(job_id)
        if not path.exists():
            return None
        return JobRecord(**read_json(path))

    def list_jobs(self) -> list[JobRecord]:
        records: list[JobRecord] = []
        for path in sorted(self.jobs_dir.glob("*.json")):
            try:
                records.append(JobRecord(**read_json(path)))
            except Exception:
                continue
        return records

    def submit_sync(
        self,
        task: str,
        runner: Callable[[Path], dict[str, Any]],
        *,
        model_id: str | None = None,
        input_files: list[str] | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = self.create_job(task, model_id=model_id, input_files=input_files, parameters=parameters)
        output_dir = Path(record.output_dir or self.outputs_dir / record.job_id)
        self.update_job(record.job_id, status="running")
        try:
            manifest = runner(output_dir)
            self.update_job(
                record.job_id,
                status="success",
                manifest_path=manifest.get("manifest_path"),
                outputs=manifest.get("outputs", {}),
                statistics=manifest.get("statistics", manifest.get("stats", {})),
                quality_flags=manifest.get("quality_flags", []),
            )
            return manifest
        except Exception as exc:
            self.update_job(record.job_id, status="failed", errors=[str(exc)])
            raise

    def _write(self, record: JobRecord) -> None:
        write_json(self._job_path(record.job_id), record.to_dict())


job_store = LocalJobStore()
