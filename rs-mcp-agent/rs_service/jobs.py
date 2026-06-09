from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from rs_service.core.manifest import read_json


@dataclass
class JobRecord:
    job_id: str
    task: str
    status: str
    created_at: float
    updated_at: float
    manifest_path: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.created_at))
        payload["updated_at_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.updated_at))
        return payload


class JobStore:
    def __init__(self) -> None:
        self._records: dict[str, JobRecord] = {}

    def submit_sync(self, task: str, runner: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        created = time.time()
        placeholder_id = f"{task}_{int(created)}"
        record = JobRecord(
            job_id=placeholder_id,
            task=task,
            status="running",
            created_at=created,
            updated_at=created,
        )
        self._records[placeholder_id] = record
        try:
            manifest = runner()
            job_id = str(manifest.get("job_id") or Path(manifest["manifest_path"]).parent.name)
            record.job_id = job_id
            record.status = "completed"
            record.manifest_path = manifest.get("manifest_path")
            record.updated_at = time.time()
            record.metadata = {"outputs": manifest.get("outputs", {}), "stats": manifest.get("stats", {})}
            self._records.pop(placeholder_id, None)
            self._records[job_id] = record
            return manifest
        except Exception as exc:
            record.status = "failed"
            record.error = str(exc)
            record.updated_at = time.time()
            raise

    def get(self, job_id: str) -> dict[str, Any]:
        record = self._records.get(job_id)
        if record:
            return record.to_dict()
        manifest_path = Path("workspace") / job_id / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            return {
                "job_id": job_id,
                "task": manifest.get("task"),
                "status": manifest.get("status", "completed"),
                "manifest_path": str(manifest_path),
                "metadata": {"outputs": manifest.get("outputs", {}), "stats": manifest.get("stats", {})},
            }
        return {"job_id": job_id, "status": "not_found"}

    def list(self) -> list[dict[str, Any]]:
        records = [record.to_dict() for record in self._records.values()]
        workspace = Path("workspace")
        if workspace.exists():
            for manifest_path in workspace.glob("*/manifest.json"):
                job_id = manifest_path.parent.name
                if any(record["job_id"] == job_id for record in records):
                    continue
                try:
                    manifest = read_json(manifest_path)
                    records.append(
                        {
                            "job_id": job_id,
                            "task": manifest.get("task"),
                            "status": manifest.get("status", "completed"),
                            "manifest_path": str(manifest_path),
                            "metadata": {"outputs": manifest.get("outputs", {}), "stats": manifest.get("stats", {})},
                        }
                    )
                except Exception:
                    continue
        return records


job_store = JobStore()
