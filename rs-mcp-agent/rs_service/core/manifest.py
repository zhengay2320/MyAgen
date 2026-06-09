from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any


def new_job_id(prefix: str = "job") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def write_json(path: str | Path, payload: dict[str, Any]) -> str:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return str(out_path)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


def build_manifest(
    *,
    task: str,
    output_dir: str | Path,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    parameters: dict[str, Any],
    stats: dict[str, Any] | None = None,
    quality_flags: list[dict[str, Any]] | None = None,
    model: dict[str, Any] | None = None,
    status: str = "completed",
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    manifest = {
        "schema_version": "0.1",
        "job_id": out_dir.name,
        "task": task,
        "status": status,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": inputs,
        "outputs": outputs,
        "parameters": parameters,
        "model": model or {"id": "fake", "backend": "fake"},
        "stats": stats or {},
        "quality_flags": quality_flags or [],
    }
    manifest["manifest_path"] = str(out_dir / "manifest.json")
    return manifest


def write_manifest(**kwargs: Any) -> dict[str, Any]:
    manifest = build_manifest(**kwargs)
    write_json(manifest["manifest_path"], manifest)
    return manifest
