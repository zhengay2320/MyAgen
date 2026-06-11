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
        return str(value)
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
    status: str = "success",
    metrics: dict[str, Any] | None = None,
    conclusion: str | None = None,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    model_payload = model or {"id": "fake", "backend": "fake"}
    statistics = stats or {}
    normalized_outputs = {str(key): str(value) for key, value in (outputs or {}).items()}
    manifest = {
        "schema_version": "0.1",
        "job_id": out_dir.name,
        "task": task,
        "status": status,
        "model_id": model_payload.get("id", "unknown"),
        "input_files": _collect_input_files(inputs),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parameters": parameters,
        "outputs": normalized_outputs,
        "statistics": statistics,
        "metrics": metrics or {},
        "quality_flags": quality_flags or [],
        "conclusion": conclusion or _default_conclusion(task, status, quality_flags or []),
        "errors": errors or [],
        "inputs": inputs,
        "model": model_payload,
        "stats": statistics,
    }
    manifest["manifest_path"] = str(out_dir / "manifest.json")
    return manifest


def write_manifest(**kwargs: Any) -> dict[str, Any]:
    manifest = build_manifest(**kwargs)
    write_json(manifest["manifest_path"], manifest)
    return manifest


def _collect_input_files(inputs: dict[str, Any]) -> list[str]:
    files: list[str] = []
    for key, value in inputs.items():
        if key.endswith("raster") and isinstance(value, dict):
            continue
        if value is None:
            continue
        if isinstance(value, str):
            files.append(value)
        elif isinstance(value, list):
            files.extend(str(item) for item in value)
    return files


def _default_conclusion(task: str, status: str, quality_flags: list[dict[str, Any]]) -> str:
    if status == "failed":
        return f"{task} failed."
    error_count = sum(1 for item in quality_flags if item.get("severity") == "error")
    warning_count = sum(1 for item in quality_flags if item.get("severity") == "warning")
    if error_count:
        return f"{task} completed with {error_count} error quality flag(s)."
    if warning_count:
        return f"{task} completed with {warning_count} warning quality flag(s)."
    return f"{task} completed successfully."
