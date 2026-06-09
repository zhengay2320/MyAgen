from __future__ import annotations

from pathlib import Path
from typing import Any

from rs_service.core.manifest import new_job_id


def prepare_output_dir(output_dir: str | Path | None, workspace: str | Path = "workspace", task: str = "job") -> Path:
    if output_dir is not None:
        out = Path(output_dir)
    else:
        out = Path(workspace) / new_job_id(task)
    out.mkdir(parents=True, exist_ok=True)
    return out


def flag(code: str, message: str, severity: str = "info", **extra: Any) -> dict[str, Any]:
    payload = {"code": code, "severity": severity, "message": message}
    payload.update(extra)
    return payload


def relative_or_str(path: str | Path) -> str:
    return str(Path(path))
