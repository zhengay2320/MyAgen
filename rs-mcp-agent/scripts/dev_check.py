from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Run a local developer readiness check for the MCP project."""
    checks: list[dict[str, Any]] = []
    checks.append(_check_python_version())
    checks.extend(_check_imports())
    checks.append(_check_workspace())
    checks.append(_check_configs())
    checks.append(_check_import("FastAPI app import", "rs_service.api", required=True))
    checks.append(_check_import("MCP server import", "rs_mcp.server", required=True))
    checks.append(_check_synthetic_generation())
    checks.append(_check_smoke_test())

    ok = all(item["ok"] or not item.get("required", True) for item in checks)
    payload = {"ok": ok, "checks": checks}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not ok:
        raise SystemExit(1)


def _check_python_version() -> dict[str, Any]:
    version = sys.version_info
    ok = version >= (3, 10)
    return {
        "name": "Python version",
        "ok": ok,
        "required": True,
        "detail": f"{version.major}.{version.minor}.{version.micro}",
    }


def _check_imports() -> list[dict[str, Any]]:
    required = ["numpy", "PIL"]
    optional_runtime = ["fastapi", "uvicorn", "pydantic", "httpx", "rasterio", "geopandas", "shapely", "pandas", "typer", "rich"]
    optional_models = ["ultralytics", "sahi", "torch", "mmdet", "mmseg", "mmrotate", "opencd", "basicsr", "mmagic"]
    checks = [_check_import(f"required import: {name}", name, required=True) for name in required]
    checks.extend(_check_import(f"runtime import: {name}", name, required=False) for name in optional_runtime)
    checks.extend(_check_import(f"model import: {name}", name, required=False) for name in optional_models)
    return checks


def _check_import(name: str, module: str, required: bool) -> dict[str, Any]:
    try:
        importlib.import_module(module)
        return {"name": name, "ok": True, "required": required, "detail": "import ok"}
    except Exception as exc:
        return {"name": name, "ok": False, "required": required, "detail": str(exc)}


def _check_workspace() -> dict[str, Any]:
    try:
        from rs_service.settings import get_settings

        settings = get_settings()
        dirs = [
            settings.workspace,
            settings.inputs_dir,
            settings.outputs_dir,
            settings.previews_dir,
            settings.reports_dir,
            settings.jobs_dir,
            settings.cache_dir,
        ]
        missing = [str(path) for path in dirs if not Path(path).exists()]
        return {"name": "workspace directories", "ok": not missing, "required": True, "detail": {"missing": missing}}
    except Exception as exc:
        return {"name": "workspace directories", "ok": False, "required": True, "detail": str(exc)}


def _check_configs() -> dict[str, Any]:
    required = [
        "configs/models.yaml",
        "configs/tasks.yaml",
        "configs/analysis_rules.yaml",
        "configs/class_maps/landcover.yaml",
        "configs/class_maps/vehicle_ship_aircraft.yaml",
        "configs/class_maps/building_change.yaml",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    return {"name": "config files", "ok": not missing, "required": True, "detail": {"missing": missing}}


def _check_synthetic_generation() -> dict[str, Any]:
    path = ROOT / "workspace" / "dev_check_synthetic.tif"
    command = [sys.executable, "scripts/create_synthetic_geotiff.py", "--output", str(path)]
    return _run_command("synthetic data generation", command)


def _check_smoke_test() -> dict[str, Any]:
    return _run_command("smoke test", [sys.executable, "scripts/smoke_test.py"])


def _run_command(name: str, command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=180)
        return {
            "name": name,
            "ok": result.returncode == 0,
            "required": True,
            "detail": {
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-1000:],
                "stderr_tail": result.stderr[-1000:],
            },
        }
    except Exception as exc:
        return {"name": name, "ok": False, "required": True, "detail": str(exc)}


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
