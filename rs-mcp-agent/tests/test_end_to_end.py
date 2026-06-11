from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pytest
except ImportError as exc:  # pragma: no cover - minimal local runtime
    raise unittest.SkipTest("pytest is not installed") from exc

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from rs_service.api import app
from rs_service.core.raster import DEFAULT_TRANSFORM, create_synthetic_array, profile_from_array, write_raster


def _write_pair(root: Path) -> tuple[str, str]:
    before = create_synthetic_array(width=80, height=64, bands=4, changed=False)
    after = create_synthetic_array(width=80, height=64, bands=4, changed=True)
    profile = profile_from_array(before, transform=DEFAULT_TRANSFORM)
    before_path = root / "before.tif"
    after_path = root / "after.tif"
    write_raster(before_path, before, profile)
    write_raster(after_path, after, profile)
    return str(before_path), str(after_path)


def _submit_and_report(client: TestClient, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.post(endpoint, json=payload)
    assert response.status_code == 200, response.text
    job = response.json()
    assert job["status"] == "success"
    manifest_response = client.get(f"/jobs/{job['job_id']}/manifest")
    assert manifest_response.status_code == 200
    manifest = manifest_response.json()
    assert manifest["status"] == "success"
    analyze = client.post(f"/jobs/{job['job_id']}/analyze", json={})
    assert analyze.status_code == 200, analyze.text
    report = client.post(f"/jobs/{job['job_id']}/report", json={})
    assert report.status_code == 200, report.text
    final_manifest = report.json()["result"]
    assert final_manifest["statistics"]
    assert final_manifest["conclusion"]
    assert "report" in final_manifest["outputs"] or "report_md" in final_manifest["outputs"]
    return final_manifest


def test_api_end_to_end_fake_tasks() -> None:
    """Run the fake MCP service workflow through the FastAPI app."""
    client = TestClient(app)
    with tempfile.TemporaryDirectory() as tmp:
        before, after = _write_pair(Path(tmp))

        inspect = client.post("/inspect", json={"path": before})
        assert inspect.status_code == 200
        assert inspect.json()["result"]["count"] == 4

        preflight = client.post("/preflight", json={"image_path": before, "task": "object_detection", "tile_size": 40, "overlap": 8})
        assert preflight.status_code == 200
        assert preflight.json()["result"]["tile_count"] > 0

        manifests = [
            _submit_and_report(client, "/jobs/detection", {"image_path": before, "tile_size": 40, "overlap": 8}),
            _submit_and_report(client, "/jobs/oriented-detection", {"image_path": before, "tile_size": 40, "overlap": 8}),
            _submit_and_report(client, "/jobs/semantic-segmentation", {"image_path": before, "tile_size": 40, "overlap": 8}),
            _submit_and_report(client, "/jobs/instance-segmentation", {"image_path": before, "tile_size": 40, "overlap": 8}),
            _submit_and_report(client, "/jobs/change-detection", {"before_path": before, "after_path": after, "tile_size": 40, "overlap": 8, "threshold": 0.2}),
            _submit_and_report(client, "/jobs/super-resolution", {"image_path": before, "tile_size": 40, "overlap": 8, "scale": 2}),
            _submit_and_report(client, "/jobs/spectral-indices", {"image_path": before, "indices": ["ndvi"], "tile_size": 40, "overlap": 8}),
        ]

    assert {manifest["task"] for manifest in manifests} == {
        "object_detection",
        "oriented_detection",
        "semantic_segmentation",
        "instance_segmentation",
        "change_detection",
        "super_resolution",
        "spectral_indices",
    }
