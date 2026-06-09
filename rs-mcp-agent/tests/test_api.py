from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import pytest
except ImportError as exc:  # pragma: no cover - minimal local runtime
    raise unittest.SkipTest("pytest is not installed") from exc

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from rs_service.api import app
from rs_service.core.raster_io import profile_for_array, write_geotiff


def _create_test_raster(path: Path) -> str:
    """Create a small raster for API tests."""
    array = np.zeros((3, 48, 64), dtype=np.uint8)
    array[0, 10:25, 12:30] = 230
    array[1, 28:40, 36:50] = 210
    array[2, 28:40, 36:50] = 240
    profile = profile_for_array(array, crs="EPSG:3857", transform=(1, 0, 1000, 0, -1, 2000))
    write_geotiff(path, array, profile)
    return str(path)


def test_health() -> None:
    """Health endpoint should return service status."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["service"] == "rs_service"


def test_models() -> None:
    """Models endpoint should list fake adapters without importing real model frameworks."""
    client = TestClient(app)
    response = client.get("/models")
    assert response.status_code == 200
    payload = response.json()
    model_ids = {item["id"] for item in payload["models"]}
    assert "fake_segmentation" in model_ids
    assert "fake_detection" in model_ids


def test_inspect() -> None:
    """Inspect endpoint should return raster metadata."""
    client = TestClient(app)
    with tempfile.TemporaryDirectory() as tmp:
        image_path = _create_test_raster(Path(tmp) / "api_synthetic.tif")
        response = client.post("/inspect", json={"path": image_path})
        assert response.status_code == 200
        result = response.json()["result"]
        assert result["width"] == 64
        assert result["height"] == 48
        assert result["count"] == 3


def test_fake_semantic_segmentation_job_and_manifest() -> None:
    """Semantic segmentation job should synchronously return a job_id and queryable manifest."""
    client = TestClient(app)
    with tempfile.TemporaryDirectory() as tmp:
        image_path = _create_test_raster(Path(tmp) / "api_synthetic.tif")
        response = client.post(
            "/jobs/semantic-segmentation",
            json={"image_path": image_path, "tile_size": 32, "overlap": 8},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["job_id"]
        assert payload["status"] == "success"
        assert Path(payload["manifest_path"]).exists()

        job_response = client.get(f"/jobs/{payload['job_id']}")
        assert job_response.status_code == 200
        assert job_response.json()["status"] == "success"

        manifest_response = client.get(f"/jobs/{payload['job_id']}/manifest")
        assert manifest_response.status_code == 200
        manifest = manifest_response.json()
        assert manifest["job_id"] == payload["job_id"]
        assert manifest["task"] == "semantic_segmentation"
        assert manifest["status"] == "success"
