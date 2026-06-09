from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rs_service.core.manifest import build_manifest
from rs_service.job_store import LocalJobStore
from rs_service.settings import Settings


class FoundationTests(unittest.TestCase):
    def test_settings_create_workspace_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                workspace=Path(tmp) / "workspace",
                models_config=Path(tmp) / "configs" / "models.yaml",
                analysis_rules=Path(tmp) / "configs" / "analysis_rules.yaml",
                service_host="127.0.0.1",
                service_port=8765,
            )
            settings.ensure_workspace()
            for name in ["inputs", "outputs", "previews", "reports", "jobs", "cache"]:
                self.assertTrue((settings.workspace / name).is_dir())

    def test_local_job_store_persists_status_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalJobStore(jobs_dir=root / "jobs", outputs_dir=root / "outputs")
            record = store.create_job("object_detection", model_id="fake-yolo-sahi", input_files=["input.tif"])
            self.assertEqual(record.status, "queued")
            updated = store.update_job(record.job_id, status="running")
            self.assertEqual(updated.status, "running")
            loaded = store.get_job(record.job_id)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.status, "running")
            self.assertEqual(len(store.list_jobs()), 1)

    def test_manifest_contains_canonical_fields_and_legacy_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = build_manifest(
                task="object_detection",
                output_dir=Path(tmp) / "outputs" / "job_1",
                inputs={"image": "input.tif"},
                outputs={"geojson": "detections.geojson"},
                parameters={"tile_size": 512, "overlap": 64},
                stats={"count": 3},
                quality_flags=[],
                model={"id": "fake-yolo-sahi", "backend": "fake"},
            )
            for field in [
                "job_id",
                "task",
                "status",
                "model_id",
                "input_files",
                "parameters",
                "outputs",
                "statistics",
                "metrics",
                "quality_flags",
                "conclusion",
                "errors",
            ]:
                self.assertIn(field, manifest)
            self.assertEqual(manifest["model_id"], "fake-yolo-sahi")
            self.assertEqual(manifest["statistics"], {"count": 3})
            self.assertEqual(manifest["stats"], {"count": 3})


if __name__ == "__main__":
    unittest.main()
