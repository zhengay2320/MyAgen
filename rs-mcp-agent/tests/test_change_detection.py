from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

from rs_service.adapters.base import ModelBackendUnavailable
from rs_service.adapters.opencd_adapter import INSTALL_OPENCD_MESSAGE, OpenCDAdapter
from rs_service.core.alignment import check_pair_alignment
from rs_service.core.raster import DEFAULT_CRS, DEFAULT_TRANSFORM, create_synthetic_array, profile_from_array, read_raster, write_raster
from rs_service.pipelines.change_detection import run_change_detection
from rs_service.registry import get_adapter, list_models


class ChangeDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        before = create_synthetic_array(width=64, height=48, bands=3, changed=False)
        after = create_synthetic_array(width=64, height=48, bands=3, changed=True)
        self.before_path = self.root / "before.tif"
        self.after_path = self.root / "after.tif"
        profile = profile_from_array(before, crs=DEFAULT_CRS, transform=DEFAULT_TRANSFORM)
        write_raster(self.before_path, before, profile)
        write_raster(self.after_path, after, profile)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_fake_change_detection_outputs_full_manifest(self) -> None:
        """Fake change detection should produce mask, vectors, preview, stats, and manifest."""
        manifest = run_change_detection(self.before_path, self.after_path, self.root / "change", tile_size=32, overlap=8, threshold=0.2)
        self.assertEqual(manifest["status"], "success")
        self.assertGreater(manifest["stats"]["changed_pixels"], 0)
        for key in ["mask_geotiff", "probability_npy", "geojson", "gpkg", "preview", "stats_json"]:
            self.assertIn(key, manifest["outputs"])
            self.assertTrue(Path(manifest["outputs"][key]).exists())

    def test_misaligned_pair_requires_auto_align(self) -> None:
        """CRS/resolution mismatch should fail clearly unless auto_align=True."""
        after = create_synthetic_array(width=32, height=24, bands=3, changed=True)
        mismatch_path = self.root / "after_mismatch.tif"
        mismatch_transform = (2.0, 0.0, DEFAULT_TRANSFORM[2], 0.0, -2.0, DEFAULT_TRANSFORM[5])
        write_raster(mismatch_path, after, profile_from_array(after, crs="EPSG:4326", transform=mismatch_transform))
        with self.assertRaises(ValueError) as raised:
            run_change_detection(self.before_path, mismatch_path, self.root / "bad", tile_size=32, overlap=8)
        self.assertIn("not aligned", str(raised.exception))

        manifest = run_change_detection(self.before_path, mismatch_path, self.root / "aligned", tile_size=32, overlap=8, auto_align=True, threshold=0.2)
        self.assertIn("aligned_after", manifest["outputs"])
        self.assertTrue(Path(manifest["outputs"]["aligned_after"]).exists())
        mask, _, info = read_raster(manifest["outputs"]["mask_geotiff"])
        self.assertEqual(mask.shape, (1, 48, 64))
        self.assertEqual(info.crs, DEFAULT_CRS)
        self.assertTrue(manifest["stats"]["alignment_warnings"])

    def test_alignment_checker_reports_mismatch(self) -> None:
        """Alignment checker should expose concrete mismatch reasons."""
        _, _, before_info = read_raster(self.before_path)
        _, _, after_info = read_raster(self.after_path)
        aligned = check_pair_alignment(before_info, after_info)
        self.assertTrue(aligned["aligned"])
        changed_info = type(after_info)(**{**after_info.to_dict(), "crs": "EPSG:4326"})
        mismatch = check_pair_alignment(before_info, changed_info)
        self.assertFalse(mismatch["aligned"])
        self.assertIn("crs_mismatch", mismatch["issues"])

    def test_missing_opencd_error_is_readable(self) -> None:
        """Missing Open-CD should raise a clear install message."""
        if importlib.util.find_spec("opencd") is not None:
            self.skipTest("opencd is installed in this environment")
        adapter = OpenCDAdapter({"id": "opencd_changer_building", "config": "missing.py", "checkpoint": "missing.pth"})
        with self.assertRaises(ModelBackendUnavailable) as raised:
            adapter.load()
        self.assertIn("Open-CD backend is unavailable", str(raised.exception))
        self.assertIn("openmim", str(raised.exception))

    def test_registry_opencd_adapter_is_lazy(self) -> None:
        """Registry should construct Open-CD adapter without importing opencd APIs."""
        model_ids = {item["id"] for item in list_models()["models"]}
        self.assertIn("opencd_changer_building", model_ids)
        self.assertEqual(get_adapter("change_detection", model_id="opencd_changer_building").metadata.backend, "opencd")

    def test_opencd_adapter_output_with_mocked_api(self) -> None:
        """Mock Open-CD APIs to validate tile output format."""
        config = self.root / "opencd.py"
        checkpoint = self.root / "opencd.pth"
        config.write_text("# config", encoding="utf-8")
        checkpoint.write_text("checkpoint", encoding="utf-8")
        opencd_module = types.ModuleType("opencd")
        opencd_module.__path__ = []
        apis_module = types.ModuleType("opencd.apis")
        apis_module.init_model = _fake_init_model
        apis_module.inference_model = _fake_inference_model
        with mock.patch.dict(sys.modules, {"opencd": opencd_module, "opencd.apis": apis_module}):
            with mock.patch("importlib.util.find_spec", return_value=object()):
                adapter = OpenCDAdapter({"id": "opencd_changer_building", "config": str(config), "checkpoint": str(checkpoint)})
                prediction = adapter.predict_tile(
                    np.zeros((3, 16, 20), dtype=np.uint8),
                    {"tile_id": "tile_1"},
                    tile_t2=np.ones((3, 16, 20), dtype=np.uint8) * 255,
                    threshold=0.5,
                )
        self.assertEqual(prediction.mask.shape, (16, 20))
        self.assertEqual(prediction.probability.shape, (16, 20))
        self.assertGreater(float(prediction.probability.max()), 0.5)

    def test_install_message_constant(self) -> None:
        """Install hint should mention Open-CD and OpenMMLab tooling."""
        self.assertIn("Open-CD", INSTALL_OPENCD_MESSAGE)
        self.assertIn("openmim", INSTALL_OPENCD_MESSAGE)


def _fake_init_model(_config: str, _checkpoint: str, device: str = "cpu"):
    return types.SimpleNamespace(device=device)


def _fake_inference_model(_model, image_t1: np.ndarray, image_t2: np.ndarray):
    height, width = image_t1.shape[:2]
    logits = np.zeros((2, height, width), dtype=np.float32)
    logits[0] = 0.1
    logits[1, 4:12, 5:14] = 3.0
    mask = np.zeros((1, height, width), dtype=np.uint8)
    mask[0, 4:12, 5:14] = 1
    return types.SimpleNamespace(seg_logits=types.SimpleNamespace(data=logits), pred_sem_seg=types.SimpleNamespace(data=mask))


if __name__ == "__main__":
    unittest.main()
