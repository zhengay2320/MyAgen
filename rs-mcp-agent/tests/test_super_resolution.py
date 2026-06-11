from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from rs_service.adapters.base import ModelBackendUnavailable
from rs_service.adapters.basicsr_adapter import BasicSRAdapter
from rs_service.adapters.mmagic_adapter import MMagicSuperResolutionAdapter
from rs_service.adapters.swinir_adapter import SwinIRAdapter
from rs_service.core.raster import DEFAULT_TRANSFORM, create_synthetic_array, profile_from_array, read_raster, write_raster
from rs_service.pipelines.super_resolution import run_super_resolution
from rs_service.registry import get_adapter, list_models


class SuperResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.lr = create_synthetic_array(width=32, height=24, bands=3, changed=False)
        self.lr_path = self.root / "lr.tif"
        write_raster(self.lr_path, self.lr, profile_from_array(self.lr, transform=DEFAULT_TRANSFORM))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_fake_sr_scale_2_output_size_and_transform(self) -> None:
        """Fake SR x2 should write the expected output shape and transform."""
        manifest = run_super_resolution(self.lr_path, self.root / "sr2", tile_size=16, overlap=4, scale=2)
        sr, _, info = read_raster(manifest["outputs"]["super_resolved_geotiff"])
        self.assertEqual(sr.shape, (3, 48, 64))
        self.assertEqual(info.transform[0], DEFAULT_TRANSFORM[0] / 2)
        self.assertEqual(info.transform[4], DEFAULT_TRANSFORM[4] / 2)
        self.assertTrue(Path(manifest["outputs"]["preview"]).exists())

    def test_fake_sr_scale_4_output_size_and_transform(self) -> None:
        """Fake SR x4 should write the expected output shape and transform."""
        manifest = run_super_resolution(self.lr_path, self.root / "sr4", tile_size=16, overlap=4, scale=4)
        sr, _, info = read_raster(manifest["outputs"]["super_resolved_geotiff"])
        self.assertEqual(sr.shape, (3, 96, 128))
        self.assertEqual(info.transform[0], DEFAULT_TRANSFORM[0] / 4)
        self.assertEqual(info.transform[4], DEFAULT_TRANSFORM[4] / 4)

    def test_no_reference_does_not_generate_reference_metrics(self) -> None:
        """Without reference image, PSNR/SSIM must not be invented."""
        manifest = run_super_resolution(self.lr_path, self.root / "sr_no_ref", tile_size=16, overlap=4, scale=2)
        self.assertNotIn("psnr", manifest["metrics"])
        self.assertNotIn("ssim", manifest["metrics"])
        self.assertIn("无参考图", manifest["conclusion"])
        self.assertFalse(manifest["statistics"]["reference_metrics_available"])

    def test_reference_metrics_are_present(self) -> None:
        """With a synthetic reference image, PSNR and SSIM should be calculated."""
        reference = np.repeat(np.repeat(self.lr, 2, axis=1), 2, axis=2)
        reference_path = self.root / "reference.tif"
        ref_profile = profile_from_array(reference, transform=(DEFAULT_TRANSFORM[0] / 2, 0, DEFAULT_TRANSFORM[2], 0, DEFAULT_TRANSFORM[4] / 2, DEFAULT_TRANSFORM[5]))
        write_raster(reference_path, reference, ref_profile)
        manifest = run_super_resolution(self.lr_path, self.root / "sr_ref", tile_size=16, overlap=4, scale=2, reference_path=reference_path)
        self.assertIn("psnr", manifest["metrics"])
        self.assertIn("ssim", manifest["metrics"])
        self.assertTrue(manifest["statistics"]["reference_metrics_available"])

    def test_real_sr_model_ids_are_lazy(self) -> None:
        """Registry should expose optional SR backends without importing heavy deps."""
        model_ids = {item["id"] for item in list_models()["models"]}
        self.assertIn("swinir_x2", model_ids)
        self.assertIn("swinir_x4", model_ids)
        self.assertIn("basicsr_x4", model_ids)
        self.assertIn("mmagic_sr_stub", model_ids)
        self.assertEqual(get_adapter("super_resolution", model_id="swinir_x2").metadata.backend, "swinir")
        self.assertEqual(get_adapter("super_resolution", model_id="basicsr_x4").metadata.backend, "basicsr")

    def test_missing_swinir_dependency_or_weight_is_readable(self) -> None:
        """SwinIR should fail with clear dependency or checkpoint errors."""
        adapter = SwinIRAdapter({"id": "swinir_x2", "checkpoint": str(self.root / "missing.pth"), "scale": 2})
        with self.assertRaises((ModelBackendUnavailable, FileNotFoundError)) as raised:
            adapter.load()
        self.assertTrue("SwinIR" in str(raised.exception) or "checkpoint" in str(raised.exception))

    def test_missing_basicsr_error_is_readable(self) -> None:
        """BasicSR should fail clearly when optional deps are absent."""
        if importlib.util.find_spec("basicsr") is not None:
            self.skipTest("basicsr is installed in this environment")
        adapter = BasicSRAdapter({"id": "basicsr_x4", "config": "missing.yml", "checkpoint": "missing.pth", "scale": 4})
        with self.assertRaises(ModelBackendUnavailable) as raised:
            adapter.load()
        self.assertIn("BasicSR backend is unavailable", str(raised.exception))

    def test_mmagic_stub_error_is_readable(self) -> None:
        """MMagic stub should reserve the interface with a clear error."""
        adapter = MMagicSuperResolutionAdapter({"id": "mmagic_sr_stub", "scale": 4})
        with self.assertRaises(ModelBackendUnavailable) as raised:
            adapter.load()
        self.assertIn("reserved", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
