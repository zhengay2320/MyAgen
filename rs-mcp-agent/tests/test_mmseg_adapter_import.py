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
from rs_service.adapters.mmseg_adapter import INSTALL_MMSEG_MESSAGE, MMSegmentationAdapter
from rs_service.registry import get_adapter, list_models


class MMSegAdapterImportTests(unittest.TestCase):
    def test_api_import_does_not_require_mmseg(self) -> None:
        """Importing service modules should not import OpenMMLab packages."""
        import rs_service.api  # noqa: F401

        model_ids = {item["id"] for item in list_models()["models"]}
        self.assertIn("mmseg_segformer_landcover", model_ids)
        self.assertIn("mmseg_deeplab_building", model_ids)

    def test_missing_mmseg_error_is_readable(self) -> None:
        """When mmseg is absent, load() should raise a clear install message."""
        if importlib.util.find_spec("mmseg") is not None:
            self.skipTest("mmseg is installed in this environment")
        adapter = MMSegmentationAdapter(
            {
                "id": "mmseg_segformer_landcover",
                "config": "external/missing.py",
                "checkpoint": "weights/missing.pth",
            }
        )
        with self.assertRaises(ModelBackendUnavailable) as raised:
            adapter.load()
        self.assertIn("MMSegmentation backend is unavailable", str(raised.exception))
        self.assertIn("openmim", str(raised.exception))

    def test_registry_mmseg_adapter_is_lazy(self) -> None:
        """Registry should build the adapter without importing mmseg APIs."""
        adapter = get_adapter("semantic_segmentation", model_id="mmseg_segformer_landcover")
        self.assertEqual(adapter.metadata.backend, "mmseg")

    def test_predict_tile_with_mocked_mmseg(self) -> None:
        """Mock MMSeg APIs to validate output mask/probability format."""
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "model.py"
            checkpoint = Path(tmp) / "model.pth"
            config.write_text("# mock config", encoding="utf-8")
            checkpoint.write_text("mock checkpoint", encoding="utf-8")

            mmseg_module = types.ModuleType("mmseg")
            mmseg_module.__path__ = []
            apis_module = types.ModuleType("mmseg.apis")
            apis_module.init_model = _fake_init_model
            apis_module.inference_model = _fake_inference_model

            with mock.patch.dict(sys.modules, {"mmseg": mmseg_module, "mmseg.apis": apis_module}):
                with mock.patch("importlib.util.find_spec", return_value=object()):
                    adapter = MMSegmentationAdapter(
                        {
                            "id": "mmseg_segformer_landcover",
                            "config": str(config),
                            "checkpoint": str(checkpoint),
                            "classes": ["background", "building"],
                            "device": "cpu",
                        }
                    )
                    prediction = adapter.predict_tile(np.zeros((3, 8, 10), dtype=np.uint8), {"tile_id": "tile_00001"})

        self.assertEqual(prediction.mask.shape, (8, 10))
        self.assertEqual(prediction.probabilities.shape, (2, 8, 10))
        self.assertEqual(prediction.class_names[1], "building")
        self.assertEqual(prediction.metadata["backend"], "mmseg")

    def test_install_message_constant(self) -> None:
        """The shared install hint should mention OpenMMLab tooling."""
        self.assertIn("MMSegmentation", INSTALL_MMSEG_MESSAGE)
        self.assertIn("openmim", INSTALL_MMSEG_MESSAGE)


def _fake_init_model(_config: str, _checkpoint: str, device: str = "cpu"):
    return types.SimpleNamespace(CLASSES=["background", "building"], device=device)


def _fake_inference_model(_model, image: np.ndarray):
    height, width = image.shape[:2]
    mask = np.zeros((1, height, width), dtype=np.uint8)
    mask[:, 2:6, 3:8] = 1
    logits = np.zeros((2, height, width), dtype=np.float32)
    logits[0] = 0.1
    logits[1, 2:6, 3:8] = 3.0
    return types.SimpleNamespace(
        pred_sem_seg=types.SimpleNamespace(data=mask),
        seg_logits=types.SimpleNamespace(data=logits),
    )


if __name__ == "__main__":
    unittest.main()
