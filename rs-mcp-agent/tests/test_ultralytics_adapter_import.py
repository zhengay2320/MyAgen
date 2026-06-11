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
from rs_service.adapters.ultralytics_adapter import (
    INSTALL_DETECTION_MESSAGE,
    UltralyticsDetectionAdapter,
    UltralyticsInstanceSegmentationAdapter,
)
from rs_service.registry import get_adapter, list_models


class UltralyticsAdapterImportTests(unittest.TestCase):
    def test_api_import_does_not_require_ultralytics(self) -> None:
        """Importing service modules should not import optional model frameworks."""
        import rs_service.api  # noqa: F401

        model_ids = {item["id"] for item in list_models()["models"]}
        self.assertIn("yolo_detection", model_ids)
        self.assertIn("sahi_yolo_detection", model_ids)
        self.assertIn("yolo_instance_segmentation", model_ids)

    def test_missing_ultralytics_error_is_readable(self) -> None:
        """When ultralytics is absent, load() should raise a clear install message."""
        if importlib.util.find_spec("ultralytics") is not None:
            self.skipTest("ultralytics is installed in this environment")
        adapter = UltralyticsDetectionAdapter({"id": "yolo_detection", "weights": "weights/missing.pt"})
        with self.assertRaises(ModelBackendUnavailable) as raised:
            adapter.load()
        self.assertIn('pip install ".[detection]"', str(raised.exception))
        self.assertIn("pip install ultralytics sahi", str(raised.exception))

    def test_registry_real_adapters_are_lazy(self) -> None:
        """Registry should construct real adapters only when their model_id is selected."""
        adapter = get_adapter("object_detection", model_id="yolo_detection")
        self.assertEqual(adapter.metadata.backend, "ultralytics")
        instance_adapter = get_adapter("instance_segmentation", model_id="yolo_instance_segmentation")
        self.assertEqual(instance_adapter.metadata.backend, "ultralytics")

    def test_detection_adapter_output_with_mocked_ultralytics(self) -> None:
        """Mock a tiny YOLO result to validate the adapter output contract."""
        with TemporaryDirectory() as tmp:
            weight = Path(tmp) / "model.pt"
            weight.write_text("mock", encoding="utf-8")
            fake_module = types.ModuleType("ultralytics")
            fake_module.YOLO = lambda _: _FakeYoloModel(instance=False)
            with mock.patch.dict(sys.modules, {"ultralytics": fake_module}):
                with mock.patch("importlib.util.find_spec", return_value=object()):
                    adapter = UltralyticsDetectionAdapter({"id": "yolo_detection", "weights": str(weight)})
                    predictions = adapter.predict_tile(np.zeros((3, 16, 20), dtype=np.uint8), {"tile_id": "t0"})

        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0].label, "vehicle")
        self.assertEqual(predictions[0].class_id, 2)
        self.assertEqual(predictions[0].bbox, [1.0, 2.0, 8.0, 10.0])

    def test_instance_adapter_output_with_mocked_ultralytics(self) -> None:
        """Mock a tiny YOLO segmentation result to validate instance output fields."""
        with TemporaryDirectory() as tmp:
            weight = Path(tmp) / "model.pt"
            weight.write_text("mock", encoding="utf-8")
            fake_module = types.ModuleType("ultralytics")
            fake_module.YOLO = lambda _: _FakeYoloModel(instance=True)
            with mock.patch.dict(sys.modules, {"ultralytics": fake_module}):
                with mock.patch("importlib.util.find_spec", return_value=object()):
                    adapter = UltralyticsInstanceSegmentationAdapter(
                        {"id": "yolo_instance_segmentation", "weights": str(weight)}
                    )
                    predictions = adapter.predict_tile(np.zeros((16, 20, 3), dtype=np.uint8), {"tile_id": "t0"})

        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0].label, "vehicle")
        self.assertGreaterEqual(len(predictions[0].polygon or []), 4)
        self.assertEqual(predictions[0].mask.shape, (16, 20))

    def test_install_message_constant(self) -> None:
        """The shared install hint should be explicit enough for API/MCP users."""
        self.assertIn("ultralytics", INSTALL_DETECTION_MESSAGE.lower())
        self.assertIn("sahi", INSTALL_DETECTION_MESSAGE.lower())


class _FakeYoloModel:
    names = {2: "vehicle"}

    def __init__(self, instance: bool) -> None:
        self.instance = instance

    def predict(self, *_args, **_kwargs):
        return [_FakeResult(instance=self.instance)]


class _FakeResult:
    names = {2: "vehicle"}

    def __init__(self, instance: bool) -> None:
        self.boxes = types.SimpleNamespace(
            xyxy=np.asarray([[1, 2, 8, 10]], dtype=np.float32),
            conf=np.asarray([0.87], dtype=np.float32),
            cls=np.asarray([2], dtype=np.float32),
        )
        self.masks = _FakeMasks() if instance else None


class _FakeMasks:
    def __init__(self) -> None:
        self.xy = [
            np.asarray(
                [[1, 2], [8, 2], [8, 10], [1, 10]],
                dtype=np.float32,
            )
        ]
        data = np.zeros((1, 16, 20), dtype=np.float32)
        data[0, 2:10, 1:8] = 1.0
        self.data = data


if __name__ == "__main__":
    unittest.main()
