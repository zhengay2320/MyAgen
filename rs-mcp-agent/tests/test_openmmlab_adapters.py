from __future__ import annotations

import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

from rs_service.adapters.base import AdapterMetadata, DetectionPrediction, ModelBackendUnavailable
from rs_service.adapters.mmdet_adapter import INSTALL_MMDET_MESSAGE, MMDetectionInstanceAdapter
from rs_service.adapters.mmrotate_adapter import INSTALL_MMROTATE_MESSAGE, MMRotateDetectionAdapter
from rs_service.core.raster import DEFAULT_TRANSFORM, create_synthetic_array, profile_from_array, write_raster
from rs_service.pipelines.detection import run_oriented_detection
from rs_service.registry import get_adapter, list_models


class OpenMMLabAdapterTests(unittest.TestCase):
    def test_api_import_does_not_require_openmmlab_detection_packages(self) -> None:
        """Service imports and model listing should not import mmdet/mmrotate."""
        import rs_service.api  # noqa: F401

        model_ids = {item["id"] for item in list_models()["models"]}
        self.assertIn("mmdet_maskrcnn_instance", model_ids)
        self.assertIn("mmrotate_dota_oriented", model_ids)

    def test_missing_mmdet_error_is_readable(self) -> None:
        """Missing MMDetection should raise a clear install message."""
        if importlib.util.find_spec("mmdet") is not None:
            self.skipTest("mmdet is installed in this environment")
        adapter = MMDetectionInstanceAdapter(
            {"id": "mmdet_maskrcnn_instance", "config": "missing.py", "checkpoint": "missing.pth"}
        )
        with self.assertRaises(ModelBackendUnavailable) as raised:
            adapter.load()
        self.assertIn("MMDetection backend is unavailable", str(raised.exception))
        self.assertIn("openmim", str(raised.exception))

    def test_missing_mmrotate_error_is_readable(self) -> None:
        """Missing MMRotate should raise a clear install message."""
        if importlib.util.find_spec("mmrotate") is not None:
            self.skipTest("mmrotate is installed in this environment")
        adapter = MMRotateDetectionAdapter(
            {"id": "mmrotate_dota_oriented", "config": "missing.py", "checkpoint": "missing.pth"}
        )
        with self.assertRaises(ModelBackendUnavailable) as raised:
            adapter.load()
        self.assertIn("MMRotate backend is unavailable", str(raised.exception))
        self.assertIn("openmim", str(raised.exception))

    def test_registry_openmmlab_adapters_are_lazy(self) -> None:
        """Registry should construct adapters only when selected."""
        self.assertEqual(get_adapter("instance_segmentation", model_id="mmdet_maskrcnn_instance").metadata.backend, "mmdet")
        self.assertEqual(get_adapter("oriented_detection", model_id="mmrotate_dota_oriented").metadata.backend, "mmrotate")

    def test_mmdet_instance_output_with_mocked_api(self) -> None:
        """Mock MMDetection APIs to validate instance output format."""
        with TemporaryDirectory() as tmp:
            config, checkpoint = _mock_files(tmp)
            with _mock_mmdet_modules(_fake_mmdet_inference):
                adapter = MMDetectionInstanceAdapter(
                    {
                        "id": "mmdet_maskrcnn_instance",
                        "config": str(config),
                        "checkpoint": str(checkpoint),
                        "classes": ["background", "building"],
                    }
                )
                predictions = adapter.predict_tile(np.zeros((3, 16, 20), dtype=np.uint8), {"tile_id": "tile_1"})

        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0].label, "building")
        self.assertEqual(predictions[0].class_id, 1)
        self.assertEqual(predictions[0].mask.shape, (16, 20))
        self.assertGreaterEqual(len(predictions[0].polygon or []), 4)

    def test_mmrotate_output_with_mocked_api(self) -> None:
        """Mock MMRotate/MMDet APIs to validate rotated output format."""
        with TemporaryDirectory() as tmp:
            config, checkpoint = _mock_files(tmp)
            with _mock_mmrotate_modules(_fake_mmrotate_inference):
                adapter = MMRotateDetectionAdapter(
                    {
                        "id": "mmrotate_dota_oriented",
                        "config": str(config),
                        "checkpoint": str(checkpoint),
                        "classes": ["ship"],
                    }
                )
                predictions = adapter.predict_tile(np.zeros((3, 32, 32), dtype=np.uint8), {"tile_id": "tile_1"})

        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0].label, "ship")
        self.assertEqual(predictions[0].class_id, 0)
        self.assertIsNotNone(predictions[0].rotated_box)
        self.assertGreaterEqual(len(predictions[0].polygon or []), 5)

    def test_oriented_pipeline_restores_pixel_and_geo_coordinates(self) -> None:
        """Pipeline should translate oriented local boxes to full pixels and geo polygons."""
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "image.tif"
            output_dir = Path(tmp) / "oriented"
            array = create_synthetic_array(width=40, height=32, bands=3)
            write_raster(image_path, array, profile_from_array(array, transform=DEFAULT_TRANSFORM))
            with mock.patch("rs_service.pipelines.detection.get_adapter", return_value=_StaticOrientedAdapter()):
                manifest = run_oriented_detection(image_path, output_dir=output_dir, tile_size=64, overlap=0)
            payload = json.loads(Path(manifest["outputs"]["geojson"]).read_text(encoding="utf-8"))

        feature = payload["features"][0]
        bbox = feature["properties"]["bbox_pixel"]
        self.assertLess(bbox[0], 10.0)
        self.assertGreater(bbox[2], 10.0)
        geo_x = feature["geometry"]["coordinates"][0][0][0]
        self.assertGreaterEqual(geo_x, DEFAULT_TRANSFORM[2])


def _mock_files(tmp: str) -> tuple[Path, Path]:
    config = Path(tmp) / "model.py"
    checkpoint = Path(tmp) / "model.pth"
    config.write_text("# config", encoding="utf-8")
    checkpoint.write_text("checkpoint", encoding="utf-8")
    return config, checkpoint


def _mock_mmdet_modules(inference_func):
    mmdet_module = types.ModuleType("mmdet")
    mmdet_module.__path__ = []
    apis_module = types.ModuleType("mmdet.apis")
    apis_module.init_detector = _fake_init_detector
    apis_module.inference_detector = inference_func
    return _PatchedModules({"mmdet": mmdet_module, "mmdet.apis": apis_module})


def _mock_mmrotate_modules(inference_func):
    mmrotate_module = types.ModuleType("mmrotate")
    mmrotate_module.__path__ = []
    mmdet_module = types.ModuleType("mmdet")
    mmdet_module.__path__ = []
    apis_module = types.ModuleType("mmdet.apis")
    apis_module.init_detector = _fake_init_detector
    apis_module.inference_detector = inference_func
    modules = {"mmrotate": mmrotate_module, "mmdet": mmdet_module, "mmdet.apis": apis_module}
    return _PatchedModules(modules)


class _PatchedModules:
    def __init__(self, modules: dict[str, types.ModuleType]) -> None:
        self.modules = modules
        self.patch_modules = mock.patch.dict(sys.modules, modules)
        self.patch_find = mock.patch("importlib.util.find_spec", return_value=object())

    def __enter__(self):
        self.patch_modules.__enter__()
        self.patch_find.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.patch_find.__exit__(exc_type, exc, tb)
        return self.patch_modules.__exit__(exc_type, exc, tb)


def _fake_init_detector(_config: str, _checkpoint: str, device: str = "cpu"):
    return types.SimpleNamespace(dataset_meta={"classes": ["ship", "building"]}, device=device)


def _fake_mmdet_inference(_model, image: np.ndarray):
    height, width = image.shape[:2]
    mask = np.zeros((1, height, width), dtype=np.uint8)
    mask[0, 3:11, 4:12] = 1
    pred_instances = types.SimpleNamespace(
        bboxes=np.asarray([[4, 3, 12, 11]], dtype=np.float32),
        scores=np.asarray([0.91], dtype=np.float32),
        labels=np.asarray([1], dtype=np.int64),
        masks=mask,
    )
    return types.SimpleNamespace(pred_instances=pred_instances)


def _fake_mmrotate_inference(_model, _image: np.ndarray):
    pred_instances = types.SimpleNamespace(
        bboxes=np.asarray([[16, 16, 12, 6, np.pi / 6]], dtype=np.float32),
        scores=np.asarray([0.88], dtype=np.float32),
        labels=np.asarray([0], dtype=np.int64),
    )
    return types.SimpleNamespace(pred_instances=pred_instances)


class _StaticOrientedAdapter:
    metadata = AdapterMetadata(
        id="static_oriented",
        task="oriented_detection",
        backend="test",
        framework="unittest",
    )

    def predict(self, _tile: np.ndarray, context: dict | None = None) -> list[dict]:
        return [
            DetectionPrediction(
                label="ship",
                score=0.9,
                bbox=[5, 5, 15, 15],
                class_id=1,
                rotated_box={"cx": 10.0, "cy": 10.0, "width": 12.0, "height": 6.0, "angle_degrees": 30.0},
            ).to_dict()
        ]


if __name__ == "__main__":
    unittest.main()
