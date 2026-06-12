from __future__ import annotations

import os
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from rs_service.adapters.base import ChangePrediction, SegmentationPrediction, SuperResolutionPrediction
from rs_service.adapters.external_subprocess_adapter import ExternalSubprocessAdapter
from rs_service.settings import get_settings


class ExternalSubprocessAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_models_config = os.environ.get("RS_MODELS_CONFIG")
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        if self.previous_models_config is None:
            os.environ.pop("RS_MODELS_CONFIG", None)
        else:
            os.environ["RS_MODELS_CONFIG"] = self.previous_models_config
        self._clear_registry_caches()
        self.tmp.cleanup()

    def test_builtin_fake_model_stays_inprocess(self) -> None:
        """Built-in fake adapters should not be forced through subprocess."""
        from rs_service.registry import get_adapter

        adapter = get_adapter("object_detection", model_id="fake_detection")
        self.assertNotIsInstance(adapter, ExternalSubprocessAdapter)
        predictions = adapter.predict_tile(_bright_tile(), {"tile_id": "t0"})
        self.assertGreaterEqual(len(predictions), 1)

    def test_default_fake_external_models_are_available(self) -> None:
        """Repository models.yaml should expose default fake_external subprocess models."""
        self._use_default_models_yaml()
        from rs_service.registry import get_adapter, list_models

        model_ids = {item["id"] for item in list_models()["models"]}
        for model_id in [
            "fake_external_detection",
            "fake_external_segmentation",
            "fake_external_instance",
            "fake_external_change",
            "fake_external_super_resolution",
        ]:
            self.assertIn(model_id, model_ids)
        adapter = get_adapter("object_detection", model_id="fake_external_detection")
        self.assertIsInstance(adapter, ExternalSubprocessAdapter)

    def test_default_fake_external_segmentation_returns_mask(self) -> None:
        """Default fake_external_segmentation should return a tile mask."""
        self._use_default_models_yaml()
        from rs_service.registry import get_adapter

        adapter = get_adapter("semantic_segmentation", model_id="fake_external_segmentation")
        prediction = adapter.predict_tile(_bright_tile(), {"tile_id": "t0"})
        self.assertIsInstance(prediction, SegmentationPrediction)
        self.assertEqual(prediction.mask.shape, (24, 24))

    def test_default_fake_external_change_accepts_tile_t2(self) -> None:
        """Default fake_external_change should pass tile_t2 to the worker."""
        self._use_default_models_yaml()
        from rs_service.registry import get_adapter

        tile = _bright_tile()
        changed = tile.copy()
        changed[:, 8:20, 8:20] = 20
        adapter = get_adapter("change_detection", model_id="fake_external_change")
        prediction = adapter.predict_tile(tile, {"tile_id": "t0"}, tile_t2=changed, threshold=0.1)
        self.assertIsInstance(prediction, ChangePrediction)
        self.assertEqual(prediction.probability.shape, (24, 24))
        self.assertGreater(float(prediction.probability.max()), 0.0)

    def test_default_fake_external_super_resolution_returns_image(self) -> None:
        """Default fake_external_super_resolution should return an upscaled image."""
        self._use_default_models_yaml()
        from rs_service.registry import get_adapter

        adapter = get_adapter("super_resolution", model_id="fake_external_super_resolution", scale=2)
        prediction = adapter.predict_tile(_bright_tile(), {"tile_id": "t0"}, scale=2)
        self.assertIsInstance(prediction, SuperResolutionPrediction)
        self.assertEqual(prediction.image.shape, (3, 48, 48))

    def test_registry_returns_external_adapter_for_subprocess_runner(self) -> None:
        """YAML runner=subprocess should route through ExternalSubprocessAdapter."""
        self._write_models_yaml(
            """
models:
  - id: fake_external_detection
    task: object_detection
    backend: fake_external
    framework: numpy
    runner: subprocess
    conda_env: null
    entrypoint: rs_service.workers.fake_external_infer
    runner_timeout_sec: 60
"""
        )
        from rs_service.registry import get_adapter, list_models

        adapter = get_adapter("object_detection", model_id="fake_external_detection")
        self.assertIsInstance(adapter, ExternalSubprocessAdapter)
        model_ids = {item["id"] for item in list_models()["models"]}
        self.assertIn("fake_external_detection", model_ids)

    def test_subprocess_fake_external_worker_runs_one_tile(self) -> None:
        """A subprocess runner with empty conda_env should use current Python."""
        self._write_models_yaml(
            """
models:
  - id: fake_external_detection
    task: object_detection
    backend: fake_external
    framework: numpy
    runner: subprocess
    conda_env: null
    entrypoint: rs_service.workers.fake_external_infer
    runner_timeout_sec: 60
"""
        )
        from rs_service.registry import get_adapter

        adapter = get_adapter("object_detection", model_id="fake_external_detection")
        predictions = adapter.predict(_bright_tile(), context={"tile": types.SimpleNamespace(tile_id="t0", x0=0, y0=0)})
        self.assertGreaterEqual(len(predictions), 1)
        self.assertIn("bbox", predictions[0])
        self.assertEqual(predictions[0]["label"], "bright_object")

    def test_subprocess_instance_mask_is_returned_from_npy(self) -> None:
        """Instance workers should pass binary masks back through .npy files."""
        self._write_models_yaml(
            """
models:
  - id: fake_external_instance
    task: instance_segmentation
    backend: fake_external
    framework: numpy
    runner: subprocess
    conda_env: null
    entrypoint: rs_service.workers.fake_external_infer
    runner_timeout_sec: 60
"""
        )
        from rs_service.registry import get_adapter

        adapter = get_adapter("instance_segmentation", model_id="fake_external_instance")
        predictions = adapter.predict_tile(_bright_tile(), {"tile_id": "t0"})
        self.assertGreaterEqual(len(predictions), 1)
        self.assertEqual(predictions[0].mask.shape, (24, 24))
        self.assertGreater(int(predictions[0].mask.sum()), 0)

    def test_subprocess_worker_failure_is_readable(self) -> None:
        """A missing worker module should return a clear subprocess error."""
        self._write_models_yaml(
            """
models:
  - id: broken_external_detection
    task: object_detection
    backend: fake_external
    framework: numpy
    runner: subprocess
    conda_env: null
    entrypoint: rs_service.workers.does_not_exist
    runner_timeout_sec: 60
"""
        )
        from rs_service.registry import get_adapter

        adapter = get_adapter("object_detection", model_id="broken_external_detection")
        with self.assertRaises(RuntimeError) as raised:
            adapter.predict(_bright_tile(), context={"tile": types.SimpleNamespace(tile_id="t0", x0=0, y0=0)})
        message = str(raised.exception)
        self.assertIn("Subprocess worker failed", message)
        self.assertIn("rs_service.workers.does_not_exist", message)

    def test_runner_http_is_reserved_value_error(self) -> None:
        """runner=http is reserved and should fail with the documented ValueError."""
        self._write_models_yaml(
            """
models:
  - id: future_http_detection
    task: object_detection
    backend: http
    framework: remote
    runner: http
"""
        )
        from rs_service.registry import get_adapter

        with self.assertRaises(ValueError) as raised:
            get_adapter("object_detection", model_id="future_http_detection")
        self.assertIn("runner=http is reserved", str(raised.exception))

    def _write_models_yaml(self, text: str) -> None:
        path = self.root / "models.yaml"
        path.write_text(text.strip() + "\n", encoding="utf-8")
        os.environ["RS_MODELS_CONFIG"] = str(path)
        self._clear_registry_caches()

    def _use_default_models_yaml(self) -> None:
        os.environ["RS_MODELS_CONFIG"] = str(Path(__file__).resolve().parents[1] / "configs" / "models.yaml")
        self._clear_registry_caches()

    def _clear_registry_caches(self) -> None:
        get_settings.cache_clear()
        from rs_service import registry

        registry._load_yaml_models_cached.cache_clear()


def _bright_tile() -> np.ndarray:
    tile = np.zeros((3, 24, 24), dtype=np.uint8)
    tile[:, 4:16, 5:18] = 230
    return tile


if __name__ == "__main__":
    unittest.main()
