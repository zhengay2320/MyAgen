from __future__ import annotations

import os
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

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

    def _write_models_yaml(self, text: str) -> None:
        path = self.root / "models.yaml"
        path.write_text(text.strip() + "\n", encoding="utf-8")
        os.environ["RS_MODELS_CONFIG"] = str(path)
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
