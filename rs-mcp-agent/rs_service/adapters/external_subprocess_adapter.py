from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from rs_service.adapters.base import (
    AdapterMetadata,
    BaseAdapter,
    ChangePrediction,
    DetectionPrediction,
    InstancePrediction,
    SegmentationPrediction,
    SuperResolutionPrediction,
)


RUNNER_CONFIG_KEYS = {"runner", "conda_env", "entrypoint", "runner_timeout_sec"}


class ExternalSubprocessAdapter(BaseAdapter):
    """Adapter that delegates single-tile inference to an external worker process."""

    def __init__(self, model_config: dict[str, Any]) -> None:
        """Create an external adapter from a YAML model config."""
        self.model_config = dict(model_config)
        self.task = str(self.model_config.get("task", "unknown"))
        self.entrypoint = str(self.model_config.get("entrypoint") or "")
        self.conda_env = str(self.model_config.get("conda_env") or "")
        self.timeout_sec = int(self.model_config.get("runner_timeout_sec") or 600)
        if not self.entrypoint:
            raise ValueError(f"Subprocess model {self.model_config.get('id')!r} is missing entrypoint.")
        self.metadata = AdapterMetadata(
            id=str(self.model_config.get("id", "external_subprocess")),
            task=self.task,
            backend=str(self.model_config.get("backend", "subprocess")),
            framework=str(self.model_config.get("framework", "external")),
            weights=self.model_config.get("weights") or self.model_config.get("weight") or self.model_config.get("checkpoint"),
            description=str(self.model_config.get("description", "External subprocess adapter.")),
        )

    def predict(self, tile: np.ndarray, context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Run detection-like tile inference and return pipeline-compatible dictionaries."""
        tile_info = _tile_info_from_context(context)
        response = self._run_worker({"tile": tile}, tile_info, dict((context or {}).get("adapter_kwargs", {})))
        return list(response.get("predictions", []))

    def predict_tile(self, tile_array: np.ndarray, tile_info: dict[str, Any], **kwargs: Any) -> Any:
        """Run one tile through the configured worker and normalize the response."""
        kwargs = dict(kwargs)
        arrays = {"tile": tile_array}
        tile_t2 = kwargs.pop("tile_t2", None)
        after_tile = kwargs.pop("after_tile", None)
        if tile_t2 is None:
            tile_t2 = after_tile
        if tile_t2 is not None:
            arrays["tile_t2"] = tile_t2
        response = self._run_worker(arrays, tile_info, kwargs)
        if self.task in {"object_detection", "oriented_detection"}:
            return [_detection_from_dict(item) for item in response.get("predictions", [])]
        if self.task == "instance_segmentation":
            return [_instance_from_dict(item, tile_array.shape[-2:]) for item in response.get("predictions", [])]
        if self.task == "semantic_segmentation":
            mask = _load_response_array(response, "mask_path")
            probability = _load_optional_response_array(response, "probability_path")
            if mask is None and probability is not None:
                mask = np.argmax(probability, axis=0).astype(np.uint8)
            if mask is None:
                raise RuntimeError(f"Subprocess worker {self.entrypoint} did not return mask_path or probability_path.")
            return SegmentationPrediction(
                mask=mask.astype(np.uint8),
                probabilities=probability.astype(np.float32) if probability is not None else None,
                class_names={int(k): str(v) for k, v in dict(response.get("class_names", {})).items()},
                metadata=dict(response.get("metadata", {})),
            )
        if self.task == "change_detection":
            probability = _load_response_array(response, "probability_path")
            mask = _load_optional_response_array(response, "mask_path")
            threshold = float(response.get("threshold", kwargs.get("threshold", 0.5)))
            if probability is None:
                raise RuntimeError(f"Subprocess worker {self.entrypoint} did not return probability_path.")
            if mask is None:
                mask = (probability >= threshold).astype(np.uint8)
            return ChangePrediction(mask=mask.astype(np.uint8), probability=probability.astype(np.float32), threshold=threshold)
        if self.task == "super_resolution":
            image = _load_response_array(response, "image_path")
            if image is None:
                raise RuntimeError(f"Subprocess worker {self.entrypoint} did not return image_path.")
            return SuperResolutionPrediction(image=image, scale=int(response.get("scale", kwargs.get("scale", self.model_config.get("scale", 2)))))
        raise RuntimeError(f"Subprocess runner does not support task={self.task!r}.")

    def predict_proba(
        self,
        tile: np.ndarray,
        tile_b: np.ndarray | None = None,
        context: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Backward-compatible probability API for semantic and change detection."""
        tile_info = _tile_info_from_context(context)
        if tile_b is not None:
            prediction = self.predict_tile(tile, tile_info, tile_t2=tile_b)
            if isinstance(prediction, ChangePrediction):
                return prediction.probability
            raise RuntimeError(f"Subprocess worker {self.entrypoint} did not return change probabilities.")
        prediction = self.predict_tile(tile, tile_info)
        if isinstance(prediction, SegmentationPrediction) and prediction.probabilities is not None:
            return prediction.probabilities
        if isinstance(prediction, SegmentationPrediction):
            class_count = int(prediction.mask.max()) + 1 if prediction.mask.size else 1
            output = np.zeros((class_count, prediction.mask.shape[0], prediction.mask.shape[1]), dtype=np.float32)
            for class_id in range(class_count):
                output[class_id] = (prediction.mask == class_id).astype(np.float32)
            return output
        raise RuntimeError(f"Subprocess worker {self.entrypoint} did not return segmentation probabilities.")

    def upscale(self, tile: np.ndarray, context: dict[str, Any] | None = None) -> np.ndarray:
        """Backward-compatible super-resolution API."""
        tile_info = _tile_info_from_context(context)
        prediction = self.predict_tile(tile, tile_info, scale=int(self.model_config.get("scale", 2)))
        if not isinstance(prediction, SuperResolutionPrediction):
            raise RuntimeError(f"Subprocess worker {self.entrypoint} did not return a super-resolution image.")
        return prediction.image

    def _run_worker(self, arrays: dict[str, np.ndarray], tile_info: dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        """Serialize tile arrays, execute the worker, and read its response."""
        with tempfile.TemporaryDirectory(prefix="rs_mcp_worker_") as tmp_text:
            tmp_dir = Path(tmp_text)
            array_paths: dict[str, str] = {}
            for name, array in arrays.items():
                path = tmp_dir / f"{name}.npy"
                np.save(path, np.asarray(array))
                array_paths[name] = str(path)
            request_path = tmp_dir / "request.json"
            response_path = tmp_dir / "response.json"
            request = {
                "model_config": _worker_model_config(self.model_config),
                "task": self.task,
                "backend": self.model_config.get("backend"),
                "tile_info": _jsonable(tile_info),
                "arrays": array_paths,
                "kwargs": _jsonable(kwargs),
            }
            request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
            command = self._command(request_path, response_path)
            completed = self._run_command(command)
            if completed.returncode != 0:
                raise RuntimeError(
                    "Subprocess worker failed "
                    f"model_id={self.metadata.id} entrypoint={self.entrypoint} returncode={completed.returncode}. "
                    f"stderr={completed.stderr.strip()} stdout={completed.stdout.strip()}"
                )
            if not response_path.exists():
                raise RuntimeError(
                    f"Subprocess worker {self.entrypoint} finished without writing response.json for model_id={self.metadata.id}."
                )
            response = json.loads(response_path.read_text(encoding="utf-8"))
            if not response.get("ok", True):
                raise RuntimeError(
                    f"Subprocess worker failed for model_id={self.metadata.id}: {response.get('error', 'unknown error')}"
                )
            for key in ("mask_path", "probability_path", "image_path"):
                if response.get(key):
                    response[f"{key}_array"] = np.load(Path(str(response[key])), allow_pickle=False)
            return response

    def _command(self, request_path: Path, response_path: Path) -> list[str]:
        """Build the worker command for conda or current-Python execution."""
        if self.conda_env:
            return [
                "conda",
                "run",
                "-n",
                self.conda_env,
                "python",
                "-m",
                self.entrypoint,
                "--request",
                str(request_path),
                "--response",
                str(response_path),
            ]
        return [
            sys.executable,
            "-m",
            self.entrypoint,
            "--request",
            str(request_path),
            "--response",
            str(response_path),
        ]

    def _run_command(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        """Execute the worker command with a project-local PYTHONPATH."""
        env = os.environ.copy()
        root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        try:
            return subprocess.run(
                command,
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                timeout=self.timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Subprocess worker timed out after {self.timeout_sec}s for model_id={self.metadata.id} "
                f"entrypoint={self.entrypoint}."
            ) from exc
        except FileNotFoundError as exc:
            hint = " Install conda or leave conda_env empty to run the worker with the current Python."
            raise RuntimeError(f"Subprocess runner command not found: {command[0]!r}.{hint}") from exc


def _worker_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return model config passed to workers without runner-only fields."""
    return {key: value for key, value in config.items() if key not in RUNNER_CONFIG_KEYS}


def _tile_info_from_context(context: dict[str, Any] | None) -> dict[str, Any]:
    """Extract serializable tile metadata from pipeline context."""
    tile = (context or {}).get("tile")
    if tile is None:
        return {}
    return {
        "tile_id": getattr(tile, "tile_id", None),
        "x_off": getattr(tile, "x0", getattr(tile, "x_off", 0)),
        "y_off": getattr(tile, "y0", getattr(tile, "y_off", 0)),
        "width": getattr(tile, "width", None),
        "height": getattr(tile, "height", None),
    }


def _detection_from_dict(item: dict[str, Any]) -> DetectionPrediction:
    """Convert a worker detection dictionary to a dataclass."""
    return DetectionPrediction(
        label=str(item.get("label", item.get("class_name", "target"))),
        score=float(item.get("score", 0.0)),
        bbox=[float(value) for value in item.get("bbox", [0, 0, 0, 0])],
        class_id=int(item.get("class_id", 0) or 0),
        polygon=_optional_points(item.get("polygon")),
        rotated_box=dict(item.get("rotated_box") or {}) or None,
        attributes=dict(item.get("attributes", {})),
    )


def _instance_from_dict(item: dict[str, Any], shape: tuple[int, int]) -> InstancePrediction:
    """Convert a worker instance dictionary to a dataclass."""
    mask = np.zeros(shape, dtype=np.uint8)
    polygon = _optional_points(item.get("mask_polygon") or item.get("polygon"))
    return InstancePrediction(
        label=str(item.get("label", item.get("class_name", "instance"))),
        score=float(item.get("score", 0.0)),
        bbox=[float(value) for value in item.get("bbox", [0, 0, 0, 0])],
        mask=mask,
        class_id=int(item.get("class_id", 0) or 0),
        polygon=polygon,
        attributes=dict(item.get("attributes", {})),
    )


def _optional_points(value: Any) -> list[tuple[float, float]] | None:
    """Normalize optional polygon points from JSON."""
    if not value:
        return None
    return [(float(point[0]), float(point[1])) for point in value]


def _load_response_array(response: dict[str, Any], key: str) -> np.ndarray | None:
    """Load a required response array path when present."""
    if f"{key}_array" in response:
        return np.asarray(response[f"{key}_array"])
    path = response.get(key)
    if not path:
        return None
    return np.load(Path(str(path)), allow_pickle=False)


def _load_optional_response_array(response: dict[str, Any], key: str) -> np.ndarray | None:
    """Load an optional response array path when present."""
    if not response.get(key):
        return None
    return _load_response_array(response, key)


def _jsonable(value: Any) -> Any:
    """Convert common numpy/path values into JSON-compatible values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
