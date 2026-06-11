from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from rs_service.adapters.base import DetectionPrediction, InstancePrediction


RequestHandler = Callable[[dict[str, Any], Path], dict[str, Any]]


def worker_main(handler: RequestHandler) -> None:
    """Run a JSON request/response worker without writing ordinary logs to stdout."""
    parser = argparse.ArgumentParser(description="rs-mcp-agent tile inference worker")
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args()
    request_path = Path(args.request)
    response_path = Path(args.response)
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        payload = handler(request, response_path)
        payload.setdefault("ok", True)
        write_response(response_path, payload)
    except Exception as exc:  # pragma: no cover - exercised by subprocess integration tests
        write_response(response_path, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def write_response(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON response using stderr-safe worker behavior."""
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def load_array(request: dict[str, Any], name: str = "tile") -> np.ndarray:
    """Load a named numpy array from a worker request."""
    arrays = dict(request.get("arrays", {}))
    path = arrays.get(name)
    if not path:
        raise ValueError(f"Worker request is missing array {name!r}.")
    return np.load(Path(str(path)), allow_pickle=False)


def model_config(request: dict[str, Any]) -> dict[str, Any]:
    """Return model config from a worker request."""
    return dict(request.get("model_config", {}))


def tile_info(request: dict[str, Any]) -> dict[str, Any]:
    """Return tile metadata from a worker request."""
    return dict(request.get("tile_info", {}))


def kwargs(request: dict[str, Any]) -> dict[str, Any]:
    """Return adapter kwargs from a worker request."""
    return dict(request.get("kwargs", {}))


def detection_payload(predictions: list[DetectionPrediction] | list[Any]) -> dict[str, Any]:
    """Serialize detection predictions for the external adapter."""
    return {"predictions": [_prediction_to_dict(item) for item in predictions]}


def instance_payload(predictions: list[InstancePrediction] | list[Any]) -> dict[str, Any]:
    """Serialize instance predictions for the external adapter."""
    records: list[dict[str, Any]] = []
    for item in predictions:
        if isinstance(item, InstancePrediction):
            records.append(
                {
                    "label": item.label,
                    "score": item.score,
                    "bbox": item.bbox,
                    "mask_polygon": item.polygon,
                    "polygon": item.polygon,
                    "class_id": item.class_id,
                    "attributes": item.attributes,
                }
            )
        else:
            records.append(dict(item))
    return {"predictions": records}


def write_array(response_path: Path, name: str, array: np.ndarray) -> str:
    """Write an array beside the worker response and return its path."""
    path = response_path.with_name(f"{name}.npy")
    np.save(path, np.asarray(array))
    return str(path)


def _prediction_to_dict(item: Any) -> dict[str, Any]:
    """Serialize dataclass or dictionary predictions."""
    if isinstance(item, DetectionPrediction):
        return item.to_dict()
    if hasattr(item, "to_dict"):
        return dict(item.to_dict())
    return dict(item)


def _jsonable(value: Any) -> Any:
    """Convert numpy values to JSON-compatible objects."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def stderr(message: str) -> None:
    """Write worker diagnostics to stderr, never stdout."""
    print(message, file=sys.stderr)
