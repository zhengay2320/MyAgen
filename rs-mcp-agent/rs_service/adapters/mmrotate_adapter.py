from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from rs_service.adapters.base import AdapterMetadata, BaseAdapter, DetectionPrediction, ModelBackendUnavailable
from rs_service.adapters.ultralytics_adapter import _to_hwc_uint8
from rs_service.core.geometry import polygon_bounds, rotated_box_to_pixel_polygon


INSTALL_MMROTATE_MESSAGE = (
    "MMRotate backend is unavailable. Install OpenMMLab dependencies in a separate environment, "
    "for example: pip install -U openmim && mim install mmengine mmcv mmdet "
    "&& pip install mmrotate."
)


class MMRotateDetectionAdapter(BaseAdapter):
    """MMRotate oriented object detection adapter with lazy OpenMMLab imports."""

    def __init__(self, model_config: dict[str, Any]) -> None:
        """Create an MMRotate adapter from config/checkpoint settings."""
        self.config = dict(model_config)
        self.config_path = str(self.config.get("config") or self.config.get("config_path") or "")
        self.checkpoint = str(self.config.get("checkpoint") or self.config.get("weights") or self.config.get("weight") or "")
        self.device = str(self.config.get("device", "cpu"))
        self.classes = self.config.get("classes")
        self.angle_unit = str(self.config.get("angle_unit", "auto"))
        self.model: Any | None = None
        self._inference_detector: Any | None = None
        self.metadata = AdapterMetadata(
            id=str(self.config.get("id", "mmrotate_oriented")),
            task="oriented_detection",
            backend="mmrotate",
            framework="mmrotate",
            weights=self.checkpoint,
            description="MMRotate oriented detection adapter.",
        )

    def load(self) -> None:
        """Load MMRotate detector lazily."""
        if self.model is not None:
            return
        if importlib.util.find_spec("mmrotate") is None:
            raise ModelBackendUnavailable(INSTALL_MMROTATE_MESSAGE)
        if not self.config_path:
            raise FileNotFoundError("MMRotate config path is not configured for this model_id.")
        if not self.checkpoint:
            raise FileNotFoundError("MMRotate checkpoint path is not configured for this model_id.")
        config_path = Path(self.config_path)
        checkpoint_path = Path(self.checkpoint)
        if not config_path.exists():
            raise FileNotFoundError(f"MMRotate config not found: {config_path}. Update configs/models.yaml.")
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"MMRotate checkpoint not found: {checkpoint_path}. Place the .pth file there or update configs/models.yaml."
            )
        try:
            from mmdet.apis import inference_detector, init_detector
        except Exception as exc:  # pragma: no cover - depends on optional packages
            raise ModelBackendUnavailable(f"{INSTALL_MMROTATE_MESSAGE} Original error: {exc}") from exc
        self._inference_detector = inference_detector
        self.model = init_detector(str(config_path), str(checkpoint_path), device=self.device)

    def predict_tile(
        self,
        tile_array: np.ndarray,
        tile_info: dict[str, Any],
        **kwargs: Any,
    ) -> list[DetectionPrediction]:
        """Predict tile-local oriented detections."""
        self.load()
        assert self.model is not None
        assert self._inference_detector is not None
        image = _to_hwc_uint8(tile_array)
        try:
            result = self._inference_detector(self.model, image)
        except Exception as exc:
            tile_id = tile_info.get("tile_id", "unknown")
            raise RuntimeError(f"MMRotate tile inference failed for tile_id={tile_id}: {exc}") from exc
        return _parse_oriented(result, _class_names(self.classes, self.model), self.angle_unit, tile_info)

    def predict(self, tile: np.ndarray, context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Backward-compatible dict prediction API used by existing pipelines."""
        return [item.to_dict() for item in self.predict_tile(tile, _context_tile_info(context))]

    def close(self) -> None:
        """Release model references."""
        self.model = None
        self._inference_detector = None


def _parse_oriented(
    result: Any,
    names: dict[int, str],
    angle_unit: str,
    tile_info: dict[str, Any],
) -> list[DetectionPrediction]:
    """Parse MMRotate 2.x/3.x result styles into DetectionPrediction objects."""
    pred_instances = getattr(result, "pred_instances", None)
    if pred_instances is not None:
        bboxes = _as_numpy(getattr(pred_instances, "bboxes", []))
        scores = _as_numpy(getattr(pred_instances, "scores", []))
        labels = _as_numpy(getattr(pred_instances, "labels", []))
        return _build_predictions(bboxes, scores, labels, names, angle_unit, tile_info)
    if isinstance(result, dict) and "pred_instances" in result:
        return _parse_oriented(result["pred_instances"], names, angle_unit, tile_info)
    if isinstance(result, (list, tuple)):
        return _parse_legacy_results(result, names, angle_unit, tile_info)
    if all(hasattr(result, name) for name in ["bboxes", "scores", "labels"]):
        return _build_predictions(_as_numpy(result.bboxes), _as_numpy(result.scores), _as_numpy(result.labels), names, angle_unit, tile_info)
    return []


def _parse_legacy_results(
    result: Any,
    names: dict[int, str],
    angle_unit: str,
    tile_info: dict[str, Any],
) -> list[DetectionPrediction]:
    """Parse list-per-class rotated detections."""
    bboxes_list: list[np.ndarray] = []
    scores_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    for class_id, class_boxes in enumerate(result or []):
        boxes = _as_numpy(class_boxes)
        if boxes.size == 0:
            continue
        boxes = boxes.reshape((-1, boxes.shape[-1]))
        bboxes_list.append(boxes[:, :-1] if boxes.shape[1] in {6, 9} else boxes)
        scores_list.append(boxes[:, -1] if boxes.shape[1] in {6, 9} else np.ones((boxes.shape[0],), dtype=np.float32))
        labels_list.append(np.full((boxes.shape[0],), class_id, dtype=np.int64))
    if not bboxes_list:
        return []
    return _build_predictions(
        np.concatenate(bboxes_list, axis=0),
        np.concatenate(scores_list, axis=0),
        np.concatenate(labels_list, axis=0),
        names,
        angle_unit,
        tile_info,
    )


def _build_predictions(
    bboxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    names: dict[int, str],
    angle_unit: str,
    tile_info: dict[str, Any],
) -> list[DetectionPrediction]:
    """Build unified oriented detections from rotated boxes or polygons."""
    boxes = np.asarray(bboxes)
    if boxes.size == 0:
        return []
    boxes = boxes.reshape((-1, boxes.shape[-1]))
    predictions: list[DetectionPrediction] = []
    for index, raw_box in enumerate(boxes):
        class_id = int(labels[index]) if index < len(labels) else 0
        score = float(scores[index]) if index < len(scores) else 0.0
        polygon, rotated = _oriented_geometry(raw_box, angle_unit)
        bbox = polygon_bounds(polygon)
        predictions.append(
            DetectionPrediction(
                label=names.get(class_id, f"class_{class_id}"),
                score=score,
                bbox=bbox,
                class_id=class_id,
                polygon=polygon,
                rotated_box=rotated,
                attributes={"backend": "mmrotate", "tile_id": tile_info.get("tile_id")},
            )
        )
    return predictions


def _oriented_geometry(raw_box: np.ndarray, angle_unit: str) -> tuple[list[tuple[float, float]], dict[str, float]]:
    """Normalize cx/cy/w/h/angle or 4-point polygon boxes."""
    values = [float(v) for v in raw_box.tolist()]
    if len(values) >= 8:
        polygon = [(values[i], values[i + 1]) for i in range(0, 8, 2)]
        polygon.append(polygon[0])
        bounds = polygon_bounds(polygon)
        rotated = {
            "cx": (bounds[0] + bounds[2]) / 2.0,
            "cy": (bounds[1] + bounds[3]) / 2.0,
            "width": max(1.0, bounds[2] - bounds[0]),
            "height": max(1.0, bounds[3] - bounds[1]),
            "angle_degrees": 0.0,
        }
        return polygon, rotated
    if len(values) < 5:
        raise ValueError(f"Expected rotated bbox with 5 values or polygon with 8 values, got {len(values)}.")
    cx, cy, width, height, angle = values[:5]
    angle_degrees = _angle_to_degrees(angle, angle_unit)
    rotated = {
        "cx": cx,
        "cy": cy,
        "width": max(1.0, width),
        "height": max(1.0, height),
        "angle_degrees": angle_degrees,
    }
    return rotated_box_to_pixel_polygon(cx, cy, width, height, angle_degrees), rotated


def _angle_to_degrees(angle: float, angle_unit: str) -> float:
    """Convert MMRotate angle value to degrees."""
    if angle_unit == "degrees":
        return float(angle)
    if angle_unit == "radians":
        return float(np.degrees(angle))
    return float(np.degrees(angle)) if abs(angle) <= float(np.pi * 2.0) else float(angle)


def _as_numpy(value: Any) -> np.ndarray:
    """Convert tensors, lists, and arrays to numpy arrays."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _class_names(config_classes: Any, model: Any) -> dict[int, str]:
    """Read class names from config or model metadata."""
    if isinstance(config_classes, dict):
        return {int(key): str(value) for key, value in config_classes.items()}
    if isinstance(config_classes, (list, tuple)):
        return {index: str(value) for index, value in enumerate(config_classes)}
    dataset_meta = getattr(model, "dataset_meta", None)
    if isinstance(dataset_meta, dict) and isinstance(dataset_meta.get("classes"), (list, tuple)):
        return {index: str(value) for index, value in enumerate(dataset_meta["classes"])}
    classes = getattr(model, "CLASSES", None)
    if isinstance(classes, (list, tuple)):
        return {index: str(value) for index, value in enumerate(classes)}
    return {}


def _context_tile_info(context: dict[str, Any] | None) -> dict[str, Any]:
    """Extract tile metadata from legacy pipeline context."""
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
