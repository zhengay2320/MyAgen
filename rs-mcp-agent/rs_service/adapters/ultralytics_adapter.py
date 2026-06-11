from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from rs_service.adapters.base import (
    AdapterMetadata,
    BaseAdapter,
    DetectionPrediction,
    InstancePrediction,
    ModelBackendUnavailable,
)


INSTALL_DETECTION_MESSAGE = (
    'Ultralytics backend is unavailable. Install with pip install ".[detection]" '
    "or pip install ultralytics sahi."
)


class UltralyticsDetectionAdapter(BaseAdapter):
    """Ultralytics YOLO object detection adapter with lazy framework imports."""

    def __init__(self, model_config: dict[str, Any]) -> None:
        """Create an adapter from a models.yaml style config dict."""
        self.config = dict(model_config)
        weight = str(self.config.get("weight") or self.config.get("weights") or "")
        self.weight = weight
        self.confidence_threshold = float(self.config.get("confidence_threshold", self.config.get("conf", 0.25)))
        self.iou_threshold = float(self.config.get("iou_threshold", self.config.get("iou", 0.45)))
        self.device = self.config.get("device")
        self.imgsz = self.config.get("imgsz")
        self.model: Any | None = None
        self.metadata = AdapterMetadata(
            id=str(self.config.get("id", "yolo_detection")),
            task="object_detection",
            backend="ultralytics",
            framework="ultralytics",
            weights=weight,
            description="Ultralytics YOLO detection adapter.",
        )

    def load(self) -> None:
        """Load YOLO weights, raising readable errors for missing deps or files."""
        if self.model is not None:
            return
        if importlib.util.find_spec("ultralytics") is None:
            raise ModelBackendUnavailable(INSTALL_DETECTION_MESSAGE)
        if not self.weight:
            raise FileNotFoundError("Ultralytics model weight is not configured for this model_id.")
        weight_path = Path(self.weight)
        if not weight_path.exists():
            raise FileNotFoundError(
                f"Ultralytics model weight not found: {weight_path}. "
                "Place the .pt file there or update configs/models.yaml."
            )
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - depends on optional package internals
            raise ModelBackendUnavailable(f"{INSTALL_DETECTION_MESSAGE} Original error: {exc}") from exc
        self.model = YOLO(str(weight_path))

    def predict_tile(
        self,
        tile_array: np.ndarray,
        tile_info: dict[str, Any],
        **kwargs: Any,
    ) -> list[DetectionPrediction]:
        """Predict tile-local bounding boxes using Ultralytics YOLO."""
        self.load()
        assert self.model is not None
        image = _to_hwc_uint8(tile_array)
        conf = float(kwargs.get("confidence_threshold", self.confidence_threshold))
        iou = float(kwargs.get("iou_threshold", self.iou_threshold))
        result_list = self.model.predict(
            image,
            conf=conf,
            iou=iou,
            device=kwargs.get("device", self.device),
            imgsz=kwargs.get("imgsz", self.imgsz),
            verbose=False,
        )
        if not result_list:
            return []
        result = result_list[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        xyxy = _as_numpy(getattr(boxes, "xyxy", []))
        scores = _as_numpy(getattr(boxes, "conf", []))
        classes = _as_numpy(getattr(boxes, "cls", []))
        names = _names_map(getattr(result, "names", None), getattr(self.model, "names", None))
        predictions: list[DetectionPrediction] = []
        for index, bbox in enumerate(xyxy):
            class_id = int(classes[index]) if index < len(classes) else 0
            label = str(names.get(class_id, f"class_{class_id}"))
            score = float(scores[index]) if index < len(scores) else 0.0
            predictions.append(
                DetectionPrediction(
                    label=label,
                    score=score,
                    bbox=[float(v) for v in bbox[:4]],
                    class_id=class_id,
                    attributes={"backend": "ultralytics", "tile_id": tile_info.get("tile_id")},
                )
            )
        return predictions

    def predict(self, tile: np.ndarray, context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Backward-compatible dict prediction API used by existing pipelines."""
        kwargs = _context_adapter_kwargs(context)
        return [prediction.to_dict() for prediction in self.predict_tile(tile, _context_tile_info(context), **kwargs)]

    def close(self) -> None:
        """Release the loaded model reference."""
        self.model = None


class UltralyticsInstanceSegmentationAdapter(UltralyticsDetectionAdapter):
    """Ultralytics YOLO segmentation adapter with mask polygon extraction."""

    def __init__(self, model_config: dict[str, Any]) -> None:
        """Create an instance segmentation adapter from config."""
        super().__init__(model_config)
        self.metadata = AdapterMetadata(
            id=str(self.config.get("id", "yolo_instance_segmentation")),
            task="instance_segmentation",
            backend="ultralytics",
            framework="ultralytics",
            weights=self.weight,
            description="Ultralytics YOLO instance segmentation adapter.",
        )

    def predict_tile(
        self,
        tile_array: np.ndarray,
        tile_info: dict[str, Any],
        **kwargs: Any,
    ) -> list[InstancePrediction]:
        """Predict tile-local instance boxes and masks using Ultralytics YOLO segmentation."""
        self.load()
        assert self.model is not None
        image = _to_hwc_uint8(tile_array)
        conf = float(kwargs.get("confidence_threshold", self.confidence_threshold))
        iou = float(kwargs.get("iou_threshold", self.iou_threshold))
        result_list = self.model.predict(
            image,
            conf=conf,
            iou=iou,
            device=kwargs.get("device", self.device),
            imgsz=kwargs.get("imgsz", self.imgsz),
            verbose=False,
        )
        if not result_list:
            return []
        result = result_list[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        xyxy = _as_numpy(getattr(boxes, "xyxy", []))
        scores = _as_numpy(getattr(boxes, "conf", []))
        classes = _as_numpy(getattr(boxes, "cls", []))
        names = _names_map(getattr(result, "names", None), getattr(self.model, "names", None))
        polygons = _mask_polygons(getattr(result, "masks", None))
        masks = _mask_arrays(getattr(result, "masks", None))
        predictions: list[InstancePrediction] = []
        for index, bbox in enumerate(xyxy):
            class_id = int(classes[index]) if index < len(classes) else 0
            label = str(names.get(class_id, f"class_{class_id}"))
            score = float(scores[index]) if index < len(scores) else 0.0
            polygon = polygons[index] if index < len(polygons) else _bbox_polygon(bbox)
            mask = masks[index] if index < len(masks) else _bbox_mask(bbox, image.shape[:2])
            predictions.append(
                InstancePrediction(
                    label=label,
                    score=score,
                    bbox=[float(v) for v in bbox[:4]],
                    mask=mask.astype(np.uint8),
                    class_id=class_id,
                    polygon=polygon,
                    attributes={"backend": "ultralytics", "tile_id": tile_info.get("tile_id")},
                )
            )
        return predictions

    def predict(self, tile: np.ndarray, context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Backward-compatible dict prediction API used by existing pipelines."""
        kwargs = _context_adapter_kwargs(context)
        return [
            {
                "label": item.label,
                "score": item.score,
                "bbox": item.bbox,
                "mask_polygon": item.polygon,
                "class_id": item.class_id,
                "attributes": item.attributes,
            }
            for item in self.predict_tile(tile, _context_tile_info(context), **kwargs)
        ]


def _to_hwc_uint8(tile_array: np.ndarray) -> np.ndarray:
    """Normalize CHW/HWC tile arrays to HWC uint8 RGB-like arrays."""
    array = np.asarray(tile_array)
    if array.ndim == 2:
        array = array[:, :, None]
    elif array.ndim == 3 and array.shape[0] <= 8 and array.shape[0] < array.shape[-1]:
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 3:
        raise ValueError(f"Expected 2D or 3D tile array, got shape {array.shape}.")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] > 3:
        array = array[..., :3]
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _as_numpy(value: Any) -> np.ndarray:
    """Convert tensors, lists, and arrays to numpy arrays."""
    if value is None:
        return np.asarray([])
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _names_map(*candidates: Any) -> dict[int, str]:
    """Return a class id to class name mapping from Ultralytics result/model objects."""
    for names in candidates:
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        if isinstance(names, (list, tuple)):
            return {index: str(value) for index, value in enumerate(names)}
    return {}


def _mask_polygons(masks: Any) -> list[list[tuple[float, float]]]:
    """Extract mask polygons from an Ultralytics masks object."""
    xy = getattr(masks, "xy", None)
    if xy is None:
        return []
    polygons: list[list[tuple[float, float]]] = []
    for item in xy:
        coords = _as_numpy(item)
        if coords.ndim != 2 or coords.shape[0] < 3:
            polygons.append([])
            continue
        polygon = [(float(x), float(y)) for x, y in coords[:, :2]]
        if polygon and polygon[0] != polygon[-1]:
            polygon.append(polygon[0])
        polygons.append(polygon)
    return polygons


def _mask_arrays(masks: Any) -> list[np.ndarray]:
    """Extract binary mask arrays from an Ultralytics masks object."""
    data = getattr(masks, "data", None)
    if data is None:
        return []
    array = _as_numpy(data)
    if array.ndim == 2:
        array = array[None, :, :]
    return [(item > 0.5).astype(np.uint8) for item in array]


def _bbox_polygon(bbox: Any) -> list[tuple[float, float]]:
    """Create a simple polygon from a bbox."""
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]


def _bbox_mask(bbox: Any, shape: tuple[int, int]) -> np.ndarray:
    """Create a binary fallback mask from a bbox."""
    height, width = shape
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)] = 1
    return mask


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


def _context_adapter_kwargs(context: dict[str, Any] | None) -> dict[str, Any]:
    """Read optional adapter runtime kwargs from context."""
    return dict((context or {}).get("adapter_kwargs", {}))
