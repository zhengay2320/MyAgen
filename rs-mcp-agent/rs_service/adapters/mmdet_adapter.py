from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from rs_service.adapters.base import AdapterMetadata, BaseAdapter, InstancePrediction, ModelBackendUnavailable
from rs_service.adapters.ultralytics_adapter import _to_hwc_uint8


INSTALL_MMDET_MESSAGE = (
    "MMDetection backend is unavailable. Install OpenMMLab dependencies in a separate "
    "environment, for example: pip install -U openmim && mim install mmengine mmcv "
    "&& pip install mmdet."
)


class MMDetectionInstanceAdapter(BaseAdapter):
    """MMDetection instance segmentation adapter with lazy OpenMMLab imports."""

    def __init__(self, model_config: dict[str, Any]) -> None:
        """Create an MMDetection adapter from a config/checkpoint model config."""
        self.config = dict(model_config)
        self.config_path = str(self.config.get("config") or self.config.get("config_path") or "")
        self.checkpoint = str(self.config.get("checkpoint") or self.config.get("weights") or self.config.get("weight") or "")
        self.device = str(self.config.get("device", "cpu"))
        self.classes = self.config.get("classes")
        self.model: Any | None = None
        self._inference_detector: Any | None = None
        self.metadata = AdapterMetadata(
            id=str(self.config.get("id", "mmdet_instance")),
            task="instance_segmentation",
            backend="mmdet",
            framework="mmdetection",
            weights=self.checkpoint,
            description="MMDetection instance segmentation adapter.",
        )

    def load(self) -> None:
        """Load an MMDetection detector lazily."""
        if self.model is not None:
            return
        if importlib.util.find_spec("mmdet") is None:
            raise ModelBackendUnavailable(INSTALL_MMDET_MESSAGE)
        if not self.config_path:
            raise FileNotFoundError("MMDetection config path is not configured for this model_id.")
        if not self.checkpoint:
            raise FileNotFoundError("MMDetection checkpoint path is not configured for this model_id.")
        config_path = Path(self.config_path)
        checkpoint_path = Path(self.checkpoint)
        if not config_path.exists():
            raise FileNotFoundError(f"MMDetection config not found: {config_path}. Update configs/models.yaml.")
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"MMDetection checkpoint not found: {checkpoint_path}. Place the .pth file there or update configs/models.yaml."
            )
        try:
            from mmdet.apis import inference_detector, init_detector
        except Exception as exc:  # pragma: no cover - depends on optional packages
            raise ModelBackendUnavailable(f"{INSTALL_MMDET_MESSAGE} Original error: {exc}") from exc
        self._inference_detector = inference_detector
        self.model = init_detector(str(config_path), str(checkpoint_path), device=self.device)

    def predict_tile(
        self,
        tile_array: np.ndarray,
        tile_info: dict[str, Any],
        **kwargs: Any,
    ) -> list[InstancePrediction]:
        """Predict tile-local instances as masks, boxes, classes, and scores."""
        self.load()
        assert self.model is not None
        assert self._inference_detector is not None
        image = _to_hwc_uint8(tile_array)
        try:
            result = self._inference_detector(self.model, image)
        except Exception as exc:
            tile_id = tile_info.get("tile_id", "unknown")
            raise RuntimeError(f"MMDetection tile inference failed for tile_id={tile_id}: {exc}") from exc
        return _parse_instances(result, image.shape[:2], _class_names(self.classes, self.model), tile_info)

    def predict(self, tile: np.ndarray, context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Backward-compatible dict prediction API used by existing pipelines."""
        kwargs = dict((context or {}).get("adapter_kwargs", {}))
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

    def close(self) -> None:
        """Release model references."""
        self.model = None
        self._inference_detector = None


def _parse_instances(
    result: Any,
    image_shape: tuple[int, int],
    names: dict[int, str],
    tile_info: dict[str, Any],
) -> list[InstancePrediction]:
    """Parse MMDetection 2.x/3.x result styles into InstancePrediction objects."""
    pred_instances = getattr(result, "pred_instances", None)
    if pred_instances is not None:
        bboxes = _as_numpy(getattr(pred_instances, "bboxes", []))
        scores = _as_numpy(getattr(pred_instances, "scores", []))
        labels = _as_numpy(getattr(pred_instances, "labels", []))
        masks = _extract_masks(getattr(pred_instances, "masks", None), image_shape)
        return _build_predictions(bboxes, scores, labels, masks, names, tile_info)
    if isinstance(result, dict) and "pred_instances" in result:
        return _parse_instances(result["pred_instances"], image_shape, names, tile_info)
    if isinstance(result, tuple) and result:
        bbox_result = result[0]
        mask_result = result[1] if len(result) > 1 else None
        return _parse_legacy_results(bbox_result, mask_result, image_shape, names, tile_info)
    if isinstance(result, list):
        return _parse_legacy_results(result, None, image_shape, names, tile_info)
    if all(hasattr(result, name) for name in ["bboxes", "scores", "labels"]):
        masks = _extract_masks(getattr(result, "masks", None), image_shape)
        return _build_predictions(_as_numpy(result.bboxes), _as_numpy(result.scores), _as_numpy(result.labels), masks, names, tile_info)
    return []


def _parse_legacy_results(
    bbox_result: Any,
    mask_result: Any,
    image_shape: tuple[int, int],
    names: dict[int, str],
    tile_info: dict[str, Any],
) -> list[InstancePrediction]:
    """Parse MMDetection 2.x list-per-class outputs."""
    bboxes_list: list[np.ndarray] = []
    scores_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for class_id, class_boxes in enumerate(bbox_result or []):
        boxes = _as_numpy(class_boxes)
        if boxes.size == 0:
            continue
        boxes = boxes.reshape((-1, boxes.shape[-1]))
        bboxes_list.append(boxes[:, :4])
        scores_list.append(boxes[:, 4] if boxes.shape[1] > 4 else np.ones((boxes.shape[0],), dtype=np.float32))
        labels_list.append(np.full((boxes.shape[0],), class_id, dtype=np.int64))
        class_masks = mask_result[class_id] if mask_result is not None and class_id < len(mask_result) else []
        masks.extend(_extract_masks(class_masks, image_shape))
    if not bboxes_list:
        return []
    return _build_predictions(
        np.concatenate(bboxes_list, axis=0),
        np.concatenate(scores_list, axis=0),
        np.concatenate(labels_list, axis=0),
        masks,
        names,
        tile_info,
    )


def _build_predictions(
    bboxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    masks: list[np.ndarray],
    names: dict[int, str],
    tile_info: dict[str, Any],
) -> list[InstancePrediction]:
    """Build unified instance predictions."""
    predictions: list[InstancePrediction] = []
    for index, bbox in enumerate(np.asarray(bboxes).reshape((-1, 4))):
        class_id = int(labels[index]) if index < len(labels) else 0
        score = float(scores[index]) if index < len(scores) else 0.0
        mask = masks[index] if index < len(masks) else _bbox_mask(bbox, (int(max(bbox[3], 1)), int(max(bbox[2], 1))))
        polygon = _mask_or_bbox_polygon(mask, bbox)
        predictions.append(
            InstancePrediction(
                label=names.get(class_id, f"class_{class_id}"),
                score=score,
                bbox=[float(v) for v in bbox[:4]],
                mask=mask.astype(np.uint8),
                class_id=class_id,
                polygon=polygon,
                attributes={"backend": "mmdet", "tile_id": tile_info.get("tile_id")},
            )
        )
    return predictions


def _extract_masks(value: Any, image_shape: tuple[int, int]) -> list[np.ndarray]:
    """Extract a list of binary masks from common MMDetection containers."""
    if value is None:
        return []
    if hasattr(value, "to_ndarray"):
        value = value.to_ndarray()
    array = _as_numpy(value)
    if array.dtype == object:
        return [_as_numpy(item).astype(np.uint8) for item in value]
    if array.ndim == 2:
        return [array.astype(np.uint8)]
    if array.ndim == 3:
        return [(item > 0.5).astype(np.uint8) for item in array]
    if isinstance(value, (list, tuple)):
        masks: list[np.ndarray] = []
        for item in value:
            arr = _as_numpy(item)
            if arr.ndim == 2:
                masks.append(arr.astype(np.uint8))
        return masks
    return []


def _mask_or_bbox_polygon(mask: np.ndarray, bbox: np.ndarray) -> list[tuple[float, float]]:
    """Approximate an instance polygon using mask bounds or bbox."""
    ys, xs = np.where(mask > 0)
    if xs.size:
        return _bbox_polygon([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])
    return _bbox_polygon(bbox)


def _bbox_polygon(bbox: Any) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]


def _bbox_mask(bbox: Any, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)] = 1
    return mask


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
