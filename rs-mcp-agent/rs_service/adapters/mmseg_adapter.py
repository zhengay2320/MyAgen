from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from rs_service.adapters.base import AdapterMetadata, BaseAdapter, ModelBackendUnavailable, SegmentationPrediction
from rs_service.adapters.ultralytics_adapter import _to_hwc_uint8


INSTALL_MMSEG_MESSAGE = (
    "MMSegmentation backend is unavailable. Install OpenMMLab dependencies in a separate "
    "environment, for example: pip install -U openmim && mim install mmengine mmcv "
    "&& pip install mmsegmentation."
)


class MMSegmentationAdapter(BaseAdapter):
    """MMSegmentation semantic segmentation adapter with lazy OpenMMLab imports."""

    def __init__(self, model_config: dict[str, Any]) -> None:
        """Create an MMSeg adapter from a models.yaml style config dict."""
        self.config = dict(model_config)
        self.config_path = str(self.config.get("config") or self.config.get("config_path") or "")
        self.checkpoint = str(self.config.get("checkpoint") or self.config.get("weights") or self.config.get("weight") or "")
        self.device = str(self.config.get("device", "cpu"))
        self.classes = self.config.get("classes")
        self.palette = self.config.get("palette")
        self.model: Any | None = None
        self._init_model: Any | None = None
        self._inference_model: Any | None = None
        self.metadata = AdapterMetadata(
            id=str(self.config.get("id", "mmseg_segmentation")),
            task="semantic_segmentation",
            backend="mmseg",
            framework="mmsegmentation",
            weights=self.checkpoint,
            description="MMSegmentation config/checkpoint semantic segmentation adapter.",
        )

    def load(self) -> None:
        """Load MMSegmentation config/checkpoint lazily."""
        if self.model is not None:
            return
        if importlib.util.find_spec("mmseg") is None:
            raise ModelBackendUnavailable(INSTALL_MMSEG_MESSAGE)
        if not self.config_path:
            raise FileNotFoundError("MMSegmentation config path is not configured for this model_id.")
        if not self.checkpoint:
            raise FileNotFoundError("MMSegmentation checkpoint path is not configured for this model_id.")
        config_path = Path(self.config_path)
        checkpoint_path = Path(self.checkpoint)
        if not config_path.exists():
            raise FileNotFoundError(
                f"MMSegmentation config not found: {config_path}. Update configs/models.yaml."
            )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"MMSegmentation checkpoint not found: {checkpoint_path}. "
                "Place the .pth file there or update configs/models.yaml."
            )
        try:
            from mmseg.apis import inference_model, init_model
        except Exception as exc:  # pragma: no cover - depends on optional OpenMMLab packages
            raise ModelBackendUnavailable(f"{INSTALL_MMSEG_MESSAGE} Original error: {exc}") from exc
        self._init_model = init_model
        self._inference_model = inference_model
        self.model = init_model(str(config_path), str(checkpoint_path), device=self.device)
        if self.classes is not None:
            setattr(self.model, "CLASSES", self.classes)
        if self.palette is not None:
            setattr(self.model, "PALETTE", self.palette)

    def predict_tile(
        self,
        tile_array: np.ndarray,
        tile_info: dict[str, Any],
        **kwargs: Any,
    ) -> SegmentationPrediction:
        """Predict one tile and return a hard mask plus optional probabilities."""
        self.load()
        assert self._inference_model is not None
        assert self.model is not None
        image = _to_hwc_uint8(tile_array)
        try:
            result = self._inference_model(self.model, image)
        except Exception as exc:
            tile_id = tile_info.get("tile_id", "unknown")
            raise RuntimeError(f"MMSegmentation tile inference failed for tile_id={tile_id}: {exc}") from exc
        mask = _extract_mask(result)
        probabilities = _extract_probabilities(result)
        class_names = _class_names(self.classes, getattr(self.model, "CLASSES", None), mask=mask)
        metadata = {
            "tile_id": tile_info.get("tile_id"),
            "backend": "mmseg",
            "confidence_available": probabilities is not None,
        }
        return SegmentationPrediction(
            mask=mask.astype(np.uint8, copy=False),
            probabilities=probabilities,
            class_names=class_names,
            metadata=metadata,
        )

    def predict_proba(self, tile: np.ndarray, context: dict[str, Any] | None = None) -> np.ndarray:
        """Backward-compatible probability API for existing segmentation pipelines."""
        prediction = self.predict_tile(tile, _context_tile_info(context))
        if prediction.probabilities is not None:
            return prediction.probabilities
        class_count = int(max(prediction.class_names.keys(), default=int(prediction.mask.max())) + 1)
        return _one_hot(prediction.mask, max(class_count, int(prediction.mask.max()) + 1))

    def close(self) -> None:
        """Release the loaded model reference."""
        self.model = None
        self._init_model = None
        self._inference_model = None


def _extract_mask(result: Any) -> np.ndarray:
    """Extract a HxW class-id mask across MMSeg 0.x/1.x result styles."""
    pred = getattr(result, "pred_sem_seg", None)
    if pred is not None:
        data = getattr(pred, "data", pred)
        array = _as_numpy(data)
        if array.ndim == 3:
            array = array[0]
        return array.astype(np.uint8)
    if isinstance(result, (list, tuple)) and result:
        return _extract_mask(result[0])
    if isinstance(result, dict):
        for key in ("pred_sem_seg", "seg_pred", "semantic_seg", "mask"):
            if key in result:
                return _extract_mask(result[key])
    array = _as_numpy(result)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Could not extract MMSeg mask from result with shape {array.shape}.")
    return array.astype(np.uint8)


def _extract_probabilities(result: Any) -> np.ndarray | None:
    """Extract optional CxHxW probability/confidence map when available."""
    seg_logits = getattr(result, "seg_logits", None)
    if seg_logits is not None:
        data = getattr(seg_logits, "data", seg_logits)
        logits = _as_numpy(data).astype(np.float32)
        if logits.ndim == 3:
            logits -= np.max(logits, axis=0, keepdims=True)
            exp = np.exp(logits)
            return exp / np.maximum(exp.sum(axis=0, keepdims=True), 1e-6)
    if isinstance(result, dict):
        for key in ("probabilities", "probs", "seg_logits"):
            if key in result:
                value = result[key]
                if key == "seg_logits":
                    return _extract_probabilities(type("Result", (), {"seg_logits": value})())
                array = _as_numpy(value).astype(np.float32)
                return array if array.ndim == 3 else None
    return None


def _as_numpy(value: Any) -> np.ndarray:
    """Convert tensors, lists, and arrays to numpy arrays."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _class_names(*candidates: Any, mask: np.ndarray) -> dict[int, str]:
    """Build class-name mapping from configured/model classes or mask ids."""
    for classes in candidates:
        if isinstance(classes, dict):
            return {int(key): str(value) for key, value in classes.items()}
        if isinstance(classes, (list, tuple)):
            return {index: str(value) for index, value in enumerate(classes)}
    return {int(value): f"class_{int(value)}" for value in np.unique(mask)}


def _one_hot(mask: np.ndarray, class_count: int) -> np.ndarray:
    """Convert a hard mask to CxHxW float one-hot probabilities."""
    output = np.zeros((class_count, mask.shape[0], mask.shape[1]), dtype=np.float32)
    for class_id in range(class_count):
        output[class_id] = (mask == class_id).astype(np.float32)
    return output


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
