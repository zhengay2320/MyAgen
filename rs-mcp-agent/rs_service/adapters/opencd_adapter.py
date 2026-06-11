from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from rs_service.adapters.base import AdapterMetadata, BaseAdapter, ChangePrediction, ModelBackendUnavailable
from rs_service.adapters.ultralytics_adapter import _to_hwc_uint8


INSTALL_OPENCD_MESSAGE = (
    "Open-CD backend is unavailable. Install Open-CD and compatible OpenMMLab dependencies "
    "in a separate environment, for example: pip install -U openmim && mim install mmengine mmcv "
    "&& install Open-CD from its official repository."
)


class OpenCDAdapter(BaseAdapter):
    """Open-CD bi-temporal change detection adapter with lazy imports."""

    def __init__(self, model_config: dict[str, Any]) -> None:
        """Create an Open-CD adapter from config/checkpoint settings."""
        self.config = dict(model_config)
        self.config_path = str(self.config.get("config") or self.config.get("config_path") or "")
        self.checkpoint = str(self.config.get("checkpoint") or self.config.get("weights") or self.config.get("weight") or "")
        self.device = str(self.config.get("device", "cpu"))
        self.threshold = float(self.config.get("threshold", 0.5))
        self.model: Any | None = None
        self._inference_model: Any | None = None
        self.metadata = AdapterMetadata(
            id=str(self.config.get("id", "opencd_change_detection")),
            task="change_detection",
            backend="opencd",
            framework="open-cd",
            weights=self.checkpoint,
            description="Open-CD bi-temporal change detection adapter.",
        )

    def load(self) -> None:
        """Load an Open-CD model lazily."""
        if self.model is not None:
            return
        if importlib.util.find_spec("opencd") is None:
            raise ModelBackendUnavailable(INSTALL_OPENCD_MESSAGE)
        if not self.config_path:
            raise FileNotFoundError("Open-CD config path is not configured for this model_id.")
        if not self.checkpoint:
            raise FileNotFoundError("Open-CD checkpoint path is not configured for this model_id.")
        config_path = Path(self.config_path)
        checkpoint_path = Path(self.checkpoint)
        if not config_path.exists():
            raise FileNotFoundError(f"Open-CD config not found: {config_path}. Update configs/models.yaml.")
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Open-CD checkpoint not found: {checkpoint_path}. Place the .pth file there or update configs/models.yaml."
            )
        try:
            from opencd.apis import inference_model, init_model
        except Exception as exc:  # pragma: no cover - optional dependency internals
            raise ModelBackendUnavailable(f"{INSTALL_OPENCD_MESSAGE} Original error: {exc}") from exc
        self._inference_model = inference_model
        self.model = init_model(str(config_path), str(checkpoint_path), device=self.device)

    def predict_tile(
        self,
        tile_t1: np.ndarray,
        tile_info: dict[str, Any],
        **kwargs: Any,
    ) -> ChangePrediction:
        """Predict change for one tile pair."""
        tile_t2 = kwargs.get("tile_t2", kwargs.get("after_tile"))
        if tile_t2 is None:
            raise ValueError("tile_t2 or after_tile is required for Open-CD change detection.")
        self.load()
        assert self.model is not None
        assert self._inference_model is not None
        image_t1 = _to_hwc_uint8(tile_t1)
        image_t2 = _to_hwc_uint8(tile_t2)
        try:
            result = self._run_inference(image_t1, image_t2)
        except Exception as exc:
            tile_id = tile_info.get("tile_id", "unknown")
            raise RuntimeError(f"Open-CD tile inference failed for tile_id={tile_id}: {exc}") from exc
        probability = _extract_probability(result)
        mask = _extract_mask(result, probability, float(kwargs.get("threshold", self.threshold)))
        return ChangePrediction(mask=mask.astype(np.uint8), probability=probability.astype(np.float32), threshold=float(kwargs.get("threshold", self.threshold)))

    def predict_proba(self, tile_a: np.ndarray, tile_b: np.ndarray, context: dict[str, Any] | None = None) -> np.ndarray:
        """Backward-compatible probability API used by existing pipelines."""
        prediction = self.predict_tile(tile_a, _context_tile_info(context), tile_t2=tile_b)
        return prediction.probability

    def close(self) -> None:
        """Release model references."""
        self.model = None
        self._inference_model = None

    def _run_inference(self, image_t1: np.ndarray, image_t2: np.ndarray) -> Any:
        """Call Open-CD inference across common API variants."""
        assert self._inference_model is not None
        try:
            return self._inference_model(self.model, image_t1, image_t2)
        except TypeError as exc:
            first_signature_error = exc
        try:
            return self._inference_model(self.model, [image_t1, image_t2])
        except TypeError as exc:
            second_signature_error = exc
        stacked = np.concatenate([image_t1, image_t2], axis=-1)
        return self._inference_model(self.model, stacked)


def _extract_probability(result: Any) -> np.ndarray:
    """Extract HxW change probability from Open-CD result styles."""
    seg_logits = getattr(result, "seg_logits", None)
    if seg_logits is not None:
        data = getattr(seg_logits, "data", seg_logits)
        logits = _as_numpy(data).astype(np.float32)
        if logits.ndim == 3 and logits.shape[0] >= 2:
            logits -= np.max(logits, axis=0, keepdims=True)
            exp = np.exp(logits)
            proba = exp / np.maximum(exp.sum(axis=0, keepdims=True), 1e-6)
            return proba[1]
        if logits.ndim == 2:
            return 1.0 / (1.0 + np.exp(-logits))
    for attr in ("probability", "probabilities", "change_probability"):
        value = getattr(result, attr, None)
        if value is not None:
            array = _as_numpy(value).astype(np.float32)
            return array[1] if array.ndim == 3 and array.shape[0] > 1 else np.squeeze(array)
    if isinstance(result, dict):
        for key in ("probability", "probabilities", "change_probability", "seg_logits"):
            if key in result:
                return _extract_probability(type("Result", (), {key: result[key]})())
    mask = _extract_mask(result, None, 0.5)
    return mask.astype(np.float32)


def _extract_mask(result: Any, probability: np.ndarray | None, threshold: float) -> np.ndarray:
    """Extract HxW change mask, falling back to thresholded probabilities."""
    pred = getattr(result, "pred_sem_seg", None)
    if pred is not None:
        data = getattr(pred, "data", pred)
        array = _as_numpy(data)
        if array.ndim == 3:
            array = array[0]
        return (array > 0).astype(np.uint8)
    if isinstance(result, dict):
        for key in ("pred_sem_seg", "mask", "change_mask"):
            if key in result:
                return _extract_mask(type("Result", (), {"pred_sem_seg": result[key]})(), probability, threshold)
    if probability is not None:
        return (probability >= threshold).astype(np.uint8)
    array = _as_numpy(result)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 2:
        return (array > 0).astype(np.uint8)
    raise ValueError(f"Could not extract Open-CD mask from result with shape {array.shape}.")


def _as_numpy(value: Any) -> np.ndarray:
    """Convert tensors, lists, and arrays to numpy arrays."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


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
