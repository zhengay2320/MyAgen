from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from rs_service.adapters.base import AdapterMetadata, BaseAdapter, DetectionPrediction, ModelBackendUnavailable
from rs_service.adapters.ultralytics_adapter import INSTALL_DETECTION_MESSAGE, _to_hwc_uint8


class SahiDetectionAdapter(BaseAdapter):
    """SAHI sliced inference adapter for YOLO-style object detection."""

    def __init__(self, model_config: dict[str, Any]) -> None:
        """Create a SAHI adapter from a models.yaml style config dict."""
        self.config = dict(model_config)
        self.weight = str(self.config.get("weight") or self.config.get("weights") or "")
        self.confidence_threshold = float(self.config.get("confidence_threshold", 0.25))
        self.device = str(self.config.get("device", "cpu"))
        self.slice_height = int(self.config.get("slice_height", self.config.get("tile_size", 512)))
        self.slice_width = int(self.config.get("slice_width", self.config.get("tile_size", 512)))
        self.overlap_ratio = float(self.config.get("overlap_ratio", 0.2))
        self.model: Any | None = None
        self.metadata = AdapterMetadata(
            id=str(self.config.get("id", "sahi_yolo_detection")),
            task="object_detection",
            backend="sahi",
            framework="sahi+ultralytics",
            weights=self.weight,
            description="SAHI sliced detection adapter using an Ultralytics YOLO backend.",
        )

    def load(self) -> None:
        """Load the SAHI detection model lazily."""
        if self.model is not None:
            return
        if importlib.util.find_spec("sahi") is None or importlib.util.find_spec("ultralytics") is None:
            raise ModelBackendUnavailable(
                f"SAHI detection backend is unavailable. {INSTALL_DETECTION_MESSAGE}"
            )
        if not self.weight:
            raise FileNotFoundError("SAHI YOLO model weight is not configured for this model_id.")
        weight_path = Path(self.weight)
        if not weight_path.exists():
            raise FileNotFoundError(
                f"SAHI YOLO model weight not found: {weight_path}. "
                "Place the .pt file there or update configs/models.yaml."
            )
        try:
            from sahi.models.ultralytics import UltralyticsDetectionModel
        except Exception as exc:  # pragma: no cover - depends on optional package internals
            raise ModelBackendUnavailable(f"SAHI could not import its Ultralytics backend. {INSTALL_DETECTION_MESSAGE}") from exc
        self.model = UltralyticsDetectionModel(
            model_path=str(weight_path),
            confidence_threshold=self.confidence_threshold,
            device=self.device,
        )

    def predict_tile(
        self,
        tile_array: np.ndarray,
        tile_info: dict[str, Any],
        **kwargs: Any,
    ) -> list[DetectionPrediction]:
        """Run SAHI sliced prediction on one pipeline tile."""
        self.load()
        assert self.model is not None
        try:
            from sahi.predict import get_sliced_prediction
        except Exception as exc:  # pragma: no cover - depends on optional package internals
            raise ModelBackendUnavailable(f"SAHI prediction API is unavailable. {INSTALL_DETECTION_MESSAGE}") from exc
        image = _to_hwc_uint8(tile_array)
        slice_height = int(kwargs.get("slice_height", self.slice_height))
        slice_width = int(kwargs.get("slice_width", self.slice_width))
        overlap_ratio = float(kwargs.get("overlap_ratio", self.overlap_ratio))
        result = get_sliced_prediction(
            image,
            self.model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            verbose=0,
        )
        predictions: list[DetectionPrediction] = []
        for item in getattr(result, "object_prediction_list", []):
            bbox_obj = getattr(item, "bbox", None)
            if bbox_obj is None:
                continue
            bbox = [
                float(getattr(bbox_obj, "minx", 0.0)),
                float(getattr(bbox_obj, "miny", 0.0)),
                float(getattr(bbox_obj, "maxx", 0.0)),
                float(getattr(bbox_obj, "maxy", 0.0)),
            ]
            category = getattr(item, "category", None)
            class_id = int(getattr(category, "id", 0) or 0)
            label = str(getattr(category, "name", f"class_{class_id}") or f"class_{class_id}")
            score_obj = getattr(item, "score", None)
            score = float(getattr(score_obj, "value", score_obj if score_obj is not None else 0.0))
            predictions.append(
                DetectionPrediction(
                    label=label,
                    score=score,
                    bbox=bbox,
                    class_id=class_id,
                    attributes={"backend": "sahi", "tile_id": tile_info.get("tile_id")},
                )
            )
        return predictions

    def predict(self, tile: np.ndarray, context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Backward-compatible dict prediction API used by existing pipelines."""
        tile_info = _context_tile_info(context)
        kwargs = dict((context or {}).get("adapter_kwargs", {}))
        return [prediction.to_dict() for prediction in self.predict_tile(tile, tile_info, **kwargs)]

    def close(self) -> None:
        """Release the loaded model reference."""
        self.model = None


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
