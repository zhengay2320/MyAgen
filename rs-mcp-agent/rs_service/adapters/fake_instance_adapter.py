from __future__ import annotations

from typing import Any

import numpy as np

from rs_service.adapters.base import AdapterMetadata, BaseAdapter, InstancePrediction
from rs_service.adapters.fake_detection_adapter import _component_boxes


class FakeInstanceSegmentationAdapter(BaseAdapter):
    """Fake instance segmenter that creates masks for bright components."""

    metadata = AdapterMetadata(
        id="fake_instance",
        task="instance_segmentation",
        backend="fake",
        framework="numpy",
        description="Connected-component fake instance segmenter for smoke tests.",
    )

    def predict_tile(self, tile_array: np.ndarray, tile_info: dict[str, Any], **kwargs: Any) -> list[InstancePrediction]:
        """Predict instance masks for bright tile regions."""
        gray = np.mean(tile_array.astype(np.float32), axis=0)
        mask = gray > max(150.0, float(gray.mean() + gray.std()))
        boxes = _component_boxes(mask, min_pixels=int(kwargs.get("min_pixels", 12)))
        predictions: list[InstancePrediction] = []
        for index, bbox in enumerate(boxes[: int(kwargs.get("max_instances", 8))]):
            x1, y1, x2, y2 = bbox
            instance_mask = np.zeros_like(mask, dtype=np.uint8)
            instance_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2].astype(np.uint8)
            polygon = [(float(x1), float(y1)), (float(x2), float(y1)), (float(x2), float(y2)), (float(x1), float(y2)), (float(x1), float(y1))]
            score = float(np.clip(gray[y1:y2, x1:x2].mean() / 255.0, 0.05, 0.99))
            predictions.append(
                InstancePrediction(
                    label="bright_instance",
                    score=round(score, 4),
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    mask=instance_mask,
                    class_id=0,
                    polygon=polygon,
                    attributes={"fake_index": index, "tile_id": tile_info.get("tile_id")},
                )
            )
        return predictions

    def predict(self, tile: np.ndarray, context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Backward-compatible dict prediction API."""
        return [
            {
                "label": item.label,
                "score": item.score,
                "bbox": item.bbox,
                "mask_polygon": item.polygon,
                "class_id": item.class_id,
            }
            for item in self.predict_tile(tile, {})
        ]
