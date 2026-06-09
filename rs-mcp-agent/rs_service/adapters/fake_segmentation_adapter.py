from __future__ import annotations

from typing import Any

import numpy as np

from rs_service.adapters.base import AdapterMetadata, BaseAdapter, SegmentationPrediction


class FakeSemanticSegmentationAdapter(BaseAdapter):
    """Fake semantic segmenter that thresholds color and brightness cues."""

    metadata = AdapterMetadata(
        id="fake_segmentation",
        task="semantic_segmentation",
        backend="fake",
        framework="numpy",
        description="Color-threshold fake semantic segmenter for smoke tests.",
    )

    def predict_tile(self, tile_array: np.ndarray, tile_info: dict[str, Any], **kwargs: Any) -> SegmentationPrediction:
        """Return a three-class mask and class probabilities for one tile."""
        data = tile_array.astype(np.float32)
        red = data[0]
        green = data[min(1, data.shape[0] - 1)]
        blue = data[min(2, data.shape[0] - 1)]
        brightness = np.mean(data, axis=0)
        mask = np.zeros(brightness.shape, dtype=np.uint8)
        mask[brightness > 185] = 1
        mask[(blue > green) & (blue > red) & (blue > 120)] = 2
        class_count = 3
        probabilities = np.zeros((class_count, *mask.shape), dtype=np.float32)
        for class_id in range(class_count):
            probabilities[class_id] = (mask == class_id).astype(np.float32)
        probabilities = probabilities * 0.85 + 0.05
        probabilities /= np.maximum(probabilities.sum(axis=0, keepdims=True), 1e-6)
        return SegmentationPrediction(mask=mask, probabilities=probabilities, class_names={0: "background", 1: "bright_object", 2: "blue_object"})

    def predict_proba(self, tile: np.ndarray, context: dict[str, Any] | None = None) -> np.ndarray:
        """Backward-compatible probability API."""
        return self.predict_tile(tile, {}).probabilities  # type: ignore[return-value]
