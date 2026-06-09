from __future__ import annotations

from typing import Any

import numpy as np

from rs_service.adapters.base import AdapterMetadata, BaseAdapter, ChangePrediction


class FakeChangeDetectionAdapter(BaseAdapter):
    """Fake bi-temporal change detector based on absolute pixel differences."""

    metadata = AdapterMetadata(
        id="fake_change",
        task="change_detection",
        backend="fake",
        framework="numpy",
        description="Difference-threshold fake change detector for smoke tests.",
    )

    def predict_tile(self, tile_array: np.ndarray, tile_info: dict[str, Any], **kwargs: Any) -> ChangePrediction:
        """Predict change mask from `tile_array` and `after_tile`."""
        after_tile = kwargs.get("after_tile")
        if after_tile is None:
            raise ValueError("after_tile is required for fake change detection")
        threshold = float(kwargs.get("threshold", 0.5))
        diff = np.mean(np.abs(after_tile.astype(np.float32) - tile_array.astype(np.float32)), axis=0)
        probability = np.clip(diff / 128.0, 0.0, 1.0).astype(np.float32)
        mask = (probability >= threshold).astype(np.uint8)
        return ChangePrediction(mask=mask, probability=probability, threshold=threshold)

    def predict_proba(self, tile_a: np.ndarray, tile_b: np.ndarray, context: dict[str, Any] | None = None) -> np.ndarray:
        """Backward-compatible probability API."""
        return self.predict_tile(tile_a, {}, after_tile=tile_b, threshold=0.5).probability
