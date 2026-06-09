from __future__ import annotations

from typing import Any

import numpy as np

from rs_service.adapters.base import AdapterMetadata, BaseAdapter, SuperResolutionPrediction


class FakeSuperResolutionAdapter(BaseAdapter):
    """Fake super-resolution adapter using nearest-neighbor numpy resize."""

    metadata = AdapterMetadata(
        id="fake_super_resolution",
        task="super_resolution",
        backend="fake",
        framework="numpy",
        description="Numpy nearest-neighbor fake super-resolution adapter.",
    )

    def __init__(self, scale: int = 2) -> None:
        if scale not in {2, 4}:
            raise ValueError("fake super resolution supports scale 2 or 4")
        self.scale = scale

    def predict_tile(self, tile_array: np.ndarray, tile_info: dict[str, Any], **kwargs: Any) -> SuperResolutionPrediction:
        """Upscale one tile by nearest-neighbor repetition."""
        scale = int(kwargs.get("scale", self.scale))
        image = np.repeat(np.repeat(tile_array, scale, axis=1), scale, axis=2)
        return SuperResolutionPrediction(image=image.astype(tile_array.dtype, copy=False), scale=scale)

    def upscale(self, tile: np.ndarray, context: dict[str, Any] | None = None) -> np.ndarray:
        """Backward-compatible upscale API."""
        return self.predict_tile(tile, {}, scale=self.scale).image
