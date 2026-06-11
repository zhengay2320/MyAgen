from __future__ import annotations

from typing import Any

import numpy as np

from rs_service.adapters.base import AdapterMetadata, BaseAdapter, ModelBackendUnavailable, SuperResolutionPrediction


INSTALL_MMAGIC_MESSAGE = (
    "MMagic super-resolution backend is reserved but not implemented in this MVP. "
    "Install mmagic/mmengine/mmcv and add an adapter implementation before using this model_id."
)


class MMagicSuperResolutionAdapter(BaseAdapter):
    """MMagic super-resolution placeholder adapter with a clear MVP error."""

    def __init__(self, model_config: dict[str, Any]) -> None:
        self.config = dict(model_config)
        self.scale = int(self.config.get("scale", 4))
        self.metadata = AdapterMetadata(
            id=str(self.config.get("id", "mmagic_sr_stub")),
            task="super_resolution",
            backend="mmagic",
            framework="mmagic",
            weights=str(self.config.get("checkpoint") or self.config.get("weights") or ""),
            description="MMagic SR reserved adapter stub.",
        )

    def load(self) -> None:
        """Always raise a clear reserved-backend message."""
        raise ModelBackendUnavailable(INSTALL_MMAGIC_MESSAGE)

    def predict_tile(self, tile_array: np.ndarray, tile_info: dict[str, Any], **kwargs: Any) -> SuperResolutionPrediction:
        """Raise the reserved-backend error."""
        self.load()
        return SuperResolutionPrediction(image=tile_array, scale=self.scale)
