from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from rs_service.adapters.base import AdapterMetadata, BaseAdapter, ModelBackendUnavailable, SuperResolutionPrediction
from rs_service.adapters.swinir_adapter import _match_input_bands
from rs_service.adapters.ultralytics_adapter import _to_hwc_uint8


INSTALL_BASICSR_MESSAGE = (
    "BasicSR backend is unavailable. Install basicsr and torch in a separate environment, "
    "then provide a BasicSR config and checkpoint."
)


class BasicSRAdapter(BaseAdapter):
    """BasicSR super-resolution adapter with lazy imports."""

    def __init__(self, model_config: dict[str, Any]) -> None:
        """Create a BasicSR adapter from config/checkpoint/scale settings."""
        self.config = dict(model_config)
        self.config_path = str(self.config.get("config") or self.config.get("config_path") or "")
        self.checkpoint = str(self.config.get("checkpoint") or self.config.get("weights") or self.config.get("weight") or "")
        self.scale = int(self.config.get("scale", 4))
        self.device = str(self.config.get("device", "cpu"))
        self.model: Any | None = None
        self.torch: Any | None = None
        self.metadata = AdapterMetadata(
            id=str(self.config.get("id", f"basicsr_x{self.scale}")),
            task="super_resolution",
            backend="basicsr",
            framework="basicsr",
            weights=self.checkpoint,
            description="BasicSR super-resolution adapter.",
        )

    def load(self) -> None:
        """Load BasicSR model lazily."""
        if self.model is not None:
            return
        if importlib.util.find_spec("basicsr") is None or importlib.util.find_spec("torch") is None:
            raise ModelBackendUnavailable(INSTALL_BASICSR_MESSAGE)
        if not self.config_path:
            raise FileNotFoundError("BasicSR config path is not configured for this model_id.")
        if not self.checkpoint:
            raise FileNotFoundError("BasicSR checkpoint path is not configured for this model_id.")
        config_path = Path(self.config_path)
        checkpoint_path = Path(self.checkpoint)
        if not config_path.exists():
            raise FileNotFoundError(f"BasicSR config not found: {config_path}. Update configs/models.yaml.")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"BasicSR checkpoint not found: {checkpoint_path}. Place the .pth file there or update configs/models.yaml.")
        try:
            import torch
            from basicsr.archs import build_network
        except Exception as exc:  # pragma: no cover - optional dependency internals
            raise ModelBackendUnavailable(f"{INSTALL_BASICSR_MESSAGE} Original error: {exc}") from exc
        options = _load_basic_config(config_path)
        network_options = options.get("network_g") or options.get("model") or {}
        if not network_options:
            raise ModelBackendUnavailable("BasicSR config does not contain network_g/model settings needed to build the network.")
        self.torch = torch
        model = build_network(network_options)
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        state = checkpoint.get("params_ema") or checkpoint.get("params") or checkpoint.get("state_dict") if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state, strict=False)
        self.model = model.to(self.device).eval()

    def predict_tile(self, tile_array: np.ndarray, tile_info: dict[str, Any], **kwargs: Any) -> SuperResolutionPrediction:
        """Upscale one tile with BasicSR."""
        scale = int(kwargs.get("scale", self.scale))
        self.load()
        assert self.torch is not None
        assert self.model is not None
        image = _to_hwc_uint8(tile_array)
        tensor = self.torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            output = self.model(tensor)
        if isinstance(output, (list, tuple)):
            output = output[0]
        output = output.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        sr = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        return SuperResolutionPrediction(image=_match_input_bands(sr, tile_array), scale=scale)

    def upscale(self, tile: np.ndarray, context: dict[str, Any] | None = None) -> np.ndarray:
        """Backward-compatible upscale API."""
        return self.predict_tile(tile, {}, scale=self.scale).image

    def close(self) -> None:
        """Release model references."""
        self.model = None
        self.torch = None


def _load_basic_config(path: Path) -> dict[str, Any]:
    """Load a BasicSR config using YAML when available."""
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - PyYAML is a base dependency in normal installs
        raise ModelBackendUnavailable("PyYAML is required to read BasicSR config files.") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}
