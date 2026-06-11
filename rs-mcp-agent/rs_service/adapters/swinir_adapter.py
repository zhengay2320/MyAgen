from __future__ import annotations

import importlib.util
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from rs_service.adapters.base import AdapterMetadata, BaseAdapter, ModelBackendUnavailable, SuperResolutionPrediction
from rs_service.adapters.ultralytics_adapter import _to_hwc_uint8

try:  # pragma: no cover - pillow is available in normal installs
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


INSTALL_SWINIR_MESSAGE = (
    "SwinIR backend is unavailable. Install torch and provide either a compatible checkpoint "
    "or an external_command that runs a SwinIR inference script."
)


class SwinIRAdapter(BaseAdapter):
    """SwinIR super-resolution adapter with lazy torch import or external command support."""

    def __init__(self, model_config: dict[str, Any]) -> None:
        """Create a SwinIR adapter from checkpoint/scale/device settings."""
        self.config = dict(model_config)
        self.checkpoint = str(self.config.get("checkpoint") or self.config.get("weights") or self.config.get("weight") or "")
        self.scale = int(self.config.get("scale", 2))
        self.device = str(self.config.get("device", "cpu"))
        self.external_command = self.config.get("external_command")
        self.model: Any | None = None
        self.torch: Any | None = None
        self.metadata = AdapterMetadata(
            id=str(self.config.get("id", f"swinir_x{self.scale}")),
            task="super_resolution",
            backend="swinir",
            framework="swinir/torch",
            weights=self.checkpoint,
            description="SwinIR super-resolution adapter.",
        )

    def load(self) -> None:
        """Load SwinIR resources lazily."""
        if self.external_command:
            return
        if self.model is not None:
            return
        if importlib.util.find_spec("torch") is None:
            raise ModelBackendUnavailable(INSTALL_SWINIR_MESSAGE)
        if not self.checkpoint:
            raise FileNotFoundError("SwinIR checkpoint path is not configured for this model_id.")
        checkpoint_path = Path(self.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"SwinIR checkpoint not found: {checkpoint_path}. Place the weight file there or update configs/models.yaml."
            )
        try:
            import torch
        except Exception as exc:  # pragma: no cover - optional dependency internals
            raise ModelBackendUnavailable(f"{INSTALL_SWINIR_MESSAGE} Original error: {exc}") from exc
        self.torch = torch
        loaded = torch.load(str(checkpoint_path), map_location=self.device)
        if isinstance(loaded, dict) and callable(loaded.get("model")):
            loaded = loaded["model"]
        if not callable(loaded):
            raise ModelBackendUnavailable(
                "SwinIR checkpoint was loaded but does not contain a callable model. "
                "Use external_command for repository-specific SwinIR scripts."
            )
        self.model = loaded.to(self.device).eval() if hasattr(loaded, "to") else loaded

    def predict_tile(self, tile_array: np.ndarray, tile_info: dict[str, Any], **kwargs: Any) -> SuperResolutionPrediction:
        """Upscale one RGB-compatible tile."""
        scale = int(kwargs.get("scale", self.scale))
        if self.external_command:
            image = _run_external_command(self.external_command, tile_array, scale)
            return SuperResolutionPrediction(image=image.astype(tile_array.dtype, copy=False), scale=scale)
        self.load()
        assert self.torch is not None
        assert self.model is not None
        image = _to_hwc_uint8(tile_array)
        tensor = self.torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            output = self.model(tensor)
        if isinstance(output, (list, tuple)):
            output = output[0]
        if hasattr(output, "detach"):
            output = output.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        sr = np.clip(np.asarray(output) * 255.0, 0, 255).astype(np.uint8)
        return SuperResolutionPrediction(image=_match_input_bands(sr, tile_array), scale=scale)

    def upscale(self, tile: np.ndarray, context: dict[str, Any] | None = None) -> np.ndarray:
        """Backward-compatible upscale API."""
        return self.predict_tile(tile, _context_tile_info(context), scale=self.scale).image

    def close(self) -> None:
        """Release model references."""
        self.model = None
        self.torch = None


def _run_external_command(command: Any, tile_array: np.ndarray, scale: int) -> np.ndarray:
    """Run a repository-specific external SwinIR command with input/output placeholders."""
    if Image is None:
        raise ModelBackendUnavailable("Pillow is required to use SwinIR external_command inference.")
    if isinstance(command, str):
        parts = command.split()
    else:
        parts = [str(item) for item in command]
    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "tile.png"
        output_path = Path(tmp) / "tile_sr.png"
        Image.fromarray(_to_hwc_uint8(tile_array), mode="RGB").save(input_path)
        args = [
            item.format(input=str(input_path), output=str(output_path), scale=scale)
            for item in parts
        ]
        subprocess.run(args, check=True, capture_output=True)
        if not output_path.exists():
            raise RuntimeError(f"SwinIR external_command did not create expected output: {output_path}")
        sr_hwc = np.asarray(Image.open(output_path).convert("RGB"))
    return _match_input_bands(sr_hwc, tile_array)


def _match_input_bands(sr_hwc: np.ndarray, tile_array: np.ndarray) -> np.ndarray:
    """Convert HWC RGB output to CHW and match the input band count."""
    chw = np.moveaxis(sr_hwc, -1, 0)
    band_count = int(tile_array.shape[0]) if tile_array.ndim == 3 else 1
    if band_count == chw.shape[0]:
        return chw
    if band_count < chw.shape[0]:
        return chw[:band_count]
    pad = np.repeat(chw[-1:], band_count - chw.shape[0], axis=0)
    return np.concatenate([chw, pad], axis=0)


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
