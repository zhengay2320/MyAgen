from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np


def psnr(reference: np.ndarray, prediction: np.ndarray, data_range: float | None = None) -> float:
    """Calculate PSNR between reference and prediction arrays."""
    ref, pred = _paired_float_arrays(reference, prediction)
    if data_range is None:
        data_range = 255.0 if max(float(ref.max(initial=0.0)), float(pred.max(initial=0.0))) > 1.5 else 1.0
    mse = float(np.mean((ref - pred) ** 2))
    if mse <= 0:
        return float("inf")
    return float(20.0 * np.log10(float(data_range)) - 10.0 * np.log10(mse))


def ssim(reference: np.ndarray, prediction: np.ndarray, data_range: float | None = None) -> float:
    """Calculate a simplified global SSIM score."""
    ref, pred = _paired_float_arrays(reference, prediction)
    if data_range is None:
        data_range = 255.0 if max(float(ref.max(initial=0.0)), float(pred.max(initial=0.0))) > 1.5 else 1.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    scores: list[float] = []
    for ref_band, pred_band in zip(ref, pred):
        mu_x = float(np.mean(ref_band))
        mu_y = float(np.mean(pred_band))
        sigma_x = float(np.var(ref_band))
        sigma_y = float(np.var(pred_band))
        sigma_xy = float(np.mean((ref_band - mu_x) * (pred_band - mu_y)))
        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
        scores.append(float(numerator / denominator) if denominator else 1.0)
    return float(np.mean(scores)) if scores else 0.0


def lpips_optional(reference: np.ndarray, prediction: np.ndarray) -> tuple[float | None, str | None]:
    """Calculate LPIPS if optional dependencies are installed, otherwise return a warning."""
    if importlib.util.find_spec("lpips") is None or importlib.util.find_spec("torch") is None:
        return None, "LPIPS skipped because lpips/torch is not installed."
    try:  # pragma: no cover - optional heavyweight dependency
        import lpips
        import torch

        ref, pred = _paired_float_arrays(reference, prediction)
        ref_rgb = _first_three_bands(ref) / 127.5 - 1.0
        pred_rgb = _first_three_bands(pred) / 127.5 - 1.0
        loss_fn = lpips.LPIPS(net="alex")
        with torch.no_grad():
            ref_tensor = torch.from_numpy(ref_rgb).unsqueeze(0).float()
            pred_tensor = torch.from_numpy(pred_rgb).unsqueeze(0).float()
            score = loss_fn(ref_tensor, pred_tensor)
        return float(score.item()), None
    except Exception as exc:  # pragma: no cover
        return None, f"LPIPS skipped: {exc}"


def reference_metrics(reference: np.ndarray, prediction: np.ndarray) -> tuple[dict[str, Any], list[str]]:
    """Calculate reference-based SR metrics and optional warnings."""
    metrics = {
        "psnr": psnr(reference, prediction),
        "ssim": ssim(reference, prediction),
    }
    warnings: list[str] = []
    lpips_value, lpips_warning = lpips_optional(reference, prediction)
    if lpips_value is not None:
        metrics["lpips"] = lpips_value
    if lpips_warning:
        warnings.append(lpips_warning)
    return metrics, warnings


def _paired_float_arrays(reference: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return CHW float arrays with matching shape."""
    ref = _as_chw(reference).astype(np.float32)
    pred = _as_chw(prediction).astype(np.float32)
    bands = min(ref.shape[0], pred.shape[0])
    height = min(ref.shape[1], pred.shape[1])
    width = min(ref.shape[2], pred.shape[2])
    return ref[:bands, :height, :width], pred[:bands, :height, :width]


def _as_chw(array: np.ndarray) -> np.ndarray:
    """Convert 2D/CHW/HWC arrays to CHW."""
    arr = np.asarray(array)
    if arr.ndim == 2:
        return arr[np.newaxis, :, :]
    if arr.ndim == 3 and arr.shape[-1] <= 4 and arr.shape[0] > 4:
        return np.moveaxis(arr, -1, 0)
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Expected 2D or 3D array, got {arr.shape}")


def _first_three_bands(array: np.ndarray) -> np.ndarray:
    """Return three bands for perceptual metrics."""
    if array.shape[0] >= 3:
        return array[:3]
    return np.repeat(array[:1], 3, axis=0)
