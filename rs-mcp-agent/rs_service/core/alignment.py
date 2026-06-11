from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rs_service.core.raster import RasterInfo, read_raster, write_raster


def check_pair_alignment(reference: RasterInfo, candidate: RasterInfo) -> dict[str, Any]:
    """Check whether two rasters are aligned for pixel-wise change detection."""
    issues: list[str] = []
    if reference.crs != candidate.crs:
        issues.append("crs_mismatch")
    if reference.width != candidate.width or reference.height != candidate.height:
        issues.append("shape_mismatch")
    if tuple(reference.transform) != tuple(candidate.transform):
        issues.append("transform_mismatch")
    if tuple(round(v, 9) for v in reference.bounds) != tuple(round(v, 9) for v in candidate.bounds):
        issues.append("bounds_mismatch")
    reference_res = (abs(reference.transform[0]), abs(reference.transform[4]))
    candidate_res = (abs(candidate.transform[0]), abs(candidate.transform[4]))
    if tuple(round(v, 9) for v in reference_res) != tuple(round(v, 9) for v in candidate_res):
        issues.append("resolution_mismatch")
    return {
        "aligned": not issues,
        "issues": issues,
        "reference": {
            "crs": reference.crs,
            "width": reference.width,
            "height": reference.height,
            "transform": reference.transform,
            "bounds": reference.bounds,
            "resolution": reference_res,
        },
        "candidate": {
            "crs": candidate.crs,
            "width": candidate.width,
            "height": candidate.height,
            "transform": candidate.transform,
            "bounds": candidate.bounds,
            "resolution": candidate_res,
        },
    }


def align_raster_to_reference(
    candidate_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
) -> tuple[np.ndarray, dict[str, Any], RasterInfo, list[str]]:
    """Align a raster to a reference grid using MVP nearest-neighbor resampling."""
    reference_data, reference_profile, reference_info = read_raster(reference_path)
    candidate_data, _, candidate_info = read_raster(candidate_path)
    warnings = estimate_simple_alignment_warning(reference_info, candidate_info)
    aligned = _resize_nearest(candidate_data, reference_info.height, reference_info.width)
    if aligned.shape[0] != reference_data.shape[0]:
        aligned = _match_band_count(aligned, reference_data.shape[0])
        warnings.append("band_count_adjusted")
    profile = dict(reference_profile)
    profile.update(count=int(aligned.shape[0]), height=reference_info.height, width=reference_info.width, dtype=str(aligned.dtype))
    aligned_info = write_raster(output_path, aligned, profile, crs=reference_info.crs, transform=reference_info.transform, dtype=str(aligned.dtype))
    return aligned, profile, aligned_info, warnings


def estimate_simple_alignment_warning(reference: RasterInfo, candidate: RasterInfo) -> list[str]:
    """Return warnings for coarse resampling-only alignment."""
    alignment = check_pair_alignment(reference, candidate)
    warnings = [f"alignment_{issue}" for issue in alignment["issues"]]
    if warnings:
        warnings.append("simple_resampling_only_no_feature_registration")
    return warnings


def _resize_nearest(array: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a CHW array with nearest-neighbor sampling."""
    if array.ndim != 3:
        raise ValueError(f"Expected CHW array, got {array.shape}")
    y_idx = np.linspace(0, array.shape[1] - 1, height).round().astype(np.int64)
    x_idx = np.linspace(0, array.shape[2] - 1, width).round().astype(np.int64)
    return array[:, y_idx][:, :, x_idx]


def _match_band_count(array: np.ndarray, count: int) -> np.ndarray:
    """Match candidate band count to the reference band count."""
    if array.shape[0] == count:
        return array
    if array.shape[0] > count:
        return array[:count]
    pad = np.repeat(array[-1:], count - array.shape[0], axis=0)
    return np.concatenate([array, pad], axis=0)
