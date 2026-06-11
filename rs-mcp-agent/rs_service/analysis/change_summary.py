from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rs_service.core.raster import read_raster


def summarize_change_mask(
    change_mask_path: str | Path,
    pixel_area: float,
    alignment_warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Summarize a binary change mask for reporting and quality checks."""
    data, _, _ = read_raster(change_mask_path)
    if data.ndim != 3 or data.shape[0] < 1:
        raise ValueError(f"Change mask has no readable bands: {change_mask_path}")
    mask = data[0] > 0
    changed_pixels = int(mask.sum())
    total_pixels = int(mask.size)
    components = _connected_component_sizes(mask)
    patch_areas = [size * pixel_area for size in components]
    return {
        "type": "change_detection",
        "mask_path": str(change_mask_path),
        "pixel_area_m2": pixel_area,
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "changed_area_m2": changed_pixels * pixel_area,
        "changed_area_km2": changed_pixels * pixel_area / 1_000_000.0,
        "changed_area_ratio": changed_pixels / total_pixels if total_pixels else 0.0,
        "change_patch_count": len(components),
        "largest_change_patch_m2": max(patch_areas) if patch_areas else 0.0,
        "small_change_patch_count": sum(1 for area in patch_areas if 0 < area < 9 * pixel_area),
        "alignment_warnings": alignment_warnings or [],
    }


def _connected_component_sizes(mask: np.ndarray) -> list[int]:
    """Return connected component sizes for a boolean mask using 4-neighbor fill."""
    binary = np.asarray(mask, dtype=bool)
    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    sizes: list[int] = []
    for y in range(height):
        for x in range(width):
            if not binary[y, x] or visited[y, x]:
                continue
            size = 0
            stack = [(x, y)]
            visited[y, x] = True
            while stack:
                cx, cy = stack.pop()
                size += 1
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < width and 0 <= ny < height and binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((nx, ny))
            sizes.append(size)
    return sizes
