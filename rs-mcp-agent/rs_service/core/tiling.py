from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np


@dataclass(frozen=True)
class Tile:
    tile_id: str
    x0: int
    y0: int
    x1: int
    y1: int
    data: np.ndarray

    @property
    def width(self) -> int:
        """Return tile width in pixels."""
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        """Return tile height in pixels."""
        return self.y1 - self.y0


@dataclass(frozen=True)
class TileSpec:
    """Metadata-only tile description for windowed raster processing."""

    tile_id: str
    row: int
    col: int
    x_off: int
    y_off: int
    width: int
    height: int

    @property
    def x0(self) -> int:
        """Return the left pixel offset."""
        return self.x_off

    @property
    def y0(self) -> int:
        """Return the top pixel offset."""
        return self.y_off

    @property
    def x1(self) -> int:
        """Return the exclusive right pixel coordinate."""
        return self.x_off + self.width

    @property
    def y1(self) -> int:
        """Return the exclusive bottom pixel coordinate."""
        return self.y_off + self.height

    def to_dict(self) -> dict[str, int | str]:
        """Serialize tile metadata to a plain dictionary."""
        return {
            "tile_id": self.tile_id,
            "row": self.row,
            "col": self.col,
            "x_off": self.x_off,
            "y_off": self.y_off,
            "width": self.width,
            "height": self.height,
        }


TASK_TILING_DEFAULTS: dict[str, tuple[int, int]] = {
    "detection": (1024, 192),
    "object_detection": (1024, 192),
    "oriented_detection": (1024, 192),
    "instance_segmentation": (1024, 192),
    "segmentation": (1024, 128),
    "semantic_segmentation": (1024, 128),
    "change_detection": (1024, 128),
    "super_resolution": (256, 32),
}


def validate_tiling(tile_size: int, overlap: int) -> None:
    """Validate shared tiling parameters."""
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= tile_size:
        raise ValueError("overlap must be smaller than tile_size")


def generate_tiles(width: int, height: int, tile_size: int, overlap: int) -> list[TileSpec]:
    """Generate metadata-only tiles, allowing smaller edge tiles."""
    validate_tiling(tile_size, overlap)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    stride = tile_size - overlap
    tiles: list[TileSpec] = []
    row = 0
    y_off = 0
    while y_off < height:
        col = 0
        x_off = 0
        tile_height = min(tile_size, height - y_off)
        while x_off < width:
            tile_width = min(tile_size, width - x_off)
            tiles.append(
                TileSpec(
                    tile_id=f"r{row:04d}_c{col:04d}",
                    row=row,
                    col=col,
                    x_off=x_off,
                    y_off=y_off,
                    width=tile_width,
                    height=tile_height,
                )
            )
            if x_off + tile_size >= width:
                break
            x_off += stride
            col += 1
        if y_off + tile_size >= height:
            break
        y_off += stride
        row += 1
    return tiles


def preflight_plan(
    width: int,
    height: int,
    task: str = "detection",
    tile_size: int | None = None,
    overlap: int | None = None,
) -> dict[str, Any]:
    """Build a tiling preflight plan with task-specific defaults."""
    default_tile_size, default_overlap = TASK_TILING_DEFAULTS.get(task, (1024, 128))
    resolved_tile_size = int(tile_size or default_tile_size)
    resolved_overlap = int(overlap if overlap is not None else default_overlap)
    tiles = generate_tiles(width, height, resolved_tile_size, resolved_overlap)
    return {
        "width": width,
        "height": height,
        "task": task,
        "tile_size": resolved_tile_size,
        "overlap": resolved_overlap,
        "tile_count": len(tiles),
        "tiles": [tile.to_dict() for tile in tiles],
    }


def iter_windows(width: int, height: int, tile_size: int, overlap: int) -> Iterator[tuple[int, int, int, int]]:
    """Yield legacy full-size-biased windows as x0, y0, x1, y1 tuples."""
    validate_tiling(tile_size, overlap)
    stride = tile_size - overlap
    y = 0
    while y < height:
        x = 0
        y1 = min(y + tile_size, height)
        y0 = max(0, y1 - tile_size) if y1 == height else y
        while x < width:
            x1 = min(x + tile_size, width)
            x0 = max(0, x1 - tile_size) if x1 == width else x
            yield x0, y0, x1, y1
            if x1 == width:
                break
            x += stride
        if y1 == height:
            break
        y += stride


def iter_tiles(array: np.ndarray, tile_size: int, overlap: int) -> Iterator[Tile]:
    """Yield legacy array tiles as CHW data slices."""
    if array.ndim != 3:
        raise ValueError("Expected CHW array")
    _, height, width = array.shape
    seen: set[tuple[int, int, int, int]] = set()
    for index, (x0, y0, x1, y1) in enumerate(iter_windows(width, height, tile_size, overlap)):
        key = (x0, y0, x1, y1)
        if key in seen:
            continue
        seen.add(key)
        yield Tile(
            tile_id=f"tile_{index:05d}",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            data=array[:, y0:y1, x0:x1],
        )


def empty_accumulator(height: int, width: int, channels: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Create value and weight accumulators for stitched raster predictions."""
    values = np.zeros((channels, height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    return values, weights


def add_tile_prediction(
    accumulator: np.ndarray,
    weights: np.ndarray,
    tile_prediction: np.ndarray,
    x0: int,
    y0: int,
) -> None:
    """Add a tile prediction to existing accumulators using uniform weights."""
    pred = tile_prediction if tile_prediction.ndim == 3 else tile_prediction[np.newaxis, :, :]
    _, tile_h, tile_w = pred.shape
    accumulator[:, y0 : y0 + tile_h, x0 : x0 + tile_w] += pred
    weights[y0 : y0 + tile_h, x0 : x0 + tile_w] += 1.0


def finalize_accumulator(accumulator: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Normalize accumulated values by accumulated weights."""
    safe_weights = np.maximum(weights, 1.0)
    return accumulator / safe_weights[np.newaxis, :, :]
