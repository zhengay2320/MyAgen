from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

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
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def validate_tiling(tile_size: int, overlap: int) -> None:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= tile_size:
        raise ValueError("overlap must be smaller than tile_size")


def iter_windows(width: int, height: int, tile_size: int, overlap: int) -> Iterator[tuple[int, int, int, int]]:
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
    pred = tile_prediction if tile_prediction.ndim == 3 else tile_prediction[np.newaxis, :, :]
    _, tile_h, tile_w = pred.shape
    accumulator[:, y0 : y0 + tile_h, x0 : x0 + tile_w] += pred
    weights[y0 : y0 + tile_h, x0 : x0 + tile_w] += 1.0


def finalize_accumulator(accumulator: np.ndarray, weights: np.ndarray) -> np.ndarray:
    safe_weights = np.maximum(weights, 1.0)
    return accumulator / safe_weights[np.newaxis, :, :]
