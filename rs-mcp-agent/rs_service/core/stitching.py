from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from rs_service.core.geometry import iou


def stitch_segmentation_tiles(
    tile_predictions: Iterable[np.ndarray],
    tiles: Iterable[Any],
    output_shape: tuple[int, int] | tuple[int, int, int],
) -> np.ndarray:
    """Stitch segmentation masks or probability maps with center-weighted overlap blending."""
    predictions = list(tile_predictions)
    tile_list = list(tiles)
    if len(predictions) != len(tile_list):
        raise ValueError("tile_predictions and tiles must have the same length")
    channels, height, width = _resolve_output_shape(output_shape, predictions)
    accumulator = np.zeros((channels, height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    integer_mask = all(pred.ndim == 2 and np.issubdtype(pred.dtype, np.integer) for pred in predictions)
    for prediction, tile in zip(predictions, tile_list):
        pred = _as_chw(prediction).astype(np.float32)
        x_off, y_off, tile_width, tile_height = _tile_window(tile)
        pred = pred[:, :tile_height, :tile_width]
        weight = _center_weight(tile_height, tile_width)
        accumulator[:, y_off : y_off + tile_height, x_off : x_off + tile_width] += pred * weight
        weights[y_off : y_off + tile_height, x_off : x_off + tile_width] += weight
    stitched = accumulator / np.maximum(weights, 1e-6)[np.newaxis, :, :]
    if integer_mask:
        return np.rint(stitched[0]).astype(predictions[0].dtype)
    return stitched[0] if len(output_shape) == 2 and channels == 1 else stitched


def stitch_sr_tiles(
    tile_predictions: Iterable[np.ndarray],
    tiles: Iterable[Any],
    output_shape: tuple[int, int] | tuple[int, int, int],
    scale: int = 1,
) -> np.ndarray:
    """Stitch super-resolution tiles with center-weighted overlap blending."""
    predictions = list(tile_predictions)
    tile_list = list(tiles)
    if len(predictions) != len(tile_list):
        raise ValueError("tile_predictions and tiles must have the same length")
    channels, height, width = _resolve_output_shape(output_shape, predictions)
    accumulator = np.zeros((channels, height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    output_dtype = predictions[0].dtype if predictions else np.float32
    for prediction, tile in zip(predictions, tile_list):
        pred = _as_chw(prediction).astype(np.float32)
        x_off, y_off, tile_width, tile_height = _tile_window(tile)
        sx = x_off * scale
        sy = y_off * scale
        scaled_width = min(tile_width * scale, pred.shape[2], width - sx)
        scaled_height = min(tile_height * scale, pred.shape[1], height - sy)
        pred = pred[:, :scaled_height, :scaled_width]
        weight = _center_weight(scaled_height, scaled_width)
        accumulator[:, sy : sy + scaled_height, sx : sx + scaled_width] += pred * weight
        weights[sy : sy + scaled_height, sx : sx + scaled_width] += weight
    stitched = accumulator / np.maximum(weights, 1e-6)[np.newaxis, :, :]
    if np.issubdtype(output_dtype, np.integer):
        info = np.iinfo(output_dtype)
        stitched = np.clip(np.rint(stitched), info.min, info.max).astype(output_dtype)
    return stitched[0] if len(output_shape) == 2 and channels == 1 else stitched.astype(output_dtype, copy=False)


def merge_detections_nms(detections: list[dict[str, Any]], iou_threshold: float = 0.5, score_key: str = "score") -> list[dict[str, Any]]:
    """Merge detection records with greedy NMS using full-image pixel bboxes."""
    ordered = sorted(detections, key=lambda item: float(item.get(score_key, 0.0)), reverse=True)
    kept: list[dict[str, Any]] = []
    for detection in ordered:
        bbox = _detection_bbox(detection)
        if bbox is None:
            kept.append(detection)
            continue
        if all(iou(bbox, _detection_bbox(existing) or [0, 0, 0, 0]) < iou_threshold for existing in kept):
            kept.append(detection)
    return kept


def _resolve_output_shape(output_shape: tuple[int, ...], predictions: list[np.ndarray]) -> tuple[int, int, int]:
    """Resolve output shape to C, H, W."""
    if len(output_shape) == 2:
        if not predictions:
            return 1, int(output_shape[0]), int(output_shape[1])
        return _as_chw(predictions[0]).shape[0], int(output_shape[0]), int(output_shape[1])
    if len(output_shape) == 3:
        return int(output_shape[0]), int(output_shape[1]), int(output_shape[2])
    raise ValueError("output_shape must be (height, width) or (channels, height, width)")


def _as_chw(array: np.ndarray) -> np.ndarray:
    """Convert 2D or CHW arrays to CHW."""
    arr = np.asarray(array)
    if arr.ndim == 2:
        return arr[np.newaxis, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or CHW array, got {arr.shape}")
    return arr


def _tile_window(tile: Any) -> tuple[int, int, int, int]:
    """Read x offset, y offset, width, and height from a tile-like object."""
    if isinstance(tile, dict):
        return int(tile.get("x_off", tile.get("x0", 0))), int(tile.get("y_off", tile.get("y0", 0))), int(tile["width"]), int(tile["height"])
    if all(hasattr(tile, name) for name in ["x_off", "y_off", "width", "height"]):
        return int(tile.x_off), int(tile.y_off), int(tile.width), int(tile.height)
    if all(hasattr(tile, name) for name in ["x0", "y0", "width", "height"]):
        return int(tile.x0), int(tile.y0), int(tile.width), int(tile.height)
    raise ValueError(f"Unsupported tile: {tile!r}")


def _center_weight(height: int, width: int) -> np.ndarray:
    """Create a 2D weight map with lower edge weights and higher center weights."""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    y = np.minimum(np.arange(height) + 1, np.arange(height, 0, -1)).astype(np.float32)
    x = np.minimum(np.arange(width) + 1, np.arange(width, 0, -1)).astype(np.float32)
    y = y / max(float(y.max()), 1.0)
    x = x / max(float(x.max()), 1.0)
    return np.maximum(np.outer(y, x), 1e-3)


def _detection_bbox(detection: dict[str, Any]) -> list[float] | None:
    """Extract a pixel bbox from a detection record."""
    bbox = detection.get("bbox_pixel", detection.get("bbox"))
    if bbox is None:
        return None
    return [float(value) for value in bbox]
