from __future__ import annotations

from typing import Any

import numpy as np

from rs_service.adapters.base import AdapterMetadata, BaseAdapter, DetectionPrediction


class FakeDetectionAdapter(BaseAdapter):
    """Fake detector that creates boxes around bright tile regions."""

    metadata = AdapterMetadata(
        id="fake_detection",
        task="object_detection",
        backend="fake",
        framework="numpy",
        description="Bright-region fake detector for smoke tests.",
    )

    def predict_tile(self, tile_array: np.ndarray, tile_info: dict[str, Any], **kwargs: Any) -> list[DetectionPrediction]:
        """Predict tile-local boxes around bright connected regions."""
        gray = np.mean(tile_array.astype(np.float32), axis=0)
        threshold = float(kwargs.get("threshold", max(150.0, float(gray.mean() + gray.std()))))
        mask = gray >= threshold
        boxes = _component_boxes(mask, min_pixels=int(kwargs.get("min_pixels", 12)))
        if not boxes and gray.size:
            y, x = np.unravel_index(int(np.argmax(gray)), gray.shape)
            half = max(4, min(gray.shape) // 12)
            boxes = [[max(0, x - half), max(0, y - half), min(gray.shape[1], x + half), min(gray.shape[0], y + half)]]
        predictions: list[DetectionPrediction] = []
        for index, bbox in enumerate(boxes[: int(kwargs.get("max_detections", 8))]):
            x1, y1, x2, y2 = bbox
            region = gray[int(y1) : int(y2), int(x1) : int(x2)]
            score = float(np.clip(region.mean() / 255.0 if region.size else 0.5, 0.05, 0.99))
            predictions.append(
                DetectionPrediction(
                    label="bright_object",
                    score=round(score, 4),
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    class_id=0,
                    attributes={"fake_index": index, "tile_id": tile_info.get("tile_id")},
                )
            )
        return predictions

    def predict(self, tile: np.ndarray, context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Backward-compatible dict prediction API."""
        tile_info = _context_tile_info(context)
        return [prediction.to_dict() for prediction in self.predict_tile(tile, tile_info)]


class FakeOrientedDetectionAdapter(FakeDetectionAdapter):
    """Fake oriented detector that wraps bright boxes with rotated-box metadata."""

    metadata = AdapterMetadata(
        id="fake_oriented_detection",
        task="oriented_detection",
        backend="fake",
        framework="numpy",
        description="Bright-region fake oriented detector for smoke tests.",
    )

    def predict_tile(self, tile_array: np.ndarray, tile_info: dict[str, Any], **kwargs: Any) -> list[DetectionPrediction]:
        """Predict tile-local rotated boxes."""
        detections = super().predict_tile(tile_array, tile_info, **kwargs)
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            detection.label = "oriented_bright_object"
            detection.rotated_box = {
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
                "width": max(1.0, x2 - x1),
                "height": max(1.0, y2 - y1),
                "angle_degrees": 25.0,
            }
        return detections

    def predict(self, tile: np.ndarray, context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Backward-compatible dict prediction API."""
        result = []
        for prediction in self.predict_tile(tile, _context_tile_info(context)):
            payload = prediction.to_dict()
            payload["rotated_box"] = prediction.rotated_box
            result.append(payload)
        return result


def _component_boxes(mask: np.ndarray, min_pixels: int) -> list[list[int]]:
    """Find simple connected-component boxes using 4-neighbor flood fill."""
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    boxes: list[list[int]] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(x, y)]
            visited[y, x] = True
            xs: list[int] = []
            ys: list[int] = []
            while stack:
                cx, cy = stack.pop()
                xs.append(cx)
                ys.append(cy)
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((nx, ny))
            if len(xs) >= min_pixels:
                boxes.append([min(xs), min(ys), max(xs) + 1, max(ys) + 1])
    return boxes


def _context_tile_info(context: dict[str, Any] | None) -> dict[str, Any]:
    """Extract tile metadata from a legacy context dict."""
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
