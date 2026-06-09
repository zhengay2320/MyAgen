from __future__ import annotations

from typing import Any

import numpy as np

from rs_service.adapters.base import AdapterMetadata


def _tile_score(tile: np.ndarray) -> float:
    mean = float(np.mean(tile)) if tile.size else 0.0
    return round(0.55 + min(mean / 255.0, 1.0) * 0.4, 4)


class FakeDetectionAdapter:
    metadata = AdapterMetadata(
        id="fake-yolo-sahi",
        task="object_detection",
        backend="fake",
        framework="ultralytics+sahi",
        description="Deterministic fake object detector for integration tests.",
    )

    def predict(self, tile: np.ndarray, context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        _, height, width = tile.shape
        box_w = max(8, width // 4)
        box_h = max(8, height // 5)
        x1 = max(0, width // 2 - box_w // 2)
        y1 = max(0, height // 2 - box_h // 2)
        return [
            {
                "label": "target",
                "score": _tile_score(tile),
                "bbox": [x1, y1, min(width, x1 + box_w), min(height, y1 + box_h)],
            }
        ]


class FakeOrientedDetectionAdapter:
    metadata = AdapterMetadata(
        id="fake-mmrotate",
        task="oriented_detection",
        backend="fake",
        framework="mmrotate",
        description="Deterministic fake rotated-box detector for integration tests.",
    )

    def predict(self, tile: np.ndarray, context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        _, height, width = tile.shape
        return [
            {
                "label": "oriented_target",
                "score": _tile_score(tile),
                "rotated_box": {
                    "cx": width / 2.0,
                    "cy": height / 2.0,
                    "width": max(10.0, width / 3.0),
                    "height": max(8.0, height / 5.0),
                    "angle_degrees": 25.0,
                },
            }
        ]


class FakeSemanticSegmentationAdapter:
    metadata = AdapterMetadata(
        id="fake-mmseg",
        task="semantic_segmentation",
        backend="fake",
        framework="mmsegmentation",
        description="Deterministic fake semantic segmenter for integration tests.",
    )

    def predict_proba(self, tile: np.ndarray, context: dict[str, Any] | None = None) -> np.ndarray:
        red = tile[min(2, tile.shape[0] - 1)].astype(np.float32)
        nir = tile[min(3, tile.shape[0] - 1)].astype(np.float32)
        ndvi_like = (nir - red) / np.maximum(nir + red, 1.0)
        foreground = np.clip((ndvi_like + 0.2) / 0.8, 0.0, 1.0)
        background = 1.0 - foreground
        return np.stack([background, foreground], axis=0).astype(np.float32)


class FakeInstanceSegmentationAdapter:
    metadata = AdapterMetadata(
        id="fake-mmdet-instance",
        task="instance_segmentation",
        backend="fake",
        framework="mmdetection",
        description="Deterministic fake instance segmenter for integration tests.",
    )

    def predict(self, tile: np.ndarray, context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        _, height, width = tile.shape
        x1 = width * 0.25
        y1 = height * 0.25
        x2 = width * 0.75
        y2 = height * 0.70
        polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
        return [
            {
                "label": "instance",
                "score": _tile_score(tile),
                "bbox": [x1, y1, x2, y2],
                "mask_polygon": polygon,
            }
        ]


class FakeChangeDetectionAdapter:
    metadata = AdapterMetadata(
        id="fake-opencd",
        task="change_detection",
        backend="fake",
        framework="open-cd",
        description="Deterministic fake bi-temporal change detector for integration tests.",
    )

    def predict_proba(
        self,
        tile_a: np.ndarray,
        tile_b: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> np.ndarray:
        diff = np.mean(np.abs(tile_b.astype(np.float32) - tile_a.astype(np.float32)), axis=0)
        return np.clip(diff / 128.0, 0.0, 1.0).astype(np.float32)


class FakeSuperResolutionAdapter:
    metadata = AdapterMetadata(
        id="fake-sr",
        task="super_resolution",
        backend="fake",
        framework="basicsr/mmagic/swinir",
        description="Nearest-neighbor fake super-resolution adapter for integration tests.",
    )

    def __init__(self, scale: int = 2) -> None:
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.scale = int(scale)

    def upscale(self, tile: np.ndarray, context: dict[str, Any] | None = None) -> np.ndarray:
        return np.repeat(np.repeat(tile, self.scale, axis=1), self.scale, axis=2)
