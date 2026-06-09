from __future__ import annotations

import math
from typing import Iterable

from rs_service.core.raster import pixel_to_geo


def clip_box(box: Iterable[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    ]


def translate_box(box: Iterable[float], x_offset: int, y_offset: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset]


def box_to_pixel_polygon(box: Iterable[float]) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]


def rotated_box_to_pixel_polygon(
    cx: float,
    cy: float,
    width: float,
    height: float,
    angle_degrees: float,
) -> list[tuple[float, float]]:
    angle = math.radians(angle_degrees)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    half_w = width / 2.0
    half_h = height / 2.0
    corners = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
    polygon = []
    for x, y in corners:
        rx = cx + x * cos_a - y * sin_a
        ry = cy + x * sin_a + y * cos_a
        polygon.append((rx, ry))
    polygon.append(polygon[0])
    return polygon


def pixel_polygon_to_geojson_coords(
    pixel_polygon: Iterable[tuple[float, float]],
    transform: Iterable[float],
) -> list[list[float]]:
    return [[float(gx), float(gy)] for gx, gy in (pixel_to_geo(transform, x, y) for x, y in pixel_polygon)]


def polygon_bounds(polygon: Iterable[tuple[float, float]]) -> list[float]:
    points = list(polygon)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else intersection / union


def nms(records: list[dict], threshold: float = 0.5) -> list[dict]:
    ordered = sorted(records, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    kept: list[dict] = []
    for record in ordered:
        box = record.get("bbox_pixel")
        if not box:
            kept.append(record)
            continue
        if all(iou(box, other.get("bbox_pixel", [0, 0, 0, 0])) < threshold for other in kept):
            kept.append(record)
    return kept
