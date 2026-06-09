from __future__ import annotations

from typing import Any, Iterable, Sequence

from rs_service.core.raster import pixel_to_geo, update_transform_for_super_resolution

BBox = Sequence[float]
Point = tuple[float, float]


def pixel_bbox_to_geo_polygon(bbox: BBox, transform: Iterable[float]) -> dict[str, Any]:
    """Convert a full-image pixel bounding box to a GeoJSON polygon."""
    x1, y1, x2, y2 = [float(value) for value in bbox]
    points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    return pixel_polygon_to_geo_polygon(points, transform)


def pixel_polygon_to_geo_polygon(points: Iterable[Sequence[float]], transform: Iterable[float]) -> dict[str, Any]:
    """Convert full-image pixel polygon points to a GeoJSON polygon."""
    pixel_points = [(float(point[0]), float(point[1])) for point in points]
    if not pixel_points:
        raise ValueError("points must not be empty")
    if pixel_points[0] != pixel_points[-1]:
        pixel_points.append(pixel_points[0])
    geo_points = [pixel_to_geo(transform, x, y) for x, y in pixel_points]
    return {
        "type": "Polygon",
        "coordinates": [[[float(x), float(y)] for x, y in geo_points]],
    }


def tile_pixel_to_full_pixel(tile: Any, local_bbox_or_polygon: Any) -> Any:
    """Translate tile-local bbox or polygon coordinates to full-image pixel coordinates."""
    x_off, y_off = _tile_offsets(tile)
    if _is_bbox(local_bbox_or_polygon):
        x1, y1, x2, y2 = [float(value) for value in local_bbox_or_polygon]
        return [x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off]
    points = [(float(point[0]) + x_off, float(point[1]) + y_off) for point in local_bbox_or_polygon]
    return points


def scale_transform_for_super_resolution(transform: Iterable[float], scale: int | float) -> tuple[float, float, float, float, float, float]:
    """Scale a transform so super-resolved pixels keep the same geospatial extent."""
    return update_transform_for_super_resolution(transform, scale)


def _tile_offsets(tile: Any) -> tuple[float, float]:
    """Read x/y offsets from a tile dataclass, rasterio window, or dict."""
    if isinstance(tile, dict):
        return float(tile.get("x_off", tile.get("x0", 0))), float(tile.get("y_off", tile.get("y0", 0)))
    if hasattr(tile, "x_off") and hasattr(tile, "y_off"):
        return float(tile.x_off), float(tile.y_off)
    if hasattr(tile, "x0") and hasattr(tile, "y0"):
        return float(tile.x0), float(tile.y0)
    if hasattr(tile, "col_off") and hasattr(tile, "row_off"):
        return float(tile.col_off), float(tile.row_off)
    raise ValueError(f"Unsupported tile object: {tile!r}")


def _is_bbox(value: Any) -> bool:
    """Return True when a value looks like a four-number bbox."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    return all(isinstance(item, (int, float)) for item in value)
