from __future__ import annotations

import unittest

from rs_service.core.georef import (
    pixel_bbox_to_geo_polygon,
    pixel_polygon_to_geo_polygon,
    scale_transform_for_super_resolution,
    tile_pixel_to_full_pixel,
)
from rs_service.core.tiling import TileSpec


class GeorefTests(unittest.TestCase):
    def test_tile_local_bbox_to_geo_polygon(self) -> None:
        """A tile-local bbox should translate to full pixels and then geospatial coordinates."""
        tile = TileSpec(tile_id="r0000_c0001", row=0, col=1, x_off=100, y_off=50, width=64, height=64)
        full_bbox = tile_pixel_to_full_pixel(tile, [10, 20, 30, 40])
        self.assertEqual(full_bbox, [110, 70, 130, 90])
        transform = (2.0, 0.0, 1000.0, 0.0, -3.0, 2000.0)
        polygon = pixel_bbox_to_geo_polygon(full_bbox, transform)
        self.assertEqual(polygon["type"], "Polygon")
        self.assertEqual(polygon["coordinates"][0][0], [1220.0, 1790.0])
        self.assertEqual(polygon["coordinates"][0][-1], [1220.0, 1790.0])

    def test_tile_local_polygon_to_geo_polygon(self) -> None:
        """A tile-local polygon should translate every point to full-image pixels."""
        tile = {"x_off": 5, "y_off": 7, "width": 10, "height": 10}
        full_points = tile_pixel_to_full_pixel(tile, [(0, 0), (2, 0), (2, 2), (0, 2)])
        self.assertEqual(full_points, [(5.0, 7.0), (7.0, 7.0), (7.0, 9.0), (5.0, 9.0)])
        polygon = pixel_polygon_to_geo_polygon(full_points, (1, 0, 0, 0, -1, 0))
        self.assertEqual(polygon["coordinates"][0][1], [7.0, -7.0])

    def test_scale_transform_for_super_resolution(self) -> None:
        """Super-resolution should scale pixel size while preserving origin."""
        scaled = scale_transform_for_super_resolution((4, 0, 100, 0, -6, 200), 2)
        self.assertEqual(scaled, (2.0, 0.0, 100.0, 0.0, -3.0, 200.0))


if __name__ == "__main__":
    unittest.main()
