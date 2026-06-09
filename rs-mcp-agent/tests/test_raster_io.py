from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rs_service.core.raster_io import inspect_raster, profile_for_array, read_window, validate_raster_path, write_geotiff


class RasterIoTests(unittest.TestCase):
    def test_write_inspect_and_read_window(self) -> None:
        """Raster IO should preserve profile metadata and read CHW windows."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.tif"
            array = np.arange(3 * 12 * 10, dtype=np.uint8).reshape(3, 12, 10)
            profile = profile_for_array(array, crs="EPSG:3857", transform=(2, 0, 100, 0, -2, 200))
            result = write_geotiff(path, array, profile, nodata=0)
            self.assertEqual(result["width"], 10)
            self.assertEqual(result["height"], 12)
            self.assertEqual(result["count"], 3)
            self.assertEqual(result["crs"], "EPSG:3857")

            inspected = inspect_raster(path)
            self.assertEqual(inspected["resolution"], (2.0, 2.0))
            tile = read_window(path, {"x_off": 2, "y_off": 3, "width": 4, "height": 5}, bands=[1, 3])
            self.assertEqual(tile.shape, (2, 5, 4))
            self.assertTrue(np.array_equal(tile[0], array[0, 3:8, 2:6]))
            self.assertTrue(np.array_equal(tile[1], array[2, 3:8, 2:6]))

    def test_validate_raster_path(self) -> None:
        """Path validation should reject missing files and directories."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                validate_raster_path(root / "missing.tif")
            with self.assertRaises(IsADirectoryError):
                validate_raster_path(root)

    def test_missing_crs_returns_warning(self) -> None:
        """Missing CRS should be reported as a warning rather than an exception."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "no_crs.tif"
            array = np.ones((1, 4, 4), dtype=np.uint8)
            profile = profile_for_array(array, crs=None, transform=(1, 0, 0, 0, -1, 0))
            write_geotiff(path, array, profile, crs=None)
            inspected = inspect_raster(path)
            self.assertIsNone(inspected["crs"])
            self.assertIn("Raster has no CRS.", inspected["warnings"])


if __name__ == "__main__":
    unittest.main()
