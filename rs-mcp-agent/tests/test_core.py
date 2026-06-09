from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rs_service.core.raster import (
    DEFAULT_TRANSFORM,
    create_synthetic_array,
    inspect_raster,
    profile_from_array,
    read_raster,
    update_transform_for_super_resolution,
    write_raster,
)
from rs_service.core.tiling import iter_windows


class CoreTests(unittest.TestCase):
    def test_write_read_inspect_raster(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.tif"
            array = create_synthetic_array(width=64, height=48, bands=4)
            write_raster(path, array, profile_from_array(array))
            loaded, _, info = read_raster(path)
            self.assertEqual(loaded.shape, (4, 48, 64))
            self.assertEqual(info.width, 64)
            self.assertEqual(info.height, 48)
            inspected = inspect_raster(path)
            self.assertTrue(inspected["is_georeferenced"])
            self.assertEqual(inspected["count"], 4)

    def test_tiling_and_transform_update(self) -> None:
        windows = list(iter_windows(width=100, height=80, tile_size=32, overlap=8))
        self.assertGreater(len(windows), 1)
        self.assertEqual(windows[0], (0, 0, 32, 32))
        self.assertEqual(windows[-1], (68, 48, 100, 80))
        updated = update_transform_for_super_resolution(DEFAULT_TRANSFORM, 2)
        self.assertEqual(updated[0], DEFAULT_TRANSFORM[0] / 2)
        self.assertEqual(updated[4], DEFAULT_TRANSFORM[4] / 2)
        self.assertEqual(updated[2], DEFAULT_TRANSFORM[2])


if __name__ == "__main__":
    unittest.main()
