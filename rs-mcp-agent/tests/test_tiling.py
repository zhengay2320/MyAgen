from __future__ import annotations

import unittest

from rs_service.core.tiling import generate_tiles, preflight_plan


class TilingTests(unittest.TestCase):
    def test_generate_tiles_allows_small_edge_tiles(self) -> None:
        """Tiles should cover the image and keep edge tiles smaller when needed."""
        tiles = generate_tiles(width=100, height=70, tile_size=32, overlap=8)
        self.assertEqual(tiles[0].tile_id, "r0000_c0000")
        self.assertEqual((tiles[0].x_off, tiles[0].y_off, tiles[0].width, tiles[0].height), (0, 0, 32, 32))
        self.assertEqual((tiles[-1].x_off, tiles[-1].y_off), (72, 48))
        self.assertEqual((tiles[-1].width, tiles[-1].height), (28, 22))
        self.assertEqual(max(tile.x1 for tile in tiles), 100)
        self.assertEqual(max(tile.y1 for tile in tiles), 70)

    def test_generate_tiles_validates_overlap(self) -> None:
        """Overlap must be smaller than tile size."""
        with self.assertRaises(ValueError):
            generate_tiles(width=100, height=100, tile_size=32, overlap=32)

    def test_preflight_uses_task_defaults(self) -> None:
        """Preflight should use task-specific defaults and include tile metadata."""
        plan = preflight_plan(width=2048, height=1200, task="super_resolution")
        self.assertEqual(plan["tile_size"], 256)
        self.assertEqual(plan["overlap"], 32)
        self.assertGreater(plan["tile_count"], 1)
        self.assertIn("tile_id", plan["tiles"][0])


if __name__ == "__main__":
    unittest.main()
