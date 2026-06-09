from __future__ import annotations

import unittest

import numpy as np

from rs_service.core.stitching import merge_detections_nms, stitch_segmentation_tiles, stitch_sr_tiles
from rs_service.core.tiling import TileSpec


class StitchingTests(unittest.TestCase):
    def test_stitch_segmentation_tiles_weighted_overlap(self) -> None:
        """Overlapping segmentation probabilities should blend with center-weighted weights."""
        tiles = [
            TileSpec("a", 0, 0, 0, 0, 4, 1),
            TileSpec("b", 0, 1, 2, 0, 4, 1),
        ]
        stitched = stitch_segmentation_tiles(
            [np.ones((1, 1, 4), dtype=np.float32), np.full((1, 1, 4), 3, dtype=np.float32)],
            tiles,
            output_shape=(1, 6),
        )
        self.assertEqual(stitched.shape, (1, 6))
        self.assertAlmostEqual(float(stitched[0, 0]), 1.0)
        self.assertGreater(float(stitched[0, 2]), 1.0)
        self.assertLess(float(stitched[0, 2]), 3.0)
        self.assertAlmostEqual(float(stitched[0, 5]), 3.0)

    def test_stitch_segmentation_integer_mask(self) -> None:
        """Integer masks should return a 2D integer mask when output shape is 2D."""
        tiles = [TileSpec("a", 0, 0, 0, 0, 2, 2)]
        stitched = stitch_segmentation_tiles([np.ones((2, 2), dtype=np.uint8)], tiles, output_shape=(2, 2))
        self.assertEqual(stitched.shape, (2, 2))
        self.assertEqual(stitched.dtype, np.uint8)
        self.assertEqual(int(stitched[1, 1]), 1)

    def test_stitch_sr_tiles_scaled_offsets(self) -> None:
        """Super-resolution tile offsets should be multiplied by scale."""
        tiles = [
            TileSpec("a", 0, 0, 0, 0, 2, 2),
            TileSpec("b", 0, 1, 2, 0, 2, 2),
        ]
        first = np.ones((1, 4, 4), dtype=np.uint8) * 10
        second = np.ones((1, 4, 4), dtype=np.uint8) * 20
        stitched = stitch_sr_tiles([first, second], tiles, output_shape=(1, 4, 8), scale=2)
        self.assertEqual(stitched.shape, (1, 4, 8))
        self.assertEqual(int(stitched[0, 0, 0]), 10)
        self.assertEqual(int(stitched[0, 0, 7]), 20)

    def test_merge_detections_nms(self) -> None:
        """Overlapping lower-score detections should be removed by NMS."""
        detections = [
            {"id": "a", "score": 0.9, "bbox_pixel": [0, 0, 10, 10]},
            {"id": "b", "score": 0.8, "bbox_pixel": [1, 1, 11, 11]},
            {"id": "c", "score": 0.7, "bbox_pixel": [20, 20, 30, 30]},
        ]
        merged = merge_detections_nms(detections, iou_threshold=0.5)
        self.assertEqual([item["id"] for item in merged], ["a", "c"])


if __name__ == "__main__":
    unittest.main()
