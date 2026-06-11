from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rs_service.analysis.statistics import (
    change_statistics,
    detection_statistics,
    segmentation_statistics,
    spectral_index_statistics,
    super_resolution_statistics,
)
from rs_service.core.raster import profile_from_array, write_raster


class StatisticsTests(unittest.TestCase):
    def test_segmentation_and_change_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask = np.zeros((1, 10, 10), dtype=np.uint8)
            mask[0, 2:5, 3:7] = 1
            mask_path = root / "mask.tif"
            write_raster(mask_path, mask, profile_from_array(mask, transform=(2, 0, 0, 0, -2, 0)))

            stats = segmentation_statistics(mask_path, {"classes": {0: {"name": "bg"}, 1: {"name": "target"}}}, pixel_area=4.0)
            self.assertEqual(stats["classes"]["1"]["area_m2"], 48.0)
            self.assertEqual(stats["connected_component_count"], 2)

            change = change_statistics(mask_path, pixel_area=4.0)
            self.assertEqual(change["changed_area_m2"], 48.0)
            self.assertAlmostEqual(change["change_area_ratio"], 0.12)

    def test_detection_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "detections.geojson"
            payload = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"score": 0.9, "label": "ship"},
                        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]},
                    },
                    {
                        "type": "Feature",
                        "properties": {"score": 0.4, "label": "ship"},
                        "geometry": {"type": "Polygon", "coordinates": [[[20, 0], [30, 0], [30, 10], [20, 10], [20, 0]]]},
                    },
                ],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            stats = detection_statistics(path)
            self.assertEqual(stats["target_count"], 2)
            self.assertAlmostEqual(stats["mean_confidence"], 0.65)
            self.assertEqual(stats["low_confidence_ratio"], 0.5)

    def test_super_resolution_and_index_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = np.ones((1, 4, 5), dtype=np.uint8)
            out = np.ones((1, 8, 10), dtype=np.uint8)
            src_path = root / "src.tif"
            out_path = root / "out.tif"
            write_raster(src_path, src, profile_from_array(src, transform=(4, 0, 100, 0, -4, 200)))
            write_raster(out_path, out, profile_from_array(out, transform=(2, 0, 100, 0, -2, 200)))
            stats = super_resolution_statistics(src_path, out_path, scale=2)
            self.assertTrue(stats["shape_matches_scale"])
            self.assertTrue(stats["transform_matches_scale"])

            index_path = root / "ndvi.tif"
            index = np.array([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32)
            write_raster(index_path, index, profile_from_array(index, dtype="float32"))
            index_stats = spectral_index_statistics(index_path)
            self.assertAlmostEqual(index_stats["mean"], 0.25, places=6)


if __name__ == "__main__":
    unittest.main()
