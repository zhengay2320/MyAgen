from __future__ import annotations

import unittest

from rs_service.analysis.quality_checks import run_quality_checks


class QualityCheckTests(unittest.TestCase):
    def test_quality_flags_from_statistics(self) -> None:
        manifest = {"inputs": {"raster": {"crs": None}}, "parameters": {"tile_size": 512, "overlap": 8}, "quality_flags": []}
        statistics = {
            "target_count": 0,
            "low_confidence_ratio": 0.6,
            "edge_object_ratio": 0.4,
            "connected_component_count": 10,
            "small_patch_count": 8,
        }
        flags = run_quality_checks(manifest, statistics)
        codes = {item["code"] for item in flags}
        self.assertIn("no_crs_warning", codes)
        self.assertIn("empty_result_warning", codes)
        self.assertIn("low_confidence_warning", codes)
        self.assertIn("tile_seam_risk_warning", codes)

    def test_change_and_sr_warnings(self) -> None:
        change_flags = run_quality_checks({}, {"change_area_ratio": 0.9})
        self.assertIn("change_area_too_high_warning", {item["code"] for item in change_flags})
        sr_flags = run_quality_checks({}, {"type": "super_resolution", "transform_matches_scale": False})
        self.assertIn("sr_transform_inconsistent_warning", {item["code"] for item in sr_flags})


if __name__ == "__main__":
    unittest.main()
