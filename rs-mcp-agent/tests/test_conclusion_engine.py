from __future__ import annotations

import unittest

from rs_service.analysis.conclusion_engine import build_conclusion


class ConclusionEngineTests(unittest.TestCase):
    def test_warning_lowers_confidence(self) -> None:
        conclusion = build_conclusion(
            {"task": "object_detection"},
            {"target_count": 2, "mean_confidence": 0.7, "low_confidence_ratio": 0.0},
            [{"code": "low_confidence_warning", "severity": "warning", "message": "x"}],
        )
        self.assertEqual(conclusion["confidence"], "medium")
        self.assertTrue(any("未提供人工真值" in item for item in conclusion["limitations"]))

    def test_change_detection_registration_risk_is_low_confidence(self) -> None:
        conclusion = build_conclusion(
            {"task": "change_detection"},
            {"change_area_ratio": 0.2, "changed_area_km2": 1.0, "max_patch_area_m2": 20},
            [{"code": "georeference_mismatch", "severity": "warning", "message": "risk"}],
        )
        self.assertEqual(conclusion["confidence"], "low")
        self.assertTrue(any("配准" in item for item in conclusion["limitations"]))

    def test_super_resolution_does_not_claim_quality_gain(self) -> None:
        conclusion = build_conclusion(
            {"task": "super_resolution"},
            {"type": "super_resolution", "shape_matches_scale": True, "transform_matches_scale": True},
            [],
        )
        text = " ".join(conclusion["limitations"] + [conclusion["summary"]])
        self.assertNotIn("PSNR", conclusion["key_findings"])
        self.assertIn("不能声称结果更清晰或更准确", text)


if __name__ == "__main__":
    unittest.main()
