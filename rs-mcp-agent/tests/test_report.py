from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rs_service.analysis.report_builder import build_report_markdown


class ReportBuilderTests(unittest.TestCase):
    def test_report_contains_required_chinese_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.md"
            manifest = {
                "job_id": "job_1",
                "task": "semantic_segmentation",
                "status": "success",
                "input_files": ["input.tif"],
                "model_id": "fake_segmentation",
                "model": {"id": "fake_segmentation"},
                "parameters": {"tile_size": 512},
                "outputs": {"mask_geotiff": "mask.tif"},
                "statistics": {"total_area_km2": 0.1},
                "quality_flags": [],
                "conclusion": {
                    "summary": "基于统计结果生成。",
                    "confidence": "high",
                    "key_findings": ["类别占比正常"],
                    "limitations": ["未提供人工真值。"],
                    "recommended_review_areas": ["抽样检查主要输出区域"],
                    "next_steps": ["人工复核。"],
                },
            }
            report_path = build_report_markdown(manifest, path)
            text = Path(report_path).read_text(encoding="utf-8")
            for section in ["数据概况", "使用模型", "统计分析", "质量检查", "风险与局限", "结论", "输出文件清单"]:
                self.assertIn(section, text)


if __name__ == "__main__":
    unittest.main()
