from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rs_service import services
from rs_service.core.raster import create_synthetic_array, profile_from_array, read_raster, write_raster


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        before = create_synthetic_array(width=96, height=72, bands=4, changed=False)
        after = create_synthetic_array(width=96, height=72, bands=4, changed=True)
        self.before_path = self.root / "before.tif"
        self.after_path = self.root / "after.tif"
        write_raster(self.before_path, before, profile_from_array(before))
        write_raster(self.after_path, after, profile_from_array(after))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_detection_outputs_manifest_and_geojson(self) -> None:
        manifest = services.run_object_detection(str(self.before_path), str(self.root / "det"), tile_size=40, overlap=8)
        self.assertEqual(manifest["task"], "object_detection")
        self.assertTrue(Path(manifest["manifest_path"]).exists())
        self.assertTrue(Path(manifest["outputs"]["geojson"]).exists())
        payload = json.loads(Path(manifest["outputs"]["geojson"]).read_text(encoding="utf-8"))
        self.assertGreater(len(payload["features"]), 0)
        first = payload["features"][0]
        self.assertEqual(first["geometry"]["type"], "Polygon")
        self.assertIn("bbox_pixel", first["properties"])

    def test_segmentation_change_and_super_resolution(self) -> None:
        seg = services.run_semantic_segmentation(str(self.before_path), str(self.root / "seg"), tile_size=40, overlap=8)
        self.assertTrue(Path(seg["outputs"]["mask_geotiff"]).exists())
        mask, _, mask_info = read_raster(seg["outputs"]["mask_geotiff"])
        self.assertEqual(mask.shape, (1, 72, 96))
        self.assertEqual(mask_info.width, 96)

        change = services.run_change_detection(str(self.before_path), str(self.after_path), str(self.root / "change"), tile_size=40, overlap=8, threshold=0.2)
        self.assertGreater(change["stats"]["changed_pixels"], 0)
        self.assertTrue(Path(change["outputs"]["probability_npy"]).exists())

        sr = services.run_super_resolution(str(self.before_path), str(self.root / "sr"), tile_size=40, overlap=8, scale=2)
        sr_array, _, sr_info = read_raster(sr["outputs"]["super_resolved_geotiff"])
        self.assertEqual(sr_array.shape, (4, 144, 192))
        self.assertEqual(sr_info.transform[0], sr["inputs"]["raster"]["transform"][0] / 2)

    def test_analysis_quality_and_report(self) -> None:
        det = services.run_object_detection(str(self.before_path), str(self.root / "det2"), tile_size=48, overlap=12)
        stats = services.calculate_statistics(input_path=det["outputs"]["geojson"], output_dir=str(self.root / "stats"))
        self.assertTrue(Path(stats["outputs"]["stats_json"]).exists())
        quality = services.quality_check_result(manifest_path=det["manifest_path"], output_dir=str(self.root / "quality"))
        self.assertEqual(quality["task"], "quality_check")
        report = services.generate_report(det["manifest_path"], output_dir=str(self.root / "report"))
        self.assertTrue(Path(report["outputs"]["report_md"]).exists())

    def test_spectral_indices(self) -> None:
        manifest = services.run_spectral_indices(str(self.before_path), str(self.root / "indices"), indices=["ndvi", "ndwi"], tile_size=32, overlap=8)
        self.assertIn("ndvi_geotiff", manifest["outputs"])
        self.assertIn("ndwi_geotiff", manifest["outputs"])
        self.assertEqual(manifest["parameters"]["tile_size"], 32)


if __name__ == "__main__":
    unittest.main()
