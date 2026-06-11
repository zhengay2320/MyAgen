from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from rs_service.core.raster import DEFAULT_TRANSFORM, profile_from_array, read_raster, write_raster
from rs_service.pipelines.spectral_indices import run_spectral_indices


class SpectralIndicesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_synthetic_four_band_ndvi(self) -> None:
        """NDVI should use default blue/green/red/nir mapping for 4-band rasters."""
        array = np.zeros((4, 8, 10), dtype=np.uint16)
        array[2] = 20
        array[3] = 60
        path = self.root / "four_band.tif"
        write_raster(path, array, profile_from_array(array, transform=DEFAULT_TRANSFORM))

        manifest = run_spectral_indices(path, self.root / "indices", indices=["ndvi"], thresholds={"ndvi": {">0.3": ">0.3"}})
        ndvi, _, _ = read_raster(manifest["outputs"]["ndvi_geotiff"])

        self.assertAlmostEqual(float(ndvi[0, 0, 0]), 0.5, places=5)
        self.assertIn("ndvi_preview", manifest["outputs"])
        self.assertTrue(Path(manifest["outputs"]["ndvi_preview"]).exists())
        self.assertTrue(Path(manifest["outputs"]["stats_json"]).exists())
        self.assertTrue(Path(manifest["outputs"]["report_md"]).exists())
        self.assertEqual(manifest["statistics"]["indices"]["ndvi"]["threshold_areas"][">0.3"]["pixel_count"], 80)

    def test_nodata_and_divide_by_zero_are_handled(self) -> None:
        """Nodata pixels and zero denominators should not crash index generation."""
        array = np.zeros((4, 6, 6), dtype=np.float32)
        array[2] = 0.0
        array[3] = 0.0
        array[:, 0, 0] = -9999.0
        path = self.root / "nodata.tif"
        profile = profile_from_array(array, nodata=-9999.0, transform=DEFAULT_TRANSFORM)
        write_raster(path, array, profile, nodata=-9999.0)

        manifest = run_spectral_indices(path, self.root / "nodata_indices", indices=["ndvi"])
        ndvi, _, _ = read_raster(manifest["outputs"]["ndvi_geotiff"])

        self.assertTrue(np.isnan(ndvi[0, 0, 0]) or float(ndvi[0, 0, 0]) == 0.0)
        codes = {item["code"] for item in manifest["quality_flags"]}
        self.assertIn("division_by_zero_handled", codes)

    def test_missing_band_map_for_swir_index_is_clear_error(self) -> None:
        """A 4-band raster cannot compute SWIR-dependent indices without band_map."""
        array = np.zeros((4, 8, 8), dtype=np.uint8)
        path = self.root / "missing_swir.tif"
        write_raster(path, array, profile_from_array(array))

        with self.assertRaises(ValueError) as raised:
            run_spectral_indices(path, self.root / "bad", indices=["mndwi"])
        self.assertIn("required bands are unavailable", str(raised.exception))
        self.assertIn("swir1", str(raised.exception))

    def test_explicit_band_map_supports_savi_evi_mndwi_ndbi(self) -> None:
        """Explicit six-band mapping should enable all supported indices."""
        array = np.zeros((6, 5, 7), dtype=np.float32)
        array[0] = 10
        array[1] = 20
        array[2] = 30
        array[3] = 70
        array[4] = 40
        array[5] = 50
        path = self.root / "six_band.tif"
        write_raster(path, array, profile_from_array(array))
        manifest = run_spectral_indices(
            path,
            self.root / "all_indices",
            indices=["ndvi", "ndwi", "mndwi", "ndbi", "savi", "evi"],
            band_map={"blue": 1, "green": 2, "red": 3, "nir": 4, "swir1": 5, "swir2": 6},
        )

        self.assertEqual(set(manifest["statistics"]["computed_indices"]), {"ndvi", "ndwi", "mndwi", "ndbi", "savi", "evi"})
        for name in ["ndvi", "ndwi", "mndwi", "ndbi", "savi", "evi"]:
            self.assertTrue(Path(manifest["outputs"][f"{name}_geotiff"]).exists())


if __name__ == "__main__":
    unittest.main()
