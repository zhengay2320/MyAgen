# rs-mcp-agent Audit Report

Audit date: 2026-06-09

## Current Structure

```text
rs-mcp-agent/
  pyproject.toml
  Makefile
  README.md
  configs/
    models.yaml
    service.yaml
  docs/
    cherry_studio_mcp.md
    codex_mcp.md
    model_weights.md
    audit_report.md
  rs_service/
    api.py
    services.py
    schemas.py
    jobs.py
    registry.py
    adapters/
    core/
    pipelines/
  rs_mcp/
    server.py
    tools.py
  scripts/
    create_synthetic_geotiff.py
    download_weights.py
    smoke_test.py
  tests/
    test_core.py
    test_mcp_tools.py
    test_pipelines.py
  workspace/
```

The project is organized as a standalone Python package under `rs-mcp-agent`. The service layer, MCP layer, fake adapters, core raster/vector utilities, task pipelines, scripts, tests, and docs are separated clearly enough for a first-stage engineering baseline.

## Implemented Modules

- Project metadata: `pyproject.toml` defines package metadata, runtime dependencies, optional ML dependencies, dev dependencies, console scripts, and pytest discovery settings.
- Make targets: `test`, `unittest`, `smoke`, `synthetic`, `api`, `mcp`, and `clean` are present.
- Raster core: CHW raster normalization, synthetic raster generation, raster inspect, fallback raster container, pixel-to-geo coordinate conversion, bounds calculation, and super-resolution transform scaling.
- Tiling core: tiled window iteration with `tile_size` and `overlap`, accumulator-based prediction blending, and stitched output finalization.
- Vector core: GeoJSON writing and GPKG writing when GeoPandas/Fiona are installed, with a fallback text payload when they are absent.
- Manifest core: every pipeline writes a `manifest.json` with shared top-level fields including `schema_version`, `job_id`, `task`, `status`, `inputs`, `outputs`, `parameters`, `model`, `stats`, and `quality_flags`.
- Fake adapters: deterministic fake backends exist for object detection, oriented detection, semantic segmentation, instance segmentation, change detection, and super resolution.
- Pipelines:
  - Object detection: tiled fake inference, tile-to-full pixel coordinate restoration, geo-vector export, basic NMS, manifest.
  - Oriented detection: tiled fake rotated box inference, pixel polygon restoration, geo-vector export, manifest.
  - Instance segmentation: tiled fake polygon inference, geo-vector export, manifest.
  - Semantic segmentation: probability stitching, mask GeoTIFF output, probability `.npy`, manifest.
  - Change detection: bi-temporal tile inference, probability stitching, mask output, manifest.
  - Super resolution: tiled upscaling, stitched output, transform scaling, manifest.
  - Spectral indices: NDVI, NDWI, NDBI, and EVI calculation with configurable band mapping, manifest.
  - Statistics: raster or GeoJSON summary, `stats.json`, manifest.
  - Quality check: output existence checks, CRS/fallback checks, `quality.json`, manifest.
  - Report: Markdown `report.md` from a source manifest.
- FastAPI backend: app factory and routes exist for health, models, jobs, inspect, preflight, every run task, statistics, quality, report, and manifest retrieval.
- MCP server: all required tools are registered in `rs_mcp.tools`; `rs_mcp.server` supports official `FastMCP` when installed and a lightweight JSON-RPC stdio fallback.
- Scripts: synthetic raster generation, smoke test, and model weight path preparation exist. No large weights are downloaded.
- Tests: unit tests cover raster/tiling basics, pipeline smoke paths, and MCP tool registration.

## Missing Modules And Gaps

- Real model adapters are not implemented. Ultralytics + SAHI, MMSegmentation, MMDetection, MMRotate, Open-CD, BasicSR/MMagic/SwinIR, SAMGeo, and GroundingDINO are represented as intended frameworks or optional dependencies only.
- `configs/models.yaml` and `configs/service.yaml` are not loaded by runtime code yet; model registry and service defaults are currently hard-coded.
- The fallback `.tif` writer is a NumPy archive with a `.tif` extension, not a standards-compliant GeoTIFF. Real GeoTIFF output requires Rasterio/GDAL.
- The fallback `.gpkg` writer is a JSON payload with a `.gpkg` extension, not a standards-compliant GeoPackage. Real GPKG output requires GeoPandas/Fiona.
- Job execution is synchronous and in-process. There is no queue, cancellation, progress reporting, durable job database, background worker, or multi-process safety.
- Large-raster IO is array-loaded into memory. Production-scale rasters need windowed Rasterio reads/writes and block-aware processing.
- Semantic segmentation and change detection stitch probability maps, but do not yet support model-specific logits calibration, class metadata, nodata-aware blending, or vectorized polygon extraction from masks.
- Object/instance detection uses simple axis-aligned NMS. There is no SAHI prediction object integration, rotated NMS, mask NMS, class-wise thresholds, or geospatial duplicate suppression.
- Change detection requires same-shaped rasters and does not reproject/resample/co-register inputs.
- Spectral indices accept `tile_size` and `overlap` for interface parity but currently compute full-raster arrays rather than true tiled/windowed processing.
- Statistics support raster summaries and basic GeoJSON summaries, but zonal statistics are only flagged as a future dependency-path task.
- Quality checks are rule-based and shallow. They do not yet validate CRS consistency across products, geometry validity, probability ranges per class, nodata leakage, transform alignment, or output schema completeness.
- Report generation is Markdown-only and source-manifest driven. It does not embed maps, thumbnails, charts, or richer report artifacts.
- MCP fallback supports enough JSON-RPC for tests, but production use should prefer the official `mcp` package.
- API tests are absent because FastAPI/httpx are optional dev/runtime dependencies in the current local environment.

## Risks

- Dependency risk: geospatial Python stacks can be platform-sensitive. Rasterio/GDAL/Fiona/GeoPandas installation should be validated with a pinned environment file before production use.
- Output validity risk: in environments without Rasterio/GeoPandas, outputs with `.tif` and `.gpkg` extensions are fallback containers and not GIS-standard files.
- Memory risk: current full-array reads and stitched accumulators can exceed memory on true large scenes.
- Schema drift risk: manifests are consistent at top level, but task-specific outputs and stats are not formally versioned with a JSON Schema.
- Coordinate risk: pixel-to-geo conversion is implemented, but there are no tests against known CRS/transform fixtures from real GeoTIFFs.
- Runtime risk: `workspace` is mostly relative to the process current directory. Running the API from a different directory may change job discovery behavior.
- Model integration risk: fake adapters keep interfaces clean, but real MM*/Ultralytics/Open-CD adapters will need careful tensor format, device, class metadata, and checkpoint config handling.
- Quality risk: smoke tests validate happy paths only; they do not cover invalid rasters, mismatched CRS, huge rasters, missing bands, empty masks, corrupt manifests, or failed jobs.

## Small Fixes Applied During Audit

- `Makefile` `test` now runs `python -m pytest -q`, matching the requested validation command. A separate `unittest` target preserves the lightweight unittest fallback.
- `rs_service.api` now uses a `_payload_dict()` helper so routes work with both Pydantic v2 `model_dump()` and Pydantic v1 `dict()`.

## Validation Results

Commands run with the bundled Python executable because the current shell PATH does not provide `python` or `pytest`.

```text
python -m compileall rs_service rs_mcp
PASS

python -c "import rs_service, rs_service.services, rs_mcp.server"
PASS

python -m unittest discover -s tests -p "test_*.py"
PASS: 8 tests

python scripts/smoke_test.py
PASS: detection, oriented detection, semantic segmentation, instance segmentation,
change detection, super resolution, spectral indices, statistics, quality, and report manifests generated.
```

`python -m pytest -q` could not be executed in this local bundled environment because `pytest` is not installed there:

```text
No module named pytest
```

The project declares `pytest>=8` under `.[dev]`; after `pip install -e ".[dev]"`, the existing unittest-style tests should be collected by pytest.

## Recommended Next Stage

1. Create a pinned environment setup for GIS/runtime dependencies and validate real Rasterio GeoTIFF and GeoPandas GPKG outputs.
2. Load `configs/service.yaml` and `configs/models.yaml` in the registry/service startup path instead of hard-coding defaults.
3. Add JSON Schema definitions for manifest and task-specific outputs; validate manifests in tests.
4. Replace fallback full-array raster processing with Rasterio windowed reads/writes for large images.
5. Implement the first real adapter path, preferably Ultralytics + SAHI object detection, because it exercises tiling, coordinate restoration, vector export, and model metadata end to end.
6. Add FastAPI tests with `httpx` and official MCP server tests with the `mcp` package installed.
7. Expand quality checks for CRS mismatch, geometry validity, nodata handling, probability ranges, transform alignment, and empty/oversized outputs.
8. Add asynchronous job execution, progress tracking, cancellation, and persistent job metadata before exposing long-running inference through API or MCP clients.
