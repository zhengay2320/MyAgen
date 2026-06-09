# rs-mcp-agent

`rs-mcp-agent` is a first-stage Python project for remote sensing image processing as both:

- `rs_service`: a FastAPI backend for inference, analysis, quality checks, and reports.
- `rs_mcp.server`: a stdio MCP server for Codex, Cherry Studio, and other MCP clients.

The first stage ships fake adapters and complete tests. Large model weights are not downloaded or committed. Real backends are intended to be plugged in behind the adapter interfaces for Ultralytics + SAHI, MMSegmentation, MMDetection, MMRotate, Open-CD, BasicSR/MMagic/SwinIR, SAMGeo, and GroundingDINO.

## Quick Start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
make test
make synthetic
make smoke
```

Run the FastAPI service:

```bash
uvicorn rs_service.api:app --host 127.0.0.1 --port 8765
```

Run the MCP server:

```bash
python -m rs_mcp.server
```

## Outputs

Every pipeline writes a `manifest.json`. Geospatial outputs use GeoTIFF plus GeoJSON/GPKG where appropriate:

- Detection, oriented detection, and instance segmentation: restored full-image pixel coordinates and geospatial vector outputs.
- Semantic segmentation and change detection: tiled probability or mask stitching.
- Super resolution: updated output GeoTIFF transform.
- Analysis: `stats.json`, `quality_flags`, and `report.md`.

When `rasterio`/`geopandas` are unavailable, tests use a lightweight local fallback container so the fake pipeline can still run. In production, install the declared geospatial dependencies to emit standards-compliant GeoTIFF/GPKG files.
