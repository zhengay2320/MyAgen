# Project Structure

`rs-mcp-agent` is organized around one shared service core used by both FastAPI and MCP.

## Top Level

- `pyproject.toml`: package metadata, base dependencies, optional model extras, console scripts, and pytest configuration.
- `Makefile`: install, test, basic compile lint, API, MCP, synthetic data, and smoke commands.
- `configs/`: model registry, task defaults, analysis rules, and class maps.
- `workspace/`: local runtime data. The service creates `inputs`, `outputs`, `previews`, `reports`, `jobs`, and `cache`.

## Service Package

- `rs_service/settings.py`: reads `RS_WORKSPACE`, `RS_MODELS_CONFIG`, `RS_ANALYSIS_RULES`, `RS_SERVICE_HOST`, and `RS_SERVICE_PORT`; creates workspace folders.
- `rs_service/schemas.py`: shared Pydantic request, response, job, and manifest schemas.
- `rs_service/job_store.py`: local JSON-backed job store at `workspace/jobs/{job_id}.json`.
- `rs_service/services.py`: facade used by API and MCP tools.
- `rs_service/api.py`: FastAPI app and route bindings.
- `rs_service/core/`: raster IO, tiling, geometry, vector writing, and manifest helpers.
- `rs_service/adapters/`: model adapter interfaces and fake adapters. Heavy frameworks must be lazy imported in concrete adapters.
- `rs_service/pipelines/`: task implementations. Each pipeline writes `workspace/outputs/{job_id}/manifest.json` when launched through the service layer.

## MCP Package

- `rs_mcp/tools.py`: plain Python tool functions mapped to service calls.
- `rs_mcp/server.py`: official FastMCP server when the `mcp` package is installed, plus a lightweight JSON-RPC stdio fallback for smoke tests.

## Runtime Flow

1. Client calls FastAPI or MCP.
2. `rs_service.services` creates a job in `workspace/jobs`.
3. The selected pipeline runs with `tile_size` and `overlap`.
4. Outputs are written to `workspace/outputs/{job_id}`.
5. `manifest.json` records canonical fields: `job_id`, `task`, `status`, `model_id`, `input_files`, `parameters`, `outputs`, `statistics`, `metrics`, `quality_flags`, `conclusion`, and `errors`.
6. The job store records success or failure and points to the manifest.

## Production Notes

- Base startup must not require GPU, model weights, or heavyweight model imports.
- Real model adapters should lazy import frameworks inside adapter constructors or prediction methods.
- Large rasters should move toward Rasterio windowed IO rather than full-array loading.
- GIS-standard GeoTIFF/GPKG output requires Rasterio/GDAL and GeoPandas/Fiona in the runtime environment.
