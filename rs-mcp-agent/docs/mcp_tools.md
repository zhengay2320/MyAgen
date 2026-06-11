# MCP Tools

The MCP server is a stdio bridge to the FastAPI backend. It does not run models directly. Configure the backend URL with `RS_SERVICE_URL`; the default is `http://127.0.0.1:8765`.

## Instructions

- Inspect rasters before running inference.
- Call `preflight_plan` before `run_*`.
- Large rasters must use explicit `tile_size` and `overlap`.
- Conclusions must come from `manifest`, `statistics`, `metrics`, or `quality_flags`.
- Return output paths exactly as provided by the backend.

## Tools

### inspect_raster

Parameters:

```json
{"image_path": "workspace/synthetic.tif"}
```

Returns raster metadata from `POST /inspect`.

### preflight_plan

Parameters:

```json
{"image_path": "workspace/synthetic.tif", "task": "semantic_segmentation", "model_id": "fake_semantic_segmentation"}
```

Returns a tiling plan from `POST /preflight`.

### list_models

Parameters:

```json
{}
```

Returns available backend models from `GET /models`.

### run_object_detection

Parameters:

```json
{
  "image_path": "workspace/synthetic.tif",
  "model_id": "fake_detection",
  "tile_size": 512,
  "overlap": 64,
  "confidence_threshold": 0.25
}
```

Returns a synchronous job response from `POST /jobs/detection`.

### run_oriented_detection

Parameters:

```json
{
  "image_path": "workspace/synthetic.tif",
  "model_id": "fake_oriented_detection",
  "tile_size": 512,
  "overlap": 64,
  "confidence_threshold": 0.25
}
```

Returns a synchronous job response from `POST /jobs/oriented-detection`.

### run_semantic_segmentation

Parameters:

```json
{
  "image_path": "workspace/synthetic.tif",
  "model_id": "fake_semantic_segmentation",
  "tile_size": 512,
  "overlap": 64
}
```

Returns a synchronous job response from `POST /jobs/semantic-segmentation`.

### run_instance_segmentation

Parameters:

```json
{
  "image_path": "workspace/synthetic.tif",
  "model_id": "fake_instance_segmentation",
  "tile_size": 512,
  "overlap": 64
}
```

Returns a synchronous job response from `POST /jobs/instance-segmentation`.

### run_change_detection

Parameters:

```json
{
  "image_t1_path": "workspace/before.tif",
  "image_t2_path": "workspace/after.tif",
  "model_id": "fake_change_detection",
  "tile_size": 512,
  "overlap": 64
}
```

Returns a synchronous job response from `POST /jobs/change-detection`.

### run_super_resolution

Parameters:

```json
{
  "image_path": "workspace/synthetic.tif",
  "model_id": "fake_super_resolution",
  "scale": 2,
  "tile_size": 256,
  "overlap": 32
}
```

Returns a synchronous job response from `POST /jobs/super-resolution`.

### run_spectral_indices

Parameters:

```json
{"image_path": "workspace/synthetic.tif", "indices": ["ndvi"]}
```

Returns a synchronous job response from `POST /jobs/spectral-indices`.

### calculate_statistics

Parameters:

```json
{"job_id": "semantic_segmentation_20260609_000000_abcd1234"}
```

Runs `POST /jobs/{job_id}/analyze`.

### quality_check_result

Parameters:

```json
{"job_id": "semantic_segmentation_20260609_000000_abcd1234"}
```

Fetches the job manifest and calls `POST /quality`.

### generate_report

Parameters:

```json
{"job_id": "semantic_segmentation_20260609_000000_abcd1234", "output_format": "markdown"}
```

Runs `POST /jobs/{job_id}/report`.

### get_job_status

Parameters:

```json
{"job_id": "semantic_segmentation_20260609_000000_abcd1234"}
```

Returns `GET /jobs/{job_id}`.

### get_result_manifest

Parameters:

```json
{"job_id": "semantic_segmentation_20260609_000000_abcd1234"}
```

Returns `GET /jobs/{job_id}/manifest`.
