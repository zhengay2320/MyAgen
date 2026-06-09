# rs_service API

Run the API:

```bash
uvicorn rs_service.api:app --host 127.0.0.1 --port 8765
```

or:

```bash
python scripts/run_api.py --host 127.0.0.1 --port 8765
rs-service api --host 127.0.0.1 --port 8765
```

## Health And Models

```bash
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/models
```

## Inspect Raster

```bash
curl -X POST http://127.0.0.1:8765/inspect \
  -H "Content-Type: application/json" \
  -d "{\"path\":\"workspace/synthetic.tif\"}"
```

## Preflight Tiling Plan

```bash
curl -X POST http://127.0.0.1:8765/preflight \
  -H "Content-Type: application/json" \
  -d "{\"image_path\":\"workspace/synthetic.tif\",\"task\":\"semantic_segmentation\"}"
```

## Submit Jobs

All MVP jobs execute synchronously, but responses include a stable `job_id`. Job files are persisted under `workspace/jobs/{job_id}.json`; outputs and manifests are written under `workspace/outputs/{job_id}/`.

Semantic segmentation:

```bash
curl -X POST http://127.0.0.1:8765/jobs/semantic-segmentation \
  -H "Content-Type: application/json" \
  -d "{\"image_path\":\"workspace/synthetic.tif\",\"tile_size\":512,\"overlap\":64}"
```

Object detection:

```bash
curl -X POST http://127.0.0.1:8765/jobs/detection \
  -H "Content-Type: application/json" \
  -d "{\"image_path\":\"workspace/synthetic.tif\",\"tile_size\":512,\"overlap\":64,\"score_threshold\":0.2}"
```

Change detection:

```bash
curl -X POST http://127.0.0.1:8765/jobs/change-detection \
  -H "Content-Type: application/json" \
  -d "{\"before_path\":\"workspace/before.tif\",\"after_path\":\"workspace/after.tif\",\"tile_size\":512,\"overlap\":64}"
```

Other job endpoints:

- `POST /jobs/oriented-detection`
- `POST /jobs/instance-segmentation`
- `POST /jobs/super-resolution`
- `POST /jobs/spectral-indices`

## Query Jobs And Manifests

```bash
curl http://127.0.0.1:8765/jobs/{job_id}
curl http://127.0.0.1:8765/jobs/{job_id}/manifest
```

`GET /jobs/{job_id}/manifest` returns the `manifest.json` content directly.

## Analyze And Report Existing Jobs

```bash
curl -X POST http://127.0.0.1:8765/jobs/{job_id}/analyze \
  -H "Content-Type: application/json" \
  -d "{}"

curl -X POST http://127.0.0.1:8765/jobs/{job_id}/report \
  -H "Content-Type: application/json" \
  -d "{\"title\":\"Remote Sensing Processing Report\"}"
```

## Status Lifecycle

The local job store records:

- `queued`
- `running`
- `success`
- `failed`

If a pipeline fails, the job JSON records `errors`. Real model frameworks are not imported at API startup; fake adapters are used for MVP execution.
