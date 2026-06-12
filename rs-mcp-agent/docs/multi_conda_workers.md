# Multi-Conda Subprocess Workers

`rs-mcp-agent` supports running heavyweight real model backends in separate conda environments while keeping the MCP/FastAPI base service lightweight. The base process still owns GeoTIFF IO, tiling, stitching, coordinate recovery, statistics, reports, and `manifest.json`; subprocess workers only infer one tile at a time.

## Environment Roles

| Environment | Responsibility |
| --- | --- |
| `rs-mcp-base` | MCP Server, FastAPI, GIS processing, large-image tiling, stitching, statistics, reports, manifest output |
| `rs-mcp-yolo` | YOLO / SAHI object detection and YOLO instance segmentation |
| `rs-mcp-openmmlab` | MMSegmentation, MMDetection, MMRotate |
| `rs-mcp-opencd` | Open-CD bi-temporal change detection |
| `rs-mcp-sr` | SwinIR / BasicSR super-resolution |

## How Runner Dispatch Works

`configs/models.yaml` controls the runner:

```yaml
- id: yolo_detection_subprocess
  task: object_detection
  backend: ultralytics
  runner: subprocess
  conda_env: rs-mcp-yolo
  entrypoint: rs_service.workers.yolo_infer
  runner_timeout_sec: 600
  weights: weights/yolo_detection.pt
```

When `runner: subprocess` is selected, `rs_service.registry.get_adapter()` returns `ExternalSubprocessAdapter`. The adapter writes tile arrays to temporary `.npy` files and runs:

```bash
conda run -n rs-mcp-yolo python -m rs_service.workers.yolo_infer --request request.json --response response.json
```

If `conda_env: null`, the worker runs with the current Python. This is how `fake_external_*` tests validate the subprocess protocol without conda, GPU, real weights, or deep learning frameworks.

## Model ID To Environment Mapping

| model_id | conda_env | entrypoint |
| --- | --- | --- |
| `fake_external_detection` | current Python | `rs_service.workers.fake_external_infer` |
| `fake_external_segmentation` | current Python | `rs_service.workers.fake_external_infer` |
| `fake_external_instance` | current Python | `rs_service.workers.fake_external_infer` |
| `fake_external_change` | current Python | `rs_service.workers.fake_external_infer` |
| `fake_external_super_resolution` | current Python | `rs_service.workers.fake_external_infer` |
| `yolo_detection_subprocess` | `rs-mcp-yolo` | `rs_service.workers.yolo_infer` |
| `sahi_yolo_detection_subprocess` | `rs-mcp-yolo` | `rs_service.workers.sahi_yolo_infer` |
| `yolo_instance_segmentation_subprocess` | `rs-mcp-yolo` | `rs_service.workers.yolo_infer` |
| `mmseg_segformer_landcover_subprocess` | `rs-mcp-openmmlab` | `rs_service.workers.mmseg_infer` |
| `mmdet_maskrcnn_instance_subprocess` | `rs-mcp-openmmlab` | `rs_service.workers.mmdet_infer` |
| `mmrotate_dota_oriented_subprocess` | `rs-mcp-openmmlab` | `rs_service.workers.mmrotate_infer` |
| `opencd_changer_building_subprocess` | `rs-mcp-opencd` | `rs_service.workers.opencd_infer` |
| `swinir_x4_subprocess` | `rs-mcp-sr` | `rs_service.workers.swinir_infer` |

## Fake External Test Mode

Run:

```bash
python scripts/smoke_test_subprocess.py
```

This checks direct adapter calls and tiled pipelines using only `fake_external_*` model IDs. It does not require conda, GPU, YOLO, OpenMMLab, Open-CD, SwinIR, BasicSR, or real weights.

To diagnose YOLO subprocess routing on a deployment host, run:

```bash
python scripts/diagnose_yolo_subprocess.py
```

The output includes the current working directory, `RS_MODELS_CONFIG`, the resolved `models.yaml`, duplicate `yolo_detection_subprocess` entries, the resolved model config, the adapter type, and the `/models` entry fields. `adapter_type` should be `rs_service.adapters.external_subprocess_adapter.ExternalSubprocessAdapter`.

## Real Weights And Configs

Weights are not committed. Use one of these paths and update `configs/models.yaml` if needed:

- `weights/yolo_detection.pt`
- `weights/yolo_instance.pt`
- `weights/mmseg_segformer_landcover.pth`
- `weights/mmdet_maskrcnn_instance.pth`
- `weights/mmrotate_dota_oriented.pth`
- `weights/opencd_changer_building.pth`
- `weights/swinir_x4.pth`

OpenMMLab and Open-CD config paths are placeholders under `external/...`. Install or clone the corresponding upstream projects and point each model config to the real config file.

Each worker environment should be able to import this project. The simplest setup is to run this from the project root in each conda env:

```bash
pip install -e .
```

## Troubleshooting

- `conda` does not exist: install conda/mamba or set `conda_env: null` only for test workers.
- Environment does not exist: create the named environment or update `conda_env`.
- `ModuleNotFoundError`: install the required framework in that worker env and run `pip install -e .`.
- Weight path does not exist: place the file in `weights/` or update `weights` / `checkpoint` in `configs/models.yaml`.
- Config path does not exist: update `config` to the real OpenMMLab/Open-CD config path.
- `response.json` was not generated: inspect the captured worker stderr in the API/job error message.
- Subprocess timeout: increase `runner_timeout_sec` or reduce `tile_size`.
- MCP stdio output is corrupted: keep ordinary logs off stdout. MCP stdout is reserved for the protocol.
