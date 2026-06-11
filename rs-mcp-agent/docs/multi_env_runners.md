# 多 Conda 环境真实模型运行器

`rs-mcp-agent` 的主服务环境只负责 MCP、FastAPI、GIS IO、大图切片、结果拼接、统计、报告和 `manifest.json`。真实深度学习框架可以放在独立 conda 环境中，通过 `runner: subprocess` 的模型配置运行单个 tile 推理。

## 环境分工

| 环境 | 用途 |
| --- | --- |
| `rs-mcp-base` | MCP Server、FastAPI、Rasterio/GeoPandas、切片、拼接、统计、报告、manifest |
| `rs-mcp-yolo` | Ultralytics YOLO、SAHI，目标检测和 YOLO 实例分割 |
| `rs-mcp-openmmlab` | MMSegmentation、MMDetection、MMRotate |
| `rs-mcp-opencd` | Open-CD 双时相变化检测 |
| `rs-mcp-sr` | SwinIR、BasicSR 超分辨率 |

## 模型配置

`configs/models.yaml` 中保留原有 in-process 模型，同时新增 `_subprocess` 后缀模型。关键字段：

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

运行逻辑：

1. Pipeline 在 `rs-mcp-base` 中读取 GeoTIFF 并生成 tile。
2. `rs_service.registry.get_adapter()` 读取 `configs/models.yaml`。
3. `runner: subprocess` 返回 `ExternalSubprocessAdapter`。
4. 主进程把当前 tile 写为临时 `.npy`，并执行：

```bash
conda run -n rs-mcp-yolo python -m rs_service.workers.yolo_infer --request request.json --response response.json
```

5. Worker 只做单 tile 推理并写回 response。
6. 主进程继续做拼接、坐标恢复、统计、报告和 manifest。

如果 `conda_env` 为空，runner 会使用当前 Python 运行 worker，便于本地测试：

```yaml
conda_env: null
entrypoint: rs_service.workers.fake_external_infer
```

## 已注册的 Subprocess 模型

| model_id | 环境 | worker |
| --- | --- | --- |
| `yolo_detection_subprocess` | `rs-mcp-yolo` | `rs_service.workers.yolo_infer` |
| `sahi_yolo_detection_subprocess` | `rs-mcp-yolo` | `rs_service.workers.sahi_yolo_infer` |
| `yolo_instance_segmentation_subprocess` | `rs-mcp-yolo` | `rs_service.workers.yolo_infer` |
| `mmseg_segformer_landcover_subprocess` | `rs-mcp-openmmlab` | `rs_service.workers.mmseg_infer` |
| `mmdet_maskrcnn_instance_subprocess` | `rs-mcp-openmmlab` | `rs_service.workers.mmdet_infer` |
| `mmrotate_dota_oriented_subprocess` | `rs-mcp-openmmlab` | `rs_service.workers.mmrotate_infer` |
| `opencd_changer_building_subprocess` | `rs-mcp-opencd` | `rs_service.workers.opencd_infer` |
| `swinir_x4_subprocess` | `rs-mcp-sr` | `rs_service.workers.swinir_infer` |

## 使用示例

启动主服务：

```bash
python scripts/run_api.py --host 127.0.0.1 --port 8765
```

通过 API 调用 subprocess YOLO：

```bash
curl -X POST http://127.0.0.1:8765/jobs/detection \
  -H "Content-Type: application/json" \
  -d '{"image_path":"workspace/inputs/synthetic.tif","model_id":"yolo_detection_subprocess","tile_size":1024,"overlap":192}'
```

通过 MCP 调用时工具名不变，只需传入 subprocess 模型 ID：

```text
请对 workspace/inputs/synthetic.tif 做目标检测，model_id 使用 yolo_detection_subprocess。
```

## 常见问题

- `conda` 命令不存在：安装 conda/mamba，或将测试模型的 `conda_env` 设为空。
- Worker 无法 import `rs_service`：在对应 conda 环境中执行 `pip install -e .`，或确认 runner 已通过 `PYTHONPATH` 指向项目根目录。
- 权重路径不存在：下载权重到 `weights/` 或修改 `configs/models.yaml` 的 `weights` / `checkpoint`。
- 缺少真实框架：在对应环境安装 `ultralytics/sahi`、OpenMMLab、Open-CD 或 SwinIR/BasicSR。
- Worker 超时：调大 `runner_timeout_sec`，或减小 `tile_size`。
- MCP stdio 被污染：不要在 `rs_mcp.server` 向 stdout 打日志；worker stdout 会被主进程捕获，不参与 MCP 协议。
