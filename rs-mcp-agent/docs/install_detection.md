# 目标检测与实例分割后端安装说明

本项目默认使用 fake adapters，不需要 GPU、权重或重型深度学习依赖。真实目标检测和实例分割后端为可选能力，只有当 `model_id` 选择 `yolo_detection`、`sahi_yolo_detection` 或 `yolo_instance_segmentation` 时才会 lazy import。

## 安装依赖

```bash
pip install ultralytics sahi
```

或使用项目 optional extra：

```bash
pip install -e ".[detection]"
```

如需 GPU，请按 PyTorch 官方说明安装与你 CUDA 版本匹配的 `torch` / `torchvision`。基础服务启动不依赖 GPU。

## 权重路径约定

不要将大模型权重提交到仓库。默认配置使用以下占位路径：

```text
weights/yolo_detection.pt
weights/yolo_instance.pt
```

也可以在 `configs/models.yaml` 中改为绝对路径或你的工作区模型目录，例如：

```yaml
- id: yolo_detection
  task: object_detection
  backend: ultralytics
  framework: ultralytics
  weights: D:/models/yolo_detection.pt
  confidence_threshold: 0.25
  iou_threshold: 0.45
  device: cpu
  imgsz: 1024

- id: sahi_yolo_detection
  task: object_detection
  backend: sahi
  framework: sahi+ultralytics
  weights: D:/models/yolo_detection.pt
  confidence_threshold: 0.25
  device: cpu
  slice_height: 512
  slice_width: 512
  overlap_ratio: 0.2

- id: yolo_instance_segmentation
  task: instance_segmentation
  backend: ultralytics
  framework: ultralytics
  weights: D:/models/yolo_instance.pt
  confidence_threshold: 0.25
  iou_threshold: 0.45
  device: cpu
  imgsz: 1024
```

## 运行示例

启动 FastAPI：

```bash
python scripts/run_api.py --host 127.0.0.1 --port 8765
```

目标检测：

```bash
curl -X POST http://127.0.0.1:8765/jobs/detection \
  -H "Content-Type: application/json" \
  -d "{\"image_path\":\"workspace/inputs/example.tif\",\"model_id\":\"yolo_detection\",\"tile_size\":1024,\"overlap\":192,\"score_threshold\":0.25}"
```

SAHI 大图小目标检测：

```bash
curl -X POST http://127.0.0.1:8765/jobs/detection \
  -H "Content-Type: application/json" \
  -d "{\"image_path\":\"workspace/inputs/example.tif\",\"model_id\":\"sahi_yolo_detection\",\"tile_size\":1024,\"overlap\":192,\"score_threshold\":0.25}"
```

实例分割：

```bash
curl -X POST http://127.0.0.1:8765/jobs/instance-segmentation \
  -H "Content-Type: application/json" \
  -d "{\"image_path\":\"workspace/inputs/example.tif\",\"model_id\":\"yolo_instance_segmentation\",\"tile_size\":1024,\"overlap\":128,\"score_threshold\":0.25}"
```

如果依赖或权重缺失，API/MCP 会返回清晰错误；fake pipeline、API 启动和 MCP stdio server 不受影响。
