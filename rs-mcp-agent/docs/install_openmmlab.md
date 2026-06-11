# OpenMMLab 后端安装说明

本项目的 MMSegmentation、MMDetection、MMRotate 都是可选真实模型后端。基础 FastAPI、MCP stdio server 和 fake pipelines 不依赖这些重型包；只有在选择对应 `model_id` 时才会 lazy import。

## 推荐独立环境

建议单独创建 conda 环境，避免 OpenMMLab、PyTorch、CUDA 与基础 MCP 环境互相影响。

```bash
conda create -n rs-openmmlab python=3.10 -y
conda activate rs-openmmlab
pip install -e ".[dev]"
pip install -U openmim
mim install mmengine mmcv
pip install mmdet mmsegmentation mmrotate
```

如需 GPU，请先按 PyTorch 官方说明安装与你 CUDA 版本匹配的 `torch` / `torchvision`。

## 权重与配置路径

不要提交 `.pth` 权重文件。默认占位路径如下：

```text
external/mmdetection/configs/mask_rcnn/mask-rcnn_remote_sensing.py
weights/mmdet_maskrcnn_instance.pth

external/mmrotate/configs/oriented_rcnn/oriented-rcnn_dota.py
weights/mmrotate_dota_oriented.pth

external/mmsegmentation/configs/segformer/segformer_landcover.py
weights/mmseg_segformer_landcover.pth
```

可以在 `configs/models.yaml` 中改成绝对路径：

```yaml
- id: mmdet_maskrcnn_instance
  task: instance_segmentation
  backend: mmdet
  framework: mmdetection
  config: D:/models/mmdet/mask_rcnn.py
  checkpoint: D:/models/mmdet/mask_rcnn.pth
  device: cpu
  classes: [background, target]

- id: mmrotate_dota_oriented
  task: oriented_detection
  backend: mmrotate
  framework: mmrotate
  config: D:/models/mmrotate/oriented_rcnn.py
  checkpoint: D:/models/mmrotate/oriented_rcnn.pth
  device: cpu
  angle_unit: auto
  classes: [plane, ship, storage-tank]
```

## 调用示例

实例分割：

```bash
curl -X POST http://127.0.0.1:8765/jobs/instance-segmentation \
  -H "Content-Type: application/json" \
  -d "{\"image_path\":\"workspace/inputs/example.tif\",\"model_id\":\"mmdet_maskrcnn_instance\",\"tile_size\":1024,\"overlap\":192,\"score_threshold\":0.3}"
```

旋转框检测：

```bash
curl -X POST http://127.0.0.1:8765/jobs/oriented-detection \
  -H "Content-Type: application/json" \
  -d "{\"image_path\":\"workspace/inputs/example.tif\",\"model_id\":\"mmrotate_dota_oriented\",\"tile_size\":1024,\"overlap\":192,\"score_threshold\":0.3}"
```

MCP 中建议先调用 `inspect_raster` 和 `preflight_plan`，再调用 `run_instance_segmentation` 或 `run_oriented_detection`。结论仍应来自 `manifest.json`、`stats.json` 和 `quality_flags`。

## 常见问题

- `No module named mmdet/mmrotate/mmseg`：当前环境没有安装对应 OpenMMLab 包，请在独立环境中安装。
- `config not found`：检查 `configs/models.yaml` 中的 config 路径。
- `checkpoint not found`：将 `.pth` 放到配置的路径，或改成绝对路径。
- CUDA/PyTorch 报错：先验证 `python -c "import torch; print(torch.cuda.is_available())"`，并确认 mmcv 与 PyTorch/CUDA 版本匹配。
