# MMSegmentation 语义分割后端安装说明

MMSegmentation / OpenMMLab 依赖较重，建议为真实语义分割单独创建 conda 环境。`rs-mcp-agent` 的基础 FastAPI、MCP 和 fake pipeline 不依赖 MMSeg，只有选择 `backend=mmseg` 的模型时才会 lazy import。

## 推荐安装方式

```bash
conda create -n rs-mmseg python=3.10 -y
conda activate rs-mmseg
pip install -e ".[dev]"
pip install -U openmim
mim install mmengine mmcv
pip install mmsegmentation
```

如果需要 GPU，请先按 PyTorch 官方说明安装匹配 CUDA 的 `torch` / `torchvision`，再安装 OpenMMLab 组件。

## 配置 config 与 checkpoint

默认占位模型：

```yaml
- id: mmseg_segformer_landcover
  task: semantic_segmentation
  backend: mmseg
  framework: mmsegmentation
  config: external/mmsegmentation/configs/segformer/segformer_landcover.py
  checkpoint: weights/mmseg_segformer_landcover.pth
  device: cpu
  classes:
    - background
    - landcover

- id: mmseg_deeplab_building
  task: semantic_segmentation
  backend: mmseg
  framework: mmsegmentation
  config: external/mmsegmentation/configs/deeplabv3/deeplab_building.py
  checkpoint: weights/mmseg_deeplab_building.pth
  device: cpu
  classes:
    - background
    - building
```

请将 OpenMMLab config 和 checkpoint 放到对应路径，或在 `configs/models.yaml` 中改为绝对路径。不要将 `.pth` 大权重提交到仓库。

## REST 调用示例

```bash
python scripts/run_api.py --host 127.0.0.1 --port 8765
```

```bash
curl -X POST http://127.0.0.1:8765/jobs/semantic-segmentation \
  -H "Content-Type: application/json" \
  -d "{\"image_path\":\"workspace/inputs/example.tif\",\"model_id\":\"mmseg_segformer_landcover\",\"tile_size\":1024,\"overlap\":128}"
```

## MCP 调用建议

MCP Client 中先调用：

1. `inspect_raster(image_path)`
2. `preflight_plan(image_path, task="semantic_segmentation", model_id="mmseg_segformer_landcover")`
3. `run_semantic_segmentation(image_path, model_id="mmseg_segformer_landcover", tile_size=1024, overlap=128)`

输出会包含：

- `mask.tif`：保留 CRS 和 transform 的 class-id GeoTIFF
- `probabilities.npy`：概率图或 hard mask one-hot 融合结果
- `segments.geojson` / `segments.gpkg`：按类别生成的矢量结果
- `preview.png`
- `stats.json`
- `manifest.json`

如果缺少 `mmseg/mmcv/mmengine`、config 或 checkpoint，服务会返回清晰错误；fake 语义分割不受影响。
