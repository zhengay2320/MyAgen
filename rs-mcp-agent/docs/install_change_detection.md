# Open-CD 变化检测后端安装说明

Open-CD 是可选真实模型后端。基础 FastAPI、MCP stdio server 和 fake change detection 不依赖 Open-CD；只有选择 `opencd_changer_building` 或 `opencd_changeformer_landcover` 时才会 lazy import。

## 推荐独立环境

```bash
conda create -n rs-opencd python=3.10 -y
conda activate rs-opencd
pip install -e ".[dev]"
pip install -U openmim
mim install mmengine mmcv
```

Open-CD 的安装方式会随官方仓库和 OpenMMLab 版本变化，建议按 Open-CD 官方文档从源码安装，并确保 `opencd.apis.init_model` / `opencd.apis.inference_model` 可用。

## 配置 config 与 checkpoint

默认占位配置：

```yaml
- id: opencd_changer_building
  task: change_detection
  backend: opencd
  framework: open-cd
  config: external/opencd/configs/changer/changer_building.py
  checkpoint: weights/opencd_changer_building.pth
  device: cpu
  threshold: 0.5

- id: opencd_changeformer_landcover
  task: change_detection
  backend: opencd
  framework: open-cd
  config: external/opencd/configs/changeformer/changeformer_landcover.py
  checkpoint: weights/opencd_changeformer_landcover.pth
  device: cpu
  threshold: 0.5
```

不要提交 `.pth` 权重。可以在 `configs/models.yaml` 中改成绝对路径。

## 运行示例

生成双时相测试数据：

```bash
python scripts/create_synthetic_geotiff.py --change-pair
```

启动 API：

```bash
python scripts/run_api.py --host 127.0.0.1 --port 8765
```

调用真实 Open-CD 后端：

```bash
curl -X POST http://127.0.0.1:8765/jobs/change-detection \
  -H "Content-Type: application/json" \
  -d "{\"before_path\":\"workspace/synthetic_t1.tif\",\"after_path\":\"workspace/synthetic_t2.tif\",\"model_id\":\"opencd_changer_building\",\"tile_size\":1024,\"overlap\":128,\"threshold\":0.5,\"auto_align\":false}"
```

如果两个影像 CRS、分辨率、transform、bounds 或尺寸不一致，可以设置 `auto_align=true`。MVP 对齐使用最近邻重采样到 t1 网格，不做复杂特征配准，因此 manifest 会记录配准风险。

输出包括：

- `change_mask.tif`
- `change_probability.npy`
- `changes.geojson` / `changes.gpkg`
- `preview.png`
- `stats.json`
- `manifest.json`

## MCP 调用建议

1. `inspect_raster(image_t1_path)` 和 `inspect_raster(image_t2_path)`
2. `preflight_plan(image_t1_path, task="change_detection")`
3. `run_change_detection(image_t1_path, image_t2_path, model_id="opencd_changer_building", tile_size=1024, overlap=128)`
4. `calculate_statistics(job_id)`、`quality_check_result(job_id)`、`generate_report(job_id)`

结论必须来自 manifest、statistics 和 quality_flags；若存在 alignment warnings，不应给出高置信度结论。
