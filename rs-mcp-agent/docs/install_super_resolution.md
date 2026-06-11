# 超分辨率后端安装说明

超分辨率真实后端为可选能力。基础 fake super resolution 不依赖 GPU、torch、BasicSR 或 MMagic；只有选择 `swinir_x2`、`swinir_x4`、`basicsr_x4`、`mmagic_sr_stub` 时才会 lazy import。

## SwinIR

推荐将 SwinIR 官方仓库作为外部依赖管理，不要提交权重。

```bash
pip install torch pillow
```

默认占位配置：

```yaml
- id: swinir_x2
  task: super_resolution
  backend: swinir
  framework: swinir/torch
  checkpoint: weights/swinir_x2.pth
  scale: 2
  device: cpu

- id: swinir_x4
  task: super_resolution
  backend: swinir
  framework: swinir/torch
  checkpoint: weights/swinir_x4.pth
  scale: 4
  device: cpu
```

如果使用官方脚本，可在模型配置中增加 `external_command`，命令中可使用 `{input}`、`{output}`、`{scale}` 占位符：

```yaml
external_command:
  - python
  - external/SwinIR/main_test_swinir.py
  - --input
  - "{input}"
  - --output
  - "{output}"
  - --scale
  - "{scale}"
```

## BasicSR

```bash
pip install basicsr torch pyyaml
```

```yaml
- id: basicsr_x4
  task: super_resolution
  backend: basicsr
  framework: basicsr
  config: external/basicsr/options/sr_x4.yml
  checkpoint: weights/basicsr_x4.pth
  scale: 4
  device: cpu
```

## MMagic

`mmagic_sr_stub` 是预留接口，当前 MVP 会返回清晰错误。后续可在 `rs_service/adapters/mmagic_adapter.py` 中接入真实 MMagic inferencer。

## Transform 检查

超分输出 GeoTIFF 保持输入 CRS 不变，并将 transform 的像元大小除以 `scale`。例如输入像元大小为 1m，`scale=4` 后输出像元大小为 0.25m。manifest 的 `statistics.transform_matches_scale` 会记录检查结果。

## 参考图指标

如果请求提供 `reference_path`，pipeline 会将参考图按 MVP 最近邻方式对齐到 SR 输出大小，并计算：

- `psnr`
- `ssim`
- `lpips`，可选；缺少 `lpips/torch` 时跳过并写入 quality flag

如果没有 `reference_path`，不会输出 PSNR/SSIM，manifest 结论会明确：`无参考图，无法定量评价重建精度。`

## API 示例

```bash
curl -X POST http://127.0.0.1:8765/jobs/super-resolution \
  -H "Content-Type: application/json" \
  -d "{\"image_path\":\"workspace/inputs/lr.tif\",\"model_id\":\"swinir_x4\",\"scale\":4,\"tile_size\":256,\"overlap\":32}"
```

带参考图：

```bash
curl -X POST http://127.0.0.1:8765/jobs/super-resolution \
  -H "Content-Type: application/json" \
  -d "{\"image_path\":\"workspace/inputs/lr.tif\",\"reference_path\":\"workspace/inputs/hr_ref.tif\",\"scale\":2}"
```
