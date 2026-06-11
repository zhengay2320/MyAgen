# Example Prompts

以下提示可在 Codex 或 Cherry Studio 中使用。建议先把待处理影像放入 `RS_WORKSPACE`，并使用绝对路径或 workspace 内路径。

## 读取影像信息

```text
请检查 workspace/inputs/sample.tif 的影像信息，包括尺寸、波段数、CRS、bounds、分辨率和 nodata。
```

## 建筑物语义分割

```text
请先 inspect_raster，再 preflight_plan，然后对 workspace/inputs/city.tif 运行语义分割。使用 fake_segmentation，tile_size=1024，overlap=128。完成后生成统计、质量检查和报告。
```

真实 MMSeg 后端示例：

```text
请对 workspace/inputs/city.tif 做建筑物语义分割，model_id 使用 mmseg_deeplab_building。先检查影像和切片计划，如果权重或依赖缺失，请清楚说明需要安装或配置什么。
```

## 车辆目标检测

```text
请对 workspace/inputs/road.tif 做车辆目标检测，使用 fake_detection，tile_size=1024，overlap=192，生成 detections.geojson、统计和报告。
```

真实 YOLO 后端示例：

```text
请使用 yolo_detection 对 workspace/inputs/road.tif 做目标检测，confidence_threshold=0.3。结果只基于 manifest 和 statistics 总结。
```

## 舰船旋转框检测

```text
请对 workspace/inputs/harbor.tif 做舰船旋转框检测，先做 preflight_plan，再使用 fake_oriented_detection 跑通流程，输出 oriented_detections.geojson 和报告。
```

真实 MMRotate 后端示例：

```text
请使用 mmrotate_dota_oriented 对 workspace/inputs/harbor.tif 做遥感旋转框检测。如果模型权重不存在，请返回清晰错误和需要配置的路径。
```

## 双时相变化检测

```text
请比较 workspace/inputs/t1.tif 和 workspace/inputs/t2.tif，进行双时相变化检测。先检查两期影像是否对齐；若不对齐，不要自动猜测结论。使用 fake_change，tile_size=1024，overlap=128，并生成变化面积统计和报告。
```

带自动对齐：

```text
请对 workspace/inputs/t1.tif 和 workspace/inputs/t2.tif 做变化检测，如果 CRS 或分辨率不一致，设置 auto_align=true，并在报告中说明配准风险。
```

## 超分辨率

```text
请对 workspace/inputs/lr.tif 做 2 倍超分辨率，使用 fake_super_resolution，tile_size=256，overlap=32。检查输出 GeoTIFF transform 是否正确缩放，并生成报告。
```

带参考图：

```text
请对 workspace/inputs/lr.tif 做 2 倍超分，并使用 workspace/inputs/hr_ref.tif 作为 reference_path 计算 PSNR 和 SSIM。没有 LPIPS 依赖时跳过并记录 warning。
```

## NDVI 计算

```text
请计算 workspace/inputs/multispectral.tif 的 NDVI。若未提供 band_map，按 4 波段默认 blue=1, green=2, red=3, nir=4。输出 ndvi.tif、预览、统计和报告，并统计 NDVI > 0.3 的植被面积。
```

显式 band_map：

```text
请计算 NDVI、NDWI、MNDWI、NDBI、SAVI、EVI。band_map 为 blue=1, green=2, red=3, nir=4, swir1=5, swir2=6。阈值统计包括 ndvi > 0.3 和 mndwi > 0。
```

## 生成报告

```text
请读取最近一个遥感任务的 manifest.json，基于 statistics、metrics 和 quality_flags 生成中文 Markdown 报告。不要凭空推断模型精度。
```

## 安全提醒

- 只处理 `RS_WORKSPACE` 下的文件。
- 不要把 token、密钥或私有下载链接写入 prompt、manifest 或报告。
- 不要要求 MCP 访问公网 API，除非你明确知道服务暴露范围和安全风险。
