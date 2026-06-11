# Final Release Check

## 当前支持能力

- FastAPI 后端：`rs_service.api`
- stdio MCP Server：`rs_mcp.server`
- GeoTIFF / fallback raster IO
- 大图切片、overlap 拼接、坐标恢复
- GeoJSON / GPKG 矢量输出
- 目标检测、旋转框检测、语义分割、实例分割
- 双时相变化检测与 MVP 重采样对齐
- 超分辨率与 transform 缩放检查
- NDVI、NDWI、MNDWI、NDBI、SAVI、EVI 光谱指数
- 统计分析、质量检查、规则报告
- 每个 job 统一输出 `manifest.json`

## Fake 模式状态

Fake 模式是默认可用路径，不依赖 GPU、torch、OpenMMLab 或真实模型权重。

已验证 fake 任务：

- `fake_detection`
- `fake_oriented_detection`
- `fake_segmentation`
- `fake_instance`
- `fake_change`
- `fake_super_resolution`
- spectral index calculator

`scripts/smoke_test.py` 会跑通 inspect、preflight、所有 fake job、analyze 和 report。

## 真实模型 Adapter 状态

真实模型 adapter 均为 lazy import，不会影响基础 API/MCP 启动。

- Ultralytics YOLO：`yolo_detection`、`yolo_instance_segmentation`
- SAHI：`sahi_yolo_detection`
- MMSegmentation：`mmseg_segformer_landcover`、`mmseg_deeplab_building`
- MMDetection：`mmdet_maskrcnn_instance`
- MMRotate：`mmrotate_dota_oriented`
- Open-CD：`opencd_changer_building`、`opencd_changeformer_landcover`
- SwinIR：`swinir_x2`、`swinir_x4`
- BasicSR：`basicsr_x4`
- MMagic：`mmagic_sr_stub`，预留接口，当前返回清晰 MVP 错误

真实模型权重路径是占位，用户需自行下载并配置 `configs/models.yaml`。仓库不提交 `.pt/.pth/.ckpt` 权重。

## 启动 API

```bash
python scripts/run_api.py --host 127.0.0.1 --port 8765
```

或：

```bash
make api
```

健康检查：

```bash
curl http://127.0.0.1:8765/health
```

## 启动 MCP

先启动 FastAPI，再启动 MCP：

```bash
python -m rs_mcp.server
```

或：

```bash
make mcp
```

MCP stdio server 的 stdout 只能用于 MCP 协议，普通日志必须写 stderr 或文件。

## Codex 配置

可运行：

```bash
python scripts/print_mcp_config.py
```

配置示例见：

- `.codex/config.toml.example`
- `docs/codex_mcp.md`

## Cherry Studio 配置

- 类型：`STDIO`
- 命令：`python`
- 参数：`-m rs_mcp.server`
- 工作目录：项目根目录
- 环境变量：
  - `RS_SERVICE_URL=http://127.0.0.1:8765`
  - `RS_WORKSPACE=/abs/path/rs-mcp-agent/workspace`

详见 `docs/cherry_studio_mcp.md`。

## 路径与安全

- API/MCP job 默认输出到 `workspace/outputs/{job_id}`。
- 建议只处理 `RS_WORKSPACE` 下文件。
- 不要将 FastAPI 暴露公网。
- 不要在 manifest、日志或报告中写入私有 token。
- `.gitignore` 已忽略 `weights/`、workspace 运行输出和常见权重扩展。

## 已知限制

- 当前 fake 模式用于工程闭环验证，不代表真实模型精度。
- 若未安装 Rasterio/GDAL，会使用 fallback raster container，不是生产级 GeoTIFF。
- 若未安装 GeoPandas/Fiona，GPKG 输出会降级为 JSON fallback。
- OpenMMLab、Open-CD、BasicSR、SwinIR 版本兼容需要用户按模型配置自行管理环境。
- MMagic 当前为 stub。
- 无参考超分辨率不会输出 PSNR/SSIM。
- 变化检测自动对齐仅做 MVP 最近邻重采样，不做复杂特征配准。

## 下一步建议

- 将 `rs_service.registry` 完全改为读取 `configs/models.yaml`。
- 增加真实模型小权重或 mock fixture 的 CI 测试矩阵。
- 增加 workspace 路径 allowlist 强制校验。
- 增加异步 job 队列和任务取消能力。
- 增加 STAC/COG 输入输出支持。

## 本次发布前检查记录

- `rg "TODO|NotImplementedError|\bpass\b" rs_service rs_mcp scripts tests`：无命中。
- `python -m compileall rs_service rs_mcp scripts`：通过。
- `import rs_service.api / rs_mcp.server / scripts.run_api / scripts.run_mcp`：通过。
- `python scripts/dev_check.py`：通过；重型真实模型依赖缺失均作为 optional 状态记录。
- `python scripts/create_synthetic_geotiff.py`：通过。
- `python scripts/smoke_test.py`：通过。
- `python -m unittest discover -s tests -p "test_*.py"`：通过，72 tests，2 skipped。
- `python -m pytest -q`：当前本地运行时未安装 pytest，结果为 `No module named pytest`。安装 `pip install -e ".[dev]"` 后可运行。
