# Quickstart

## 1. Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 2. Check The Project

```bash
make dev-check
```

If `make` is unavailable:

```bash
python scripts/dev_check.py
```

## 3. Create Synthetic Data

```bash
make synthetic
```

or:

```bash
python scripts/create_synthetic_geotiff.py --output workspace/synthetic.tif
```

For change detection:

```bash
python scripts/create_synthetic_geotiff.py --change-pair
```

## 4. Run Smoke Test

```bash
make smoke
```

This runs fake detection, oriented detection, semantic segmentation, instance segmentation, change detection, super-resolution, spectral indices, analysis, and report generation.

To validate the multi-conda subprocess protocol without installing real model stacks, run:

```bash
python scripts/smoke_test_subprocess.py
```

普通 fake 模式不需要多 conda。真实模型部署时可以选择 `*_subprocess` model_id，让 YOLO、OpenMMLab、Open-CD、SwinIR/BasicSR 在独立 conda 环境运行；`rs-mcp-base` 只负责 MCP、FastAPI、GIS、切片、拼接、统计、报告和 manifest。See [multi_conda_workers.md](multi_conda_workers.md).

## 5. Start FastAPI

```bash
make api
```

Default URL:

```text
http://127.0.0.1:8765
```

Health check:

```bash
curl http://127.0.0.1:8765/health
```

## 6. Start MCP Server

In another terminal:

```bash
make mcp
```

The MCP server uses stdio. Do not print ordinary logs to stdout.

## 7. Codex MCP Config

```toml
[mcp_servers.rs_remote_sensing]
command = "python"
args = ["-m", "rs_mcp.server"]
cwd = "/ABSOLUTE/PATH/TO/rs-mcp-agent"
startup_timeout_sec = 20
tool_timeout_sec = 600

[mcp_servers.rs_remote_sensing.env]
RS_SERVICE_URL = "http://127.0.0.1:8765"
RS_WORKSPACE = "/ABSOLUTE/PATH/TO/rs-mcp-agent/workspace"
```

See [codex_mcp.md](codex_mcp.md).

## 8. Cherry Studio Config

- Type: `STDIO`
- Command: `python`
- Arguments: `-m rs_mcp.server`
- Working directory: project root
- Environment:
  - `RS_SERVICE_URL=http://127.0.0.1:8765`
  - `RS_WORKSPACE=/ABSOLUTE/PATH/TO/rs-mcp-agent/workspace`

See [cherry_studio_mcp.md](cherry_studio_mcp.md).
