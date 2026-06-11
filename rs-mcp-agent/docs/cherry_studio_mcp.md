# Cherry Studio MCP 配置

## 1. 先启动 FastAPI

```bash
python scripts/run_api.py --host 127.0.0.1 --port 8765
```

确认：

```bash
curl http://127.0.0.1:8765/health
```

## 2. 添加 MCP Server

在 Cherry Studio 的 MCP 设置中新增：

- 类型：`STDIO`
- 命令：`python`
- 参数：`-m rs_mcp.server`
- 工作目录：项目根目录，例如 `/abs/path/rs-mcp-agent`
- 环境变量：
  - `RS_SERVICE_URL=http://127.0.0.1:8765`
  - `RS_WORKSPACE=/abs/path/rs-mcp-agent/workspace`

Windows 路径建议使用正斜杠：

```text
工作目录: D:/program_myself/Myagent/rs-mcp-agent
RS_WORKSPACE=D:/program_myself/Myagent/rs-mcp-agent/workspace
```

JSON 参考：

```json
{
  "name": "rs_remote_sensing",
  "type": "stdio",
  "command": "python",
  "args": ["-m", "rs_mcp.server"],
  "cwd": "/abs/path/rs-mcp-agent",
  "env": {
    "RS_SERVICE_URL": "http://127.0.0.1:8765",
    "RS_WORKSPACE": "/abs/path/rs-mcp-agent/workspace"
  }
}
```

可用当前路径生成配置提示：

```bash
python scripts/print_mcp_config.py
```

## 推荐工作流

1. `inspect_raster`
2. `preflight_plan`
3. 选择合适的 `run_*` 工具，并显式设置 `tile_size` 和 `overlap`
4. `get_result_manifest`
5. `calculate_statistics`
6. `quality_check_result`
7. `generate_report`

## 安全说明

- 强烈建议 MCP 只处理 `RS_WORKSPACE` 下的文件，避免访问无关私有目录。
- FastAPI 不要暴露公网；默认使用 `127.0.0.1:8765`。
- 不要把模型平台 token、云存储密钥写进 manifest。
- MCP stdio stdout 不得被普通日志污染。
