# Codex MCP 配置

本 MCP Server 是 stdio server，只负责把 MCP tool 调用转发给本地 FastAPI 后端。请先启动 API，再启动 Codex。

## 1. 启动 FastAPI

```bash
python scripts/run_api.py --host 127.0.0.1 --port 8765
```

确认服务可用：

```bash
curl http://127.0.0.1:8765/health
```

## 2A. 使用 `codex mcp add`

在项目根目录运行：

```bash
codex mcp add rs_remote_sensing --env RS_SERVICE_URL=http://127.0.0.1:8765 --env RS_WORKSPACE=/abs/path/workspace -- python -m rs_mcp.server
```

Windows 示例可使用正斜杠：

```bash
codex mcp add rs_remote_sensing --env RS_SERVICE_URL=http://127.0.0.1:8765 --env RS_WORKSPACE=D:/program_myself/Myagent/rs-mcp-agent/workspace -- python -m rs_mcp.server
```

## 2B. 使用 `config.toml`

将以下片段加入 Codex 配置文件：

```toml
[mcp_servers.rs_remote_sensing]
command = "python"
args = ["-m", "rs_mcp.server"]
cwd = "/ABSOLUTE/PATH/TO/rs-mcp-agent"
startup_timeout_sec = 20
tool_timeout_sec = 600
enabled = true

[mcp_servers.rs_remote_sensing.env]
RS_SERVICE_URL = "http://127.0.0.1:8765"
RS_WORKSPACE = "/ABSOLUTE/PATH/TO/rs-mcp-agent/workspace"
```

Windows 示例：

```toml
[mcp_servers.rs_remote_sensing]
command = "python"
args = ["-m", "rs_mcp.server"]
cwd = "D:/program_myself/Myagent/rs-mcp-agent"
startup_timeout_sec = 20
tool_timeout_sec = 600
enabled = true

[mcp_servers.rs_remote_sensing.env]
RS_SERVICE_URL = "http://127.0.0.1:8765"
RS_WORKSPACE = "D:/program_myself/Myagent/rs-mcp-agent/workspace"
```

也可以生成当前路径配置：

```bash
python scripts/print_mcp_config.py
```

## 3. 在 Codex 中检查工具

启动 Codex 后输入：

```text
/mcp
```

应看到 `rs_remote_sensing` 及其工具列表，例如 `inspect_raster`、`preflight_plan`、`run_change_detection`、`generate_report`。

## 安全说明

- 强烈建议将 `RS_WORKSPACE` 限制到项目 `workspace` 目录。
- 不要把 FastAPI 服务暴露到公网，默认只绑定 `127.0.0.1`。
- 不要在 manifest、日志或报告中写入私有 token。
- MCP stdio 的 stdout 只能用于 MCP 协议通信，普通日志必须写 stderr 或文件。
