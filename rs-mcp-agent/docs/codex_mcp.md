# Codex MCP Configuration

Start the FastAPI service first:

```bash
python scripts/run_api.py --host 127.0.0.1 --port 8765
```

Then configure Codex to start the stdio MCP server:

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

Windows example:

```toml
[mcp_servers.rs_remote_sensing]
command = "python"
args = ["-m", "rs_mcp.server"]
cwd = "D:\\program_myself\\Myagent\\rs-mcp-agent"
startup_timeout_sec = 20
tool_timeout_sec = 600

[mcp_servers.rs_remote_sensing.env]
RS_SERVICE_URL = "http://127.0.0.1:8765"
RS_WORKSPACE = "D:\\program_myself\\Myagent\\rs-mcp-agent\\workspace"
```

The MCP server communicates with the FastAPI backend only. If a tool reports that the service is not running, start it with `python scripts/run_api.py`.
