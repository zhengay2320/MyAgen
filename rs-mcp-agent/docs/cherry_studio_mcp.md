# Cherry Studio MCP Configuration

Start the FastAPI service first:

```bash
python scripts/run_api.py --host 127.0.0.1 --port 8765
```

Add a Cherry Studio MCP server:

- Type: `STDIO`
- Command: `python`
- Arguments: `-m rs_mcp.server`
- Working directory: project root, for example `D:\program_myself\Myagent\rs-mcp-agent`
- Environment variables:
  - `RS_SERVICE_URL=http://127.0.0.1:8765`
  - `RS_WORKSPACE=D:\program_myself\Myagent\rs-mcp-agent\workspace`

JSON-style reference:

```json
{
  "name": "rs_remote_sensing",
  "type": "stdio",
  "command": "python",
  "args": ["-m", "rs_mcp.server"],
  "cwd": "D:\\program_myself\\Myagent\\rs-mcp-agent",
  "env": {
    "RS_SERVICE_URL": "http://127.0.0.1:8765",
    "RS_WORKSPACE": "D:\\program_myself\\Myagent\\rs-mcp-agent\\workspace"
  }
}
```

Recommended workflow:

1. Call `inspect_raster`.
2. Call `preflight_plan`.
3. Run the selected `run_*` tool with explicit `tile_size` and `overlap` for large images.
4. Use `get_result_manifest`.
5. Base conclusions on `statistics`, `metrics`, and `quality_flags`.
