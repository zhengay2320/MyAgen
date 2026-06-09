# Codex MCP Configuration

Use the stdio MCP server from this project root.

```json
{
  "mcpServers": {
    "rs-mcp-agent": {
      "command": "python",
      "args": ["-m", "rs_mcp.server"],
      "cwd": "/absolute/path/to/rs-mcp-agent"
    }
  }
}
```

On Windows, use an absolute `cwd` such as:

```json
{
  "mcpServers": {
    "rs-mcp-agent": {
      "command": "python",
      "args": ["-m", "rs_mcp.server"],
      "cwd": "D:\\program_myself\\Myagent\\rs-mcp-agent"
    }
  }
}
```

The server exposes:

`inspect_raster`, `preflight_plan`, `list_models`, `run_object_detection`, `run_oriented_detection`, `run_semantic_segmentation`, `run_instance_segmentation`, `run_change_detection`, `run_super_resolution`, `run_spectral_indices`, `calculate_statistics`, `quality_check_result`, `generate_report`, `get_job_status`, and `get_result_manifest`.
