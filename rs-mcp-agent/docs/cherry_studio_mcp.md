# Cherry Studio MCP Configuration

Add a new MCP server in Cherry Studio with stdio transport.

```json
{
  "name": "rs-mcp-agent",
  "type": "stdio",
  "command": "python",
  "args": ["-m", "rs_mcp.server"],
  "cwd": "D:\\program_myself\\Myagent\\rs-mcp-agent"
}
```

Recommended first-stage workflow:

1. Run `scripts/create_synthetic_geotiff.py` to create a test raster.
2. Call `inspect_raster`.
3. Call `preflight_plan` with `tile_size` and `overlap`.
4. Run one of the tiled pipelines.
5. Load the result with `get_result_manifest`.
6. Call `calculate_statistics`, `quality_check_result`, and `generate_report`.

Large model weights are intentionally not included. Use `scripts/download_weights.py --print-plan` to create and inspect the expected local checkpoint paths.
