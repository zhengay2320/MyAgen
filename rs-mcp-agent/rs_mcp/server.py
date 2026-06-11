from __future__ import annotations

import json
import sys
from typing import Any

from rs_mcp.tools import TOOL_FUNCTIONS, tool_definitions

try:  # pragma: no cover - requires the official mcp package
    from mcp.server.fastmcp import FastMCP
except Exception:  # pragma: no cover - fallback is covered by tests
    FastMCP = None


MCP_INSTRUCTIONS = """
Remote sensing workflow instructions:
1. For raster tasks, call inspect_raster first, then preflight_plan, then the appropriate run_* tool.
2. Large images must use explicit tile_size and overlap chosen from preflight_plan.
3. Conclusions must come from manifest, statistics, metrics, or quality_flags. Do not guess.
4. Return output file paths exactly as provided by rs_service manifests and job responses.
5. This MCP server calls the FastAPI backend only; start it with python scripts/run_api.py if tools cannot connect.
""".strip()


def build_fastmcp_server() -> Any:
    """Build the official FastMCP stdio server."""
    if FastMCP is None:  # pragma: no cover
        raise RuntimeError("mcp package is not installed")
    try:
        mcp = FastMCP("rs_remote_sensing", instructions=MCP_INSTRUCTIONS)
    except TypeError:  # pragma: no cover - older SDKs may not accept instructions
        mcp = FastMCP("rs_remote_sensing")

    @mcp.tool()
    def inspect_raster(image_path: str) -> dict[str, Any]:
        """Inspect raster metadata."""
        return TOOL_FUNCTIONS["inspect_raster"](image_path=image_path)

    @mcp.tool()
    def preflight_plan(image_path: str, task: str, model_id: str | None = None) -> dict[str, Any]:
        """Plan tiled inference for a task."""
        return TOOL_FUNCTIONS["preflight_plan"](image_path=image_path, task=task, model_id=model_id)

    @mcp.tool()
    def list_models() -> dict[str, Any]:
        """List models available from rs_service."""
        return TOOL_FUNCTIONS["list_models"]()

    @mcp.tool()
    def run_object_detection(
        image_path: str,
        model_id: str = "fake_detection",
        tile_size: int | None = None,
        overlap: int | None = None,
        confidence_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Submit an object detection job."""
        return TOOL_FUNCTIONS["run_object_detection"](
            image_path=image_path,
            model_id=model_id,
            tile_size=tile_size,
            overlap=overlap,
            confidence_threshold=confidence_threshold,
        )

    @mcp.tool()
    def run_oriented_detection(
        image_path: str,
        model_id: str = "fake_oriented_detection",
        tile_size: int | None = None,
        overlap: int | None = None,
        confidence_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Submit an oriented detection job."""
        return TOOL_FUNCTIONS["run_oriented_detection"](
            image_path=image_path,
            model_id=model_id,
            tile_size=tile_size,
            overlap=overlap,
            confidence_threshold=confidence_threshold,
        )

    @mcp.tool()
    def run_semantic_segmentation(
        image_path: str,
        model_id: str = "fake_semantic_segmentation",
        tile_size: int | None = None,
        overlap: int | None = None,
    ) -> dict[str, Any]:
        """Submit a semantic segmentation job."""
        return TOOL_FUNCTIONS["run_semantic_segmentation"](
            image_path=image_path,
            model_id=model_id,
            tile_size=tile_size,
            overlap=overlap,
        )

    @mcp.tool()
    def run_instance_segmentation(
        image_path: str,
        model_id: str = "fake_instance_segmentation",
        tile_size: int | None = None,
        overlap: int | None = None,
    ) -> dict[str, Any]:
        """Submit an instance segmentation job."""
        return TOOL_FUNCTIONS["run_instance_segmentation"](
            image_path=image_path,
            model_id=model_id,
            tile_size=tile_size,
            overlap=overlap,
        )

    @mcp.tool()
    def run_change_detection(
        image_t1_path: str,
        image_t2_path: str,
        model_id: str = "fake_change",
        tile_size: int | None = None,
        overlap: int | None = None,
        auto_align: bool | None = None,
    ) -> dict[str, Any]:
        """Submit a change detection job."""
        return TOOL_FUNCTIONS["run_change_detection"](
            image_t1_path=image_t1_path,
            image_t2_path=image_t2_path,
            model_id=model_id,
            tile_size=tile_size,
            overlap=overlap,
            auto_align=auto_align,
        )

    @mcp.tool()
    def run_super_resolution(
        image_path: str,
        model_id: str = "fake_super_resolution",
        scale: int = 2,
        tile_size: int | None = None,
        overlap: int | None = None,
        reference_path: str | None = None,
    ) -> dict[str, Any]:
        """Submit a super-resolution job."""
        return TOOL_FUNCTIONS["run_super_resolution"](
            image_path=image_path,
            model_id=model_id,
            scale=scale,
            tile_size=tile_size,
            overlap=overlap,
            reference_path=reference_path,
        )

    @mcp.tool()
    def run_spectral_indices(image_path: str, indices: list[str] | None = None) -> dict[str, Any]:
        """Submit a spectral index job."""
        return TOOL_FUNCTIONS["run_spectral_indices"](image_path=image_path, indices=indices or ["ndvi"])

    @mcp.tool()
    def calculate_statistics(job_id: str) -> dict[str, Any]:
        """Calculate statistics for an existing job."""
        return TOOL_FUNCTIONS["calculate_statistics"](job_id=job_id)

    @mcp.tool()
    def quality_check_result(job_id: str) -> dict[str, Any]:
        """Quality-check an existing job."""
        return TOOL_FUNCTIONS["quality_check_result"](job_id=job_id)

    @mcp.tool()
    def generate_report(job_id: str, output_format: str = "markdown") -> dict[str, Any]:
        """Generate a report for an existing job."""
        return TOOL_FUNCTIONS["generate_report"](job_id=job_id, output_format=output_format)

    @mcp.tool()
    def get_job_status(job_id: str) -> dict[str, Any]:
        """Get job status."""
        return TOOL_FUNCTIONS["get_job_status"](job_id=job_id)

    @mcp.tool()
    def get_result_manifest(job_id: str) -> dict[str, Any]:
        """Get a job manifest."""
        return TOOL_FUNCTIONS["get_result_manifest"](job_id=job_id)

    return mcp


def _jsonrpc_result(request_id: Any, result: Any) -> dict[str, Any]:
    """Build a JSON-RPC result response."""
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _jsonrpc_error(request_id: Any, message: str, code: int = -32000) -> dict[str, Any]:
    """Build a JSON-RPC error response."""
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def handle_jsonrpc(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Handle a minimal JSON-RPC subset for environments without the MCP SDK."""
    request_id = payload.get("id")
    method = payload.get("method")
    if "tool" in payload:
        name = payload["tool"]
        args = payload.get("arguments", {})
        return {"ok": True, "result": TOOL_FUNCTIONS[name](**args)}
    if method == "initialize":
        return _jsonrpc_result(
            request_id,
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "rs_remote_sensing", "version": "0.1.0"},
                "instructions": MCP_INSTRUCTIONS,
            },
        )
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return _jsonrpc_result(request_id, {"tools": tool_definitions()})
    if method == "tools/call":
        params = payload.get("params", {})
        name = params.get("name")
        args = params.get("arguments") or {}
        if name not in TOOL_FUNCTIONS:
            return _jsonrpc_error(request_id, f"Unknown tool: {name}", code=-32601)
        try:
            result = TOOL_FUNCTIONS[name](**args)
            return _jsonrpc_result(
                request_id,
                {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, default=str)}], "isError": False},
            )
        except Exception as exc:
            return _jsonrpc_result(
                request_id,
                {"content": [{"type": "text", "text": str(exc)}], "isError": True},
            )
    return _jsonrpc_error(request_id, f"Unsupported method: {method}", code=-32601)


def run_fallback_stdio() -> None:
    """Run a minimal stdio JSON-RPC server without printing logs to stdout."""
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            response = handle_jsonrpc(payload)
        except Exception as exc:
            print(f"rs_mcp fallback error: {exc}", file=sys.stderr)
            response = _jsonrpc_error(None, str(exc))
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False, default=str) + "\n")
            sys.stdout.flush()


def main() -> None:
    """Run the stdio MCP server."""
    if FastMCP is not None:  # pragma: no cover - requires the official mcp package
        build_fastmcp_server().run()
    else:
        run_fallback_stdio()


if __name__ == "__main__":
    main()
