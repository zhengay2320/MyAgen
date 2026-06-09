from __future__ import annotations

import json
import sys
from typing import Any

from rs_mcp.tools import TOOL_FUNCTIONS, tool_definitions

try:  # pragma: no cover - requires the official mcp package
    from mcp.server.fastmcp import FastMCP
except Exception:  # pragma: no cover - fallback is covered by tests/smoke
    FastMCP = None


def build_fastmcp_server() -> Any:
    if FastMCP is None:  # pragma: no cover
        raise RuntimeError("mcp package is not installed")
    mcp = FastMCP("rs-mcp-agent")

    @mcp.tool()
    def inspect_raster(path: str) -> dict[str, Any]:
        return TOOL_FUNCTIONS["inspect_raster"](path=path)

    @mcp.tool()
    def preflight_plan(path: str, tile_size: int | None = None, overlap: int | None = None, task: str = "detection") -> dict[str, Any]:
        return TOOL_FUNCTIONS["preflight_plan"](path=path, tile_size=tile_size, overlap=overlap, task=task)

    @mcp.tool()
    def list_models() -> dict[str, Any]:
        return TOOL_FUNCTIONS["list_models"]()

    @mcp.tool()
    def run_object_detection(
        image_path: str,
        output_dir: str | None = None,
        tile_size: int = 512,
        overlap: int = 64,
        model_id: str | None = None,
        score_threshold: float = 0.0,
        nms_threshold: float = 0.5,
    ) -> dict[str, Any]:
        return TOOL_FUNCTIONS["run_object_detection"](
            image_path=image_path,
            output_dir=output_dir,
            tile_size=tile_size,
            overlap=overlap,
            model_id=model_id,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
        )

    @mcp.tool()
    def run_oriented_detection(
        image_path: str,
        output_dir: str | None = None,
        tile_size: int = 512,
        overlap: int = 64,
        model_id: str | None = None,
        score_threshold: float = 0.0,
    ) -> dict[str, Any]:
        return TOOL_FUNCTIONS["run_oriented_detection"](
            image_path=image_path,
            output_dir=output_dir,
            tile_size=tile_size,
            overlap=overlap,
            model_id=model_id,
            score_threshold=score_threshold,
        )

    @mcp.tool()
    def run_semantic_segmentation(
        image_path: str,
        output_dir: str | None = None,
        tile_size: int = 512,
        overlap: int = 64,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        return TOOL_FUNCTIONS["run_semantic_segmentation"](
            image_path=image_path,
            output_dir=output_dir,
            tile_size=tile_size,
            overlap=overlap,
            model_id=model_id,
        )

    @mcp.tool()
    def run_instance_segmentation(
        image_path: str,
        output_dir: str | None = None,
        tile_size: int = 512,
        overlap: int = 64,
        model_id: str | None = None,
        score_threshold: float = 0.0,
        nms_threshold: float = 0.5,
    ) -> dict[str, Any]:
        return TOOL_FUNCTIONS["run_instance_segmentation"](
            image_path=image_path,
            output_dir=output_dir,
            tile_size=tile_size,
            overlap=overlap,
            model_id=model_id,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
        )

    @mcp.tool()
    def run_change_detection(
        before_path: str,
        after_path: str,
        output_dir: str | None = None,
        tile_size: int = 512,
        overlap: int = 64,
        model_id: str | None = None,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        return TOOL_FUNCTIONS["run_change_detection"](
            before_path=before_path,
            after_path=after_path,
            output_dir=output_dir,
            tile_size=tile_size,
            overlap=overlap,
            model_id=model_id,
            threshold=threshold,
        )

    @mcp.tool()
    def run_super_resolution(
        image_path: str,
        output_dir: str | None = None,
        tile_size: int = 512,
        overlap: int = 64,
        scale: int = 2,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        return TOOL_FUNCTIONS["run_super_resolution"](
            image_path=image_path,
            output_dir=output_dir,
            tile_size=tile_size,
            overlap=overlap,
            scale=scale,
            model_id=model_id,
        )

    @mcp.tool()
    def run_spectral_indices(
        image_path: str,
        output_dir: str | None = None,
        indices: list[str] | None = None,
        band_mapping: dict[str, int] | None = None,
        tile_size: int = 512,
        overlap: int = 64,
    ) -> dict[str, Any]:
        return TOOL_FUNCTIONS["run_spectral_indices"](
            image_path=image_path,
            output_dir=output_dir,
            indices=indices,
            band_mapping=band_mapping,
            tile_size=tile_size,
            overlap=overlap,
        )

    @mcp.tool()
    def calculate_statistics(
        input_path: str | None = None,
        output_dir: str | None = None,
        manifest_path: str | None = None,
        zones_path: str | None = None,
    ) -> dict[str, Any]:
        return TOOL_FUNCTIONS["calculate_statistics"](
            input_path=input_path,
            output_dir=output_dir,
            manifest_path=manifest_path,
            zones_path=zones_path,
        )

    @mcp.tool()
    def quality_check_result(
        input_path: str | None = None,
        output_dir: str | None = None,
        manifest_path: str | None = None,
    ) -> dict[str, Any]:
        return TOOL_FUNCTIONS["quality_check_result"](input_path=input_path, output_dir=output_dir, manifest_path=manifest_path)

    @mcp.tool()
    def generate_report(
        manifest_path: str,
        output_dir: str | None = None,
        title: str = "Remote Sensing Processing Report",
    ) -> dict[str, Any]:
        return TOOL_FUNCTIONS["generate_report"](manifest_path=manifest_path, output_dir=output_dir, title=title)

    @mcp.tool()
    def get_job_status(job_id: str) -> dict[str, Any]:
        return TOOL_FUNCTIONS["get_job_status"](job_id=job_id)

    @mcp.tool()
    def get_result_manifest(job_id: str | None = None, manifest_path: str | None = None) -> dict[str, Any]:
        return TOOL_FUNCTIONS["get_result_manifest"](job_id=job_id, manifest_path=manifest_path)

    return mcp


def _jsonrpc_result(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _jsonrpc_error(request_id: Any, message: str, code: int = -32000) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def handle_jsonrpc(payload: dict[str, Any]) -> dict[str, Any] | None:
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
                "serverInfo": {"name": "rs-mcp-agent", "version": "0.1.0"},
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
        result = TOOL_FUNCTIONS[name](**args)
        return _jsonrpc_result(
            request_id,
            {"content": [{"type": "text", "text": json.dumps(result, default=str)}], "isError": False},
        )
    return _jsonrpc_error(request_id, f"Unsupported method: {method}", code=-32601)


def run_fallback_stdio() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            response = handle_jsonrpc(payload)
        except Exception as exc:
            response = _jsonrpc_error(None, str(exc))
        if response is not None:
            sys.stdout.write(json.dumps(response, default=str) + "\n")
            sys.stdout.flush()


def main() -> None:
    if FastMCP is not None:  # pragma: no cover - requires the official mcp package
        build_fastmcp_server().run()
    else:
        run_fallback_stdio()


if __name__ == "__main__":
    main()
