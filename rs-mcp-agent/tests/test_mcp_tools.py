from __future__ import annotations

import unittest

from rs_mcp.server import handle_jsonrpc
from rs_mcp.tools import tool_names


REQUIRED_TOOLS = {
    "inspect_raster",
    "preflight_plan",
    "list_models",
    "run_object_detection",
    "run_oriented_detection",
    "run_semantic_segmentation",
    "run_instance_segmentation",
    "run_change_detection",
    "run_super_resolution",
    "run_spectral_indices",
    "calculate_statistics",
    "quality_check_result",
    "generate_report",
    "get_job_status",
    "get_result_manifest",
}


class McpToolTests(unittest.TestCase):
    def test_required_tools_are_registered(self) -> None:
        self.assertEqual(set(tool_names()), REQUIRED_TOOLS)

    def test_fallback_tools_list_jsonrpc(self) -> None:
        response = handle_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        self.assertIsNotNone(response)
        tools = response["result"]["tools"]
        self.assertEqual({tool["name"] for tool in tools}, REQUIRED_TOOLS)


if __name__ == "__main__":
    unittest.main()
