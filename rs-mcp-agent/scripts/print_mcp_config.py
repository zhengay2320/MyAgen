from __future__ import annotations

from pathlib import Path


def main() -> None:
    """Print MCP client configuration snippets for the current checkout."""
    root = Path(__file__).resolve().parents[1]
    workspace = root / "workspace"
    root_posix = root.as_posix()
    workspace_posix = workspace.as_posix()
    service_url = "http://127.0.0.1:8765"

    print("# API start command")
    print("python scripts/run_api.py --host 127.0.0.1 --port 8765")
    print()
    print("# MCP start command")
    print("python -m rs_mcp.server")
    print()
    print("# Codex: codex mcp add")
    print(
        "codex mcp add rs_remote_sensing "
        f"--env RS_SERVICE_URL={service_url} "
        f"--env RS_WORKSPACE={workspace_posix} "
        "-- python -m rs_mcp.server"
    )
    print()
    print("# Codex: config.toml snippet")
    print("[mcp_servers.rs_remote_sensing]")
    print('command = "python"')
    print('args = ["-m", "rs_mcp.server"]')
    print(f'cwd = "{root_posix}"')
    print("startup_timeout_sec = 20")
    print("tool_timeout_sec = 600")
    print("enabled = true")
    print()
    print("[mcp_servers.rs_remote_sensing.env]")
    print(f'RS_SERVICE_URL = "{service_url}"')
    print(f'RS_WORKSPACE = "{workspace_posix}"')
    print()
    print("# Cherry Studio")
    print("Type: STDIO")
    print("Command: python")
    print("Arguments: -m rs_mcp.server")
    print(f"Working directory: {root_posix}")
    print("Environment:")
    print(f"  RS_SERVICE_URL={service_url}")
    print(f"  RS_WORKSPACE={workspace_posix}")


if __name__ == "__main__":
    main()
