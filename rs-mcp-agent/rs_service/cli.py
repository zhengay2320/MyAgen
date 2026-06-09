from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

try:  # pragma: no cover - typer is declared but may be absent in minimal local runtimes
    import typer
except Exception:  # pragma: no cover
    typer = None

from rs_service import services


if typer is not None:
    app = typer.Typer(help="rs-mcp-agent service CLI")
else:  # pragma: no cover
    app = None


def _print_json(payload: dict | list) -> None:
    """Print JSON payloads with stable formatting."""
    print(json.dumps(payload, indent=2, default=str))


if typer is not None:

    @app.command("api")
    def api(host: str = "127.0.0.1", port: int = 8765) -> None:
        """Run the FastAPI backend."""
        import uvicorn

        uvicorn.run("rs_service.api:app", host=host, port=port, reload=False)

    @app.command("inspect")
    def inspect(path: Path) -> None:
        """Inspect a raster path."""
        _print_json(services.inspect_raster(str(path)))

    @app.command("smoke")
    def smoke() -> None:
        """Run the local smoke test script."""
        from scripts.smoke_test import main as smoke_main

        smoke_main()


def main() -> None:
    """CLI entrypoint."""
    if typer is None:  # pragma: no cover
        raise SystemExit("typer is not installed. Run: pip install -e .")
    app()


if __name__ == "__main__":
    main()
