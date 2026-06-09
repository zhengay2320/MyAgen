from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    """Parse API startup arguments."""
    parser = argparse.ArgumentParser(description="Run rs_service FastAPI backend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    """Start the API server through uvicorn."""
    args = parse_args()
    import uvicorn

    uvicorn.run("rs_service.api:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
