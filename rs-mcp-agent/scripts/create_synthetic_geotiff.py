from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rs_service.core.raster import DEFAULT_CRS, DEFAULT_TRANSFORM, create_synthetic_array, profile_from_array, write_raster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a synthetic georeferenced raster for tests and smoke runs.")
    parser.add_argument("--output", default="workspace/synthetic.tif", help="Output GeoTIFF path.")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--bands", type=int, default=4)
    parser.add_argument("--changed", action="store_true", help="Inject a deterministic changed patch.")
    parser.add_argument("--crs", default=DEFAULT_CRS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    array = create_synthetic_array(width=args.width, height=args.height, bands=args.bands, changed=args.changed)
    profile = profile_from_array(array, crs=args.crs, transform=DEFAULT_TRANSFORM)
    info = write_raster(Path(args.output), array, profile)
    print(f"wrote {info.path}")
    print(f"size={info.width}x{info.height} bands={info.count} crs={info.crs} fallback={info.fallback_container}")


if __name__ == "__main__":
    main()
