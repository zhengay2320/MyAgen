from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rs_service.core.raster import DEFAULT_CRS, DEFAULT_TRANSFORM
from rs_service.core.raster_io import profile_for_array, write_geotiff


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Create a synthetic georeferenced raster for tests and smoke runs.")
    parser.add_argument("--output", default="workspace/synthetic.tif", help="Output GeoTIFF path.")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--bands", type=int, default=3)
    parser.add_argument("--changed", action="store_true", help="Inject a deterministic changed patch.")
    parser.add_argument("--crs", default=DEFAULT_CRS)
    return parser.parse_args()


def create_visual_synthetic_array(width: int, height: int, bands: int = 3, changed: bool = False) -> np.ndarray:
    """Create a small RGB-like raster with rectangle and circle targets."""
    if bands < 1:
        raise ValueError("bands must be positive")
    yy, xx = np.mgrid[0:height, 0:width]
    base = np.zeros((bands, height, width), dtype=np.uint8)
    base[0] = np.clip(30 + xx * 80 / max(width - 1, 1), 0, 255).astype(np.uint8)
    if bands > 1:
        base[1] = np.clip(40 + yy * 90 / max(height - 1, 1), 0, 255).astype(np.uint8)
    if bands > 2:
        base[2] = np.clip(55 + (xx + yy) * 60 / max(width + height - 2, 1), 0, 255).astype(np.uint8)
    for band in range(3, bands):
        base[band] = np.clip((base[0].astype(np.uint16) + base[1].astype(np.uint16)) // 2, 0, 255).astype(np.uint8)

    rect_x0, rect_x1 = width // 5, width // 5 + max(width // 4, 8)
    rect_y0, rect_y1 = height // 4, height // 4 + max(height // 5, 8)
    base[0, rect_y0:rect_y1, rect_x0:rect_x1] = 220
    if bands > 1:
        base[1, rect_y0:rect_y1, rect_x0:rect_x1] = 70
    if bands > 2:
        base[2, rect_y0:rect_y1, rect_x0:rect_x1] = 60

    radius = max(min(width, height) // 9, 5)
    cx = width * (2 if not changed else 3) // 4
    cy = height * 2 // 3
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    base[0, circle] = 60
    if bands > 1:
        base[1, circle] = 190
    if bands > 2:
        base[2, circle] = 230

    if changed:
        change_x0, change_x1 = width // 2, min(width, width // 2 + max(width // 6, 6))
        change_y0, change_y1 = height // 8, min(height, height // 8 + max(height // 6, 6))
        base[:, change_y0:change_y1, change_x0:change_x1] = 245
    return base


def main() -> None:
    """Create and write the synthetic raster."""
    args = parse_args()
    array = create_visual_synthetic_array(width=args.width, height=args.height, bands=args.bands, changed=args.changed)
    profile = profile_for_array(array, crs=args.crs, transform=DEFAULT_TRANSFORM)
    info = write_geotiff(Path(args.output), array, profile)
    print(f"wrote {info['path']}")
    print(f"size={info['width']}x{info['height']} bands={info['count']} crs={info['crs']} driver={info['driver']}")
    for warning in info.get("warnings", []):
        print(f"warning: {warning}")


if __name__ == "__main__":
    main()
