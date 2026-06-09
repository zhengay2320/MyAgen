from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:  # pragma: no cover - exercised when geospatial deps are installed
    import rasterio
    from rasterio.transform import Affine
except Exception:  # pragma: no cover - local fallback path is covered
    rasterio = None
    Affine = None


DEFAULT_CRS = "EPSG:3857"
DEFAULT_TRANSFORM = (1.0, 0.0, 500000.0, 0.0, -1.0, 4100000.0)


@dataclass(frozen=True)
class RasterInfo:
    path: str
    width: int
    height: int
    count: int
    dtype: str
    crs: str | None
    transform: tuple[float, float, float, float, float, float]
    bounds: tuple[float, float, float, float]
    nodata: float | int | None = None
    driver: str = "GTiff"
    fallback_container: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _affine_to_tuple(transform: Any) -> tuple[float, float, float, float, float, float]:
    if transform is None:
        return DEFAULT_TRANSFORM
    if Affine is not None and isinstance(transform, Affine):
        return (transform.a, transform.b, transform.c, transform.d, transform.e, transform.f)
    values = tuple(float(v) for v in transform)
    if len(values) == 6:
        return values
    if len(values) >= 9:
        return (values[0], values[1], values[2], values[3], values[4], values[5])
    raise ValueError(f"Unsupported transform with {len(values)} values")


def tuple_to_affine(transform: Iterable[float]) -> Any:
    values = tuple(float(v) for v in transform)
    if Affine is not None:
        return Affine(values[0], values[1], values[2], values[3], values[4], values[5])
    return values


def pixel_to_geo(
    transform: Iterable[float],
    x: float,
    y: float,
) -> tuple[float, float]:
    a, b, c, d, e, f = tuple(float(v) for v in transform)
    return (a * x + b * y + c, d * x + e * y + f)


def geo_bounds(
    width: int,
    height: int,
    transform: Iterable[float],
) -> tuple[float, float, float, float]:
    corners = [
        pixel_to_geo(transform, 0, 0),
        pixel_to_geo(transform, width, 0),
        pixel_to_geo(transform, width, height),
        pixel_to_geo(transform, 0, height),
    ]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def profile_from_array(
    array: np.ndarray,
    crs: str | None = DEFAULT_CRS,
    transform: Iterable[float] = DEFAULT_TRANSFORM,
    nodata: float | int | None = None,
    dtype: str | None = None,
) -> dict[str, Any]:
    data = ensure_chw(array)
    return {
        "driver": "GTiff",
        "height": int(data.shape[1]),
        "width": int(data.shape[2]),
        "count": int(data.shape[0]),
        "dtype": dtype or str(data.dtype),
        "crs": crs,
        "transform": tuple(float(v) for v in transform),
        "nodata": nodata,
    }


def ensure_chw(array: np.ndarray) -> np.ndarray:
    data = np.asarray(array)
    if data.ndim == 2:
        return data[np.newaxis, :, :]
    if data.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D raster array, got shape {data.shape}")
    return data


def _profile_to_info(path: Path, profile: dict[str, Any], fallback: bool = False) -> RasterInfo:
    transform = _affine_to_tuple(profile.get("transform"))
    width = int(profile["width"])
    height = int(profile["height"])
    return RasterInfo(
        path=str(path),
        width=width,
        height=height,
        count=int(profile["count"]),
        dtype=str(profile["dtype"]),
        crs=str(profile["crs"]) if profile.get("crs") is not None else None,
        transform=transform,
        bounds=geo_bounds(width, height, transform),
        nodata=profile.get("nodata"),
        driver=str(profile.get("driver", "GTiff")),
        fallback_container=fallback,
    )


def read_raster(path: str | Path) -> tuple[np.ndarray, dict[str, Any], RasterInfo]:
    raster_path = Path(path)
    if rasterio is not None:
        try:  # pragma: no cover - requires rasterio
            with rasterio.open(raster_path) as dataset:
                data = dataset.read()
                profile = dict(dataset.profile)
                profile["transform"] = _affine_to_tuple(dataset.transform)
                profile["crs"] = dataset.crs.to_string() if dataset.crs else None
                info = _profile_to_info(raster_path, profile, fallback=False)
                return data, profile, info
        except Exception:
            # Fall through to the local fallback so tests can read pseudo rasters.
            pass

    with raster_path.open("rb") as handle:
        loaded = np.load(handle, allow_pickle=False)
        data = loaded["data"]
        profile = json.loads(str(loaded["profile"].item()))
    info = _profile_to_info(raster_path, profile, fallback=True)
    return data, profile, info


def write_raster(
    path: str | Path,
    array: np.ndarray,
    profile: dict[str, Any] | None = None,
    *,
    crs: str | None = None,
    transform: Iterable[float] | None = None,
    nodata: float | int | None = None,
    dtype: str | None = None,
) -> RasterInfo:
    raster_path = Path(path)
    raster_path.parent.mkdir(parents=True, exist_ok=True)
    data = ensure_chw(np.asarray(array))
    out_profile = dict(profile or profile_from_array(data))
    out_profile.update(
        height=int(data.shape[1]),
        width=int(data.shape[2]),
        count=int(data.shape[0]),
        dtype=dtype or str(data.dtype),
    )
    if crs is not None:
        out_profile["crs"] = crs
    if transform is not None:
        out_profile["transform"] = tuple(float(v) for v in transform)
    if "transform" not in out_profile or out_profile["transform"] is None:
        out_profile["transform"] = DEFAULT_TRANSFORM
    if "crs" not in out_profile:
        out_profile["crs"] = DEFAULT_CRS
    if nodata is not None:
        out_profile["nodata"] = nodata
    out_profile.setdefault("driver", "GTiff")

    if rasterio is not None:  # pragma: no cover - requires rasterio
        rio_profile = dict(out_profile)
        rio_profile["transform"] = tuple_to_affine(out_profile["transform"])
        with rasterio.open(raster_path, "w", **rio_profile) as dataset:
            dataset.write(data.astype(out_profile["dtype"], copy=False))
        return _profile_to_info(raster_path, out_profile, fallback=False)

    serializable_profile = dict(out_profile)
    serializable_profile["transform"] = tuple(float(v) for v in out_profile["transform"])
    with raster_path.open("wb") as handle:
        np.savez_compressed(handle, data=data.astype(out_profile["dtype"], copy=False), profile=json.dumps(serializable_profile))
    return _profile_to_info(raster_path, serializable_profile, fallback=True)


def inspect_raster(path: str | Path) -> dict[str, Any]:
    _, _, info = read_raster(path)
    result = info.to_dict()
    result["pixel_size"] = [abs(info.transform[0]), abs(info.transform[4])]
    result["area_per_pixel"] = abs(info.transform[0] * info.transform[4] - info.transform[1] * info.transform[3])
    result["is_georeferenced"] = info.crs is not None
    return result


def update_transform_for_super_resolution(
    transform: Iterable[float],
    scale: int | float,
) -> tuple[float, float, float, float, float, float]:
    if scale <= 0:
        raise ValueError("scale must be positive")
    a, b, c, d, e, f = tuple(float(v) for v in transform)
    return (a / scale, b / scale, c, d / scale, e / scale, f)


def create_synthetic_array(
    width: int = 128,
    height: int = 96,
    bands: int = 4,
    *,
    changed: bool = False,
) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    data = []
    for band in range(bands):
        plane = ((xx * (band + 1) + yy * (band + 2)) % 255).astype(np.float32)
        plane += 20 * math.sin((band + 1) * 0.2)
        data.append(plane)
    array = np.stack(data, axis=0)
    if bands >= 4:
        array[3] = np.maximum(array[3], array[2] + 25)
    if changed:
        y0, y1 = height // 3, min(height, height // 3 + height // 5)
        x0, x1 = width // 3, min(width, width // 3 + width // 5)
        array[:, y0:y1, x0:x1] += 80
    return np.clip(array, 0, 255).astype(np.uint8)
