from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from rs_service.core.raster import (
    DEFAULT_CRS,
    DEFAULT_TRANSFORM,
    ensure_chw,
    geo_bounds,
    profile_from_array,
    read_raster,
    tuple_to_affine,
    write_raster,
)

try:  # pragma: no cover - exercised when rasterio is installed
    import rasterio
    from rasterio.windows import Window
except Exception:  # pragma: no cover - local fallback path is covered
    rasterio = None
    Window = None


WindowLike = Any


def validate_raster_path(path: str | Path) -> Path:
    """Validate that a raster path exists, is readable, and is not a directory."""
    raster_path = Path(path)
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster does not exist: {raster_path}")
    if raster_path.is_dir():
        raise IsADirectoryError(f"Raster path is a directory: {raster_path}")
    try:
        with raster_path.open("rb"):
            pass
    except OSError as exc:
        raise PermissionError(f"Raster is not readable: {raster_path}") from exc
    return raster_path


def inspect_raster(path: str | Path) -> dict[str, Any]:
    """Inspect raster metadata without failing on missing CRS."""
    raster_path = validate_raster_path(path)
    warnings: list[str] = []
    if rasterio is not None:
        try:  # pragma: no cover - requires rasterio
            with rasterio.open(raster_path) as dataset:
                transform = _transform_to_tuple(dataset.transform)
                crs = dataset.crs.to_string() if dataset.crs else None
                if crs is None:
                    warnings.append("Raster has no CRS.")
                return {
                    "path": str(raster_path),
                    "width": int(dataset.width),
                    "height": int(dataset.height),
                    "count": int(dataset.count),
                    "dtype": str(dataset.dtypes[0]) if dataset.dtypes else "",
                    "crs": crs,
                    "bounds": tuple(float(v) for v in dataset.bounds),
                    "transform": transform,
                    "resolution": (abs(float(dataset.res[0])), abs(float(dataset.res[1]))),
                    "nodata": dataset.nodata,
                    "driver": dataset.driver,
                    "warnings": warnings,
                }
        except Exception:
            # Some tests use the local fallback raster container with a .tif suffix.
            pass

    _, profile, info = read_raster(raster_path)
    if info.crs is None:
        warnings.append("Raster has no CRS.")
    return {
        "path": str(raster_path),
        "width": info.width,
        "height": info.height,
        "count": info.count,
        "dtype": info.dtype,
        "crs": info.crs,
        "bounds": info.bounds,
        "transform": info.transform,
        "resolution": (abs(info.transform[0]), abs(info.transform[4])),
        "nodata": profile.get("nodata"),
        "driver": info.driver,
        "warnings": warnings,
        "fallback_container": info.fallback_container,
    }


def read_window(path: str | Path, window: WindowLike, bands: Sequence[int] | int | None = None) -> np.ndarray:
    """Read a raster window as a CHW array, optionally selecting 1-based bands."""
    raster_path = validate_raster_path(path)
    normalized_bands = _normalize_bands(bands)
    if rasterio is not None:
        try:  # pragma: no cover - requires rasterio
            with rasterio.open(raster_path) as dataset:
                rio_window = _to_rasterio_window(window)
                data = dataset.read(indexes=normalized_bands, window=rio_window)
                return ensure_chw(data)
        except Exception:
            pass

    data, _, _ = read_raster(raster_path)
    x_off, y_off, width, height = _window_to_offsets(window)
    sliced = data[:, y_off : y_off + height, x_off : x_off + width]
    if normalized_bands is not None:
        if isinstance(normalized_bands, int):
            sliced = sliced[normalized_bands - 1 : normalized_bands]
        else:
            indices = [band - 1 for band in normalized_bands]
            sliced = sliced[indices]
    return ensure_chw(sliced)


def write_geotiff(
    path: str | Path,
    array: np.ndarray,
    reference_profile: dict[str, Any],
    transform: Iterable[float] | None = None,
    crs: str | None = None,
    nodata: float | int | None = None,
) -> dict[str, Any]:
    """Write an array to GeoTIFF, preserving reference CRS, transform, and nodata by default."""
    out_path = Path(path)
    data = ensure_chw(array)
    profile = dict(reference_profile)
    profile.update(
        driver="GTiff",
        height=int(data.shape[1]),
        width=int(data.shape[2]),
        count=int(data.shape[0]),
        dtype=str(data.dtype),
        transform=tuple(float(v) for v in (transform or profile.get("transform") or DEFAULT_TRANSFORM)),
        crs=crs if crs is not None else profile.get("crs", DEFAULT_CRS),
    )
    if nodata is not None:
        profile["nodata"] = nodata
    elif "nodata" not in profile:
        profile["nodata"] = reference_profile.get("nodata")

    if rasterio is not None:  # pragma: no cover - requires rasterio
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rio_profile = dict(profile)
        rio_profile["transform"] = tuple_to_affine(profile["transform"])
        with rasterio.open(out_path, "w", **rio_profile) as dataset:
            dataset.write(data)
        return inspect_raster(out_path)

    info = write_raster(out_path, data, profile, crs=profile.get("crs"), transform=profile.get("transform"), nodata=profile.get("nodata"), dtype=str(data.dtype))
    result = info.to_dict()
    result["resolution"] = (abs(info.transform[0]), abs(info.transform[4]))
    result["warnings"] = ["Rasterio is not installed; wrote fallback raster container instead of a standards-compliant GeoTIFF."]
    return result


def _normalize_bands(bands: Sequence[int] | int | None) -> Sequence[int] | int | None:
    """Normalize band selection and keep rasterio's 1-based indexing convention."""
    if bands is None:
        return None
    if isinstance(bands, int):
        if bands < 1:
            raise ValueError("bands must use 1-based indexes")
        return bands
    normalized = [int(band) for band in bands]
    if any(band < 1 for band in normalized):
        raise ValueError("bands must use 1-based indexes")
    return normalized


def _window_to_offsets(window: WindowLike) -> tuple[int, int, int, int]:
    """Convert a window-like object into x offset, y offset, width, and height."""
    if isinstance(window, dict):
        return (
            int(window.get("x_off", window.get("col_off", window.get("x0", 0)))),
            int(window.get("y_off", window.get("row_off", window.get("y0", 0)))),
            int(window.get("width", window.get("w", 0))),
            int(window.get("height", window.get("h", 0))),
        )
    if all(hasattr(window, name) for name in ["x_off", "y_off", "width", "height"]):
        return int(window.x_off), int(window.y_off), int(window.width), int(window.height)
    if all(hasattr(window, name) for name in ["col_off", "row_off", "width", "height"]):
        return int(window.col_off), int(window.row_off), int(window.width), int(window.height)
    if isinstance(window, tuple) and len(window) == 4:
        return tuple(int(v) for v in window)  # type: ignore[return-value]
    raise ValueError(f"Unsupported window: {window!r}")


def _to_rasterio_window(window: WindowLike) -> Any:
    """Convert a local window description to `rasterio.windows.Window`."""
    x_off, y_off, width, height = _window_to_offsets(window)
    if Window is None:
        return (x_off, y_off, width, height)
    return Window(col_off=x_off, row_off=y_off, width=width, height=height)


def _transform_to_tuple(transform: Any) -> tuple[float, float, float, float, float, float]:
    """Convert Rasterio/Affine or tuple transforms to the six-value GDAL form."""
    if hasattr(transform, "a"):
        return (float(transform.a), float(transform.b), float(transform.c), float(transform.d), float(transform.e), float(transform.f))
    values = tuple(float(v) for v in transform)
    if len(values) == 6:
        return values
    if len(values) >= 9:
        return (values[0], values[1], values[2], values[3], values[4], values[5])
    raise ValueError(f"Unsupported transform: {transform!r}")


def bounds_from_profile(profile: dict[str, Any]) -> tuple[float, float, float, float]:
    """Calculate bounds from a raster profile containing width, height, and transform."""
    return geo_bounds(int(profile["width"]), int(profile["height"]), profile.get("transform", DEFAULT_TRANSFORM))


def profile_for_array(array: np.ndarray, crs: str | None = DEFAULT_CRS, transform: Iterable[float] = DEFAULT_TRANSFORM) -> dict[str, Any]:
    """Create a GeoTIFF profile for a 2D or CHW array."""
    return profile_from_array(array, crs=crs, transform=transform)
