from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rs_service.core.manifest import read_json, write_json, write_manifest
from rs_service.core.raster import read_raster
from rs_service.pipelines.base import flag, prepare_output_dir


def _raster_stats(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    data, _, info = read_raster(path)
    stats: dict[str, Any] = {"raster": info.to_dict(), "bands": []}
    flags = []
    for band_index, band in enumerate(data, start=1):
        finite = band[np.isfinite(band)]
        band_stats = {
            "band": band_index,
            "min": float(np.min(finite)) if finite.size else 0.0,
            "max": float(np.max(finite)) if finite.size else 0.0,
            "mean": float(np.mean(finite)) if finite.size else 0.0,
            "std": float(np.std(finite)) if finite.size else 0.0,
            "valid_pixels": int(finite.size),
            "total_pixels": int(band.size),
        }
        stats["bands"].append(band_stats)
    if not info.crs:
        flags.append(flag("crs_missing", "Raster CRS is missing.", "warning"))
    if info.fallback_container:
        flags.append(flag("fallback_raster_io", "Rasterio unavailable; fallback raster container was used.", "info"))
    return stats, flags, {"image": str(path), "raster": info.to_dict()}


def _geojson_stats(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    features = payload.get("features", [])
    scores = [
        float(feature.get("properties", {}).get("score"))
        for feature in features
        if feature.get("properties", {}).get("score") is not None
    ]
    stats = {
        "feature_count": len(features),
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "labels": sorted({str(feature.get("properties", {}).get("label", "")) for feature in features}),
    }
    flags = []
    if not features:
        flags.append(flag("empty_vector", "Vector result contains no features.", "warning"))
    return stats, flags, {"vector": str(path)}


def calculate_statistics(
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    manifest_path: str | Path | None = None,
    zones_path: str | Path | None = None,
) -> dict[str, Any]:
    out_dir = prepare_output_dir(output_dir, task="statistics")
    if manifest_path:
        manifest = read_json(manifest_path)
        outputs = manifest.get("outputs", {})
        target = (
            outputs.get("geojson")
            or outputs.get("mask_geotiff")
            or outputs.get("super_resolved_geotiff")
            or next((value for value in outputs.values() if Path(str(value)).suffix.lower() in {".tif", ".tiff", ".geojson"}), None)
            or next(iter(outputs.values()), None)
            or input_path
        )
    else:
        manifest = None
        target = input_path
    if target is None:
        raise ValueError("input_path or manifest_path is required")

    suffix = Path(str(target)).suffix.lower()
    if suffix in {".geojson", ".json"} and "geojson" in Path(str(target)).name.lower():
        stats, quality_flags, inputs = _geojson_stats(target)
    else:
        stats, quality_flags, inputs = _raster_stats(target)

    if zones_path:
        quality_flags.append(
            flag(
                "zonal_stats_not_run",
                "Zonal statistics require rasterstats/geopandas and are reserved for the real dependency environment.",
                "info",
                zones_path=str(zones_path),
            )
        )
    stats_path = write_json(out_dir / "stats.json", stats)
    return write_manifest(
        task="statistics",
        output_dir=out_dir,
        inputs={**inputs, "source_manifest": str(manifest_path) if manifest_path else None, "zones": str(zones_path) if zones_path else None},
        outputs={"stats_json": stats_path},
        parameters={},
        stats=stats,
        quality_flags=quality_flags,
        model={"id": "statistics", "backend": "numpy/geopandas", "framework": "rasterstats"},
    )
