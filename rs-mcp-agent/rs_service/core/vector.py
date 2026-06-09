from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:  # pragma: no cover - requires geospatial deps
    import geopandas as gpd
    from shapely.geometry import shape
except Exception:  # pragma: no cover - fallback is covered
    gpd = None
    shape = None


def feature_collection(features: list[dict[str, Any]], crs: str | None = None) -> dict[str, Any]:
    collection: dict[str, Any] = {
        "type": "FeatureCollection",
        "features": features,
    }
    if crs:
        collection["crs"] = {"type": "name", "properties": {"name": crs}}
    return collection


def write_geojson(path: str | Path, features: list[dict[str, Any]], crs: str | None = None) -> str:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(feature_collection(features, crs), indent=2), encoding="utf-8")
    return str(out_path)


def write_gpkg(path: str | Path, features: list[dict[str, Any]], crs: str | None = None, layer: str = "results") -> str:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if gpd is not None and shape is not None and features:  # pragma: no cover - requires geospatial deps
        geometries = [shape(feature["geometry"]) for feature in features]
        properties = [_serialize_properties(feature.get("properties", {})) for feature in features]
        frame = gpd.GeoDataFrame(properties, geometry=geometries, crs=crs)
        frame.to_file(out_path, layer=layer, driver="GPKG")
    else:
        fallback_payload = feature_collection(features, crs)
        fallback_payload["format_note"] = "Install geopandas/fiona to write a standards-compliant GPKG."
        out_path.write_text(json.dumps(fallback_payload, indent=2), encoding="utf-8")
    return str(out_path)


def _serialize_properties(properties: dict[str, Any]) -> dict[str, Any]:
    serialized = {}
    for key, value in properties.items():
        if isinstance(value, (dict, list, tuple)):
            serialized[key] = json.dumps(value)
        else:
            serialized[key] = value
    return serialized
