from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from rs_service.core.raster import read_raster


def segmentation_statistics(mask_path: str | Path, class_map: dict[str, Any] | str | None, pixel_area: float) -> dict[str, Any]:
    """Calculate class area, class ratios, and patch statistics from a segmentation mask."""
    mask = _read_single_band(mask_path)
    class_names = _load_class_map(class_map)
    total_pixels = int(mask.size)
    total_area_m2 = total_pixels * pixel_area
    classes: dict[str, Any] = {}
    connected_components_total = 0
    small_patch_count = 0
    max_patch_area_m2 = 0.0
    for value in sorted(int(v) for v in np.unique(mask)):
        binary = mask == value
        pixel_count = int(binary.sum())
        components = _connected_component_sizes(binary)
        connected_components_total += len(components)
        component_areas = [size * pixel_area for size in components]
        small_patch_count += sum(1 for area in component_areas if 0 < area < 9 * pixel_area)
        max_patch_area_m2 = max([max_patch_area_m2, *component_areas] or [max_patch_area_m2])
        classes[str(value)] = {
            "class_id": value,
            "class_name": class_names.get(value, f"class_{value}"),
            "pixel_count": pixel_count,
            "area_m2": pixel_count * pixel_area,
            "area_km2": pixel_count * pixel_area / 1_000_000.0,
            "ratio": pixel_count / total_pixels if total_pixels else 0.0,
            "connected_components": len(components),
        }
    return {
        "type": "segmentation",
        "mask_path": str(mask_path),
        "pixel_area_m2": pixel_area,
        "total_pixels": total_pixels,
        "total_area_m2": total_area_m2,
        "total_area_km2": total_area_m2 / 1_000_000.0,
        "classes": classes,
        "class_ratios": {key: value["ratio"] for key, value in classes.items()},
        "connected_component_count": connected_components_total,
        "small_patch_count": small_patch_count,
        "max_patch_area_m2": max_patch_area_m2,
        "dominant_class": max(classes.items(), key=lambda item: item[1]["ratio"])[0] if classes else None,
    }


def detection_statistics(vector_path: str | Path) -> dict[str, Any]:
    """Calculate detection counts, confidence statistics, and density from vector outputs."""
    features = _load_features(vector_path)
    scores = _feature_scores(features)
    labels = sorted({str(feature.get("properties", {}).get("label", "unknown")) for feature in features})
    total_area_m2 = sum(max(_polygon_area_m2(feature.get("geometry")), 0.0) for feature in features)
    area_km2 = total_area_m2 / 1_000_000.0
    return {
        "type": "detection",
        "vector_path": str(vector_path),
        "target_count": len(features),
        "labels": labels,
        "mean_confidence": mean(scores) if scores else 0.0,
        "median_confidence": median(scores) if scores else 0.0,
        "low_confidence_ratio": _low_confidence_ratio(scores),
        "target_density_per_km2": len(features) / area_km2 if area_km2 > 0 else 0.0,
        "feature_area_m2": total_area_m2,
        "feature_area_km2": area_km2,
    }


def instance_statistics(instance_vector_path: str | Path) -> dict[str, Any]:
    """Calculate instance segmentation statistics from vectorized instances."""
    stats = detection_statistics(instance_vector_path)
    stats["type"] = "instance_segmentation"
    stats["instance_count"] = stats["target_count"]
    stats["small_patch_count"] = 0
    return stats


def change_statistics(change_mask_path: str | Path, pixel_area: float) -> dict[str, Any]:
    """Calculate changed area, changed ratio, and patch statistics from a change mask."""
    mask = _read_single_band(change_mask_path)
    changed = mask > 0
    changed_pixels = int(changed.sum())
    total_pixels = int(mask.size)
    components = _connected_component_sizes(changed)
    component_areas = [size * pixel_area for size in components]
    return {
        "type": "change_detection",
        "mask_path": str(change_mask_path),
        "pixel_area_m2": pixel_area,
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "changed_area_m2": changed_pixels * pixel_area,
        "changed_area_km2": changed_pixels * pixel_area / 1_000_000.0,
        "change_area_ratio": changed_pixels / total_pixels if total_pixels else 0.0,
        "connected_component_count": len(components),
        "small_patch_count": sum(1 for area in component_areas if 0 < area < 9 * pixel_area),
        "max_patch_area_m2": max(component_areas) if component_areas else 0.0,
    }


def super_resolution_statistics(input_path: str | Path, output_path: str | Path, scale: int) -> dict[str, Any]:
    """Validate super-resolution output dimensions, transform, and resolution."""
    _, _, input_info = read_raster(input_path)
    _, _, output_info = read_raster(output_path)
    expected_width = input_info.width * scale
    expected_height = input_info.height * scale
    expected_transform = (
        input_info.transform[0] / scale,
        input_info.transform[1] / scale,
        input_info.transform[2],
        input_info.transform[3] / scale,
        input_info.transform[4] / scale,
        input_info.transform[5],
    )
    transform_delta = max(abs(a - b) for a, b in zip(output_info.transform, expected_transform))
    return {
        "type": "super_resolution",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "scale": scale,
        "input_resolution": [abs(input_info.transform[0]), abs(input_info.transform[4])],
        "output_resolution": [abs(output_info.transform[0]), abs(output_info.transform[4])],
        "input_shape": [input_info.height, input_info.width],
        "output_shape": [output_info.height, output_info.width],
        "expected_output_shape": [expected_height, expected_width],
        "shape_matches_scale": output_info.width == expected_width and output_info.height == expected_height,
        "expected_transform": expected_transform,
        "output_transform": output_info.transform,
        "transform_delta": transform_delta,
        "transform_matches_scale": transform_delta < 1e-9,
        "crs_matches": input_info.crs == output_info.crs,
    }


def spectral_index_statistics(index_raster_path: str | Path) -> dict[str, Any]:
    """Calculate summary statistics for one spectral index raster."""
    data = _read_single_band(index_raster_path).astype(np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        min_value = max_value = mean_value = median_value = std_value = 0.0
    else:
        min_value = float(np.min(finite))
        max_value = float(np.max(finite))
        mean_value = float(np.mean(finite))
        median_value = float(np.median(finite))
        std_value = float(np.std(finite))
    return {
        "type": "spectral_index",
        "index_raster_path": str(index_raster_path),
        "min": min_value,
        "max": max_value,
        "mean": mean_value,
        "median": median_value,
        "std": std_value,
        "valid_pixels": int(finite.size),
        "total_pixels": int(data.size),
    }


def _read_single_band(path: str | Path) -> np.ndarray:
    """Read a raster and return its first band with clear missing-file errors."""
    raster_path = Path(path)
    if not raster_path.exists():
        raise FileNotFoundError(f"Missing raster file: {raster_path}")
    data, _, _ = read_raster(raster_path)
    if data.ndim != 3 or data.shape[0] < 1:
        raise ValueError(f"Raster has no readable bands: {raster_path}")
    return data[0]


def _load_features(path: str | Path) -> list[dict[str, Any]]:
    """Load GeoJSON-style features from GeoJSON or JSON vector output."""
    vector_path = Path(path)
    if not vector_path.exists():
        raise FileNotFoundError(f"Missing vector file: {vector_path}")
    payload = json.loads(vector_path.read_text(encoding="utf-8"))
    if "features" in payload:
        return list(payload.get("features", []))
    if "records" in payload:
        return [
            {"type": "Feature", "properties": record, "geometry": record.get("geometry")}
            for record in payload.get("records", [])
        ]
    return []


def _feature_scores(features: list[dict[str, Any]]) -> list[float]:
    """Extract confidence scores from vector features."""
    scores: list[float] = []
    for feature in features:
        score = feature.get("properties", {}).get("score")
        if score is not None:
            scores.append(float(score))
    return scores


def _low_confidence_ratio(scores: list[float], threshold: float = 0.5) -> float:
    """Return the fraction of scores below a confidence threshold."""
    return sum(1 for score in scores if score < threshold) / len(scores) if scores else 0.0


def _polygon_area_m2(geometry: dict[str, Any] | None) -> float:
    """Calculate planar polygon area using the shoelace formula."""
    if not geometry or geometry.get("type") != "Polygon":
        return 0.0
    rings = geometry.get("coordinates") or []
    if not rings:
        return 0.0
    coords = rings[0]
    if len(coords) < 4:
        return 0.0
    area = 0.0
    for first, second in zip(coords, coords[1:]):
        area += float(first[0]) * float(second[1]) - float(second[0]) * float(first[1])
    return abs(area) / 2.0


def _connected_component_sizes(mask: np.ndarray) -> list[int]:
    """Return connected component sizes for a boolean mask using 4-neighbor fill."""
    binary = np.asarray(mask, dtype=bool)
    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    sizes: list[int] = []
    for y in range(height):
        for x in range(width):
            if not binary[y, x] or visited[y, x]:
                continue
            size = 0
            stack = [(x, y)]
            visited[y, x] = True
            while stack:
                cx, cy = stack.pop()
                size += 1
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < width and 0 <= ny < height and binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((nx, ny))
            sizes.append(size)
    return sizes


def _load_class_map(class_map: dict[str, Any] | str | None) -> dict[int, str]:
    """Load class names from a dict or YAML-like class map path."""
    if class_map is None:
        return {}
    if isinstance(class_map, str):
        path = Path(class_map)
        if not path.exists():
            return {}
        text = path.read_text(encoding="utf-8")
        names: dict[int, str] = {}
        current_id: int | None = None
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.endswith(":") and stripped[:-1].isdigit():
                current_id = int(stripped[:-1])
            elif stripped.startswith("name:") and current_id is not None:
                names[current_id] = stripped.split(":", 1)[1].strip().strip('"')
        return names
    classes = class_map.get("classes", class_map)
    names = {}
    for key, value in classes.items():
        class_id = int(key)
        names[class_id] = value.get("name", f"class_{class_id}") if isinstance(value, dict) else str(value)
    return names
