from __future__ import annotations

from pathlib import Path
from typing import Any

from rs_service.core.manifest import read_json, write_json, write_manifest
from rs_service.core.raster import read_raster
from rs_service.pipelines.base import flag, prepare_output_dir


def quality_check_result(
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    out_dir = prepare_output_dir(output_dir, task="quality_check")
    quality_flags: list[dict[str, Any]] = []
    inputs: dict[str, Any] = {"input_path": str(input_path) if input_path else None}
    source_manifest: dict[str, Any] | None = None
    outputs_to_check: dict[str, Any] = {}

    if manifest_path:
        source_manifest = read_json(manifest_path)
        inputs["source_manifest"] = str(manifest_path)
        outputs_to_check = source_manifest.get("outputs", {})
        quality_flags.extend(source_manifest.get("quality_flags", []))
        if source_manifest.get("stats", {}).get("count") == 0:
            quality_flags.append(flag("empty_result", "Source manifest reports zero records.", "warning"))
    elif input_path:
        outputs_to_check = {"input_path": str(input_path)}
    else:
        raise ValueError("input_path or manifest_path is required")

    missing = []
    for name, value in outputs_to_check.items():
        if value and not Path(str(value)).exists():
            missing.append(name)
    if missing:
        quality_flags.append(flag("missing_outputs", f"Missing output files: {', '.join(missing)}.", "error"))

    target_raster = input_path
    if not target_raster and source_manifest:
        for value in source_manifest.get("outputs", {}).values():
            suffix = Path(str(value)).suffix.lower()
            if suffix in {".tif", ".tiff"}:
                target_raster = value
                break
    if target_raster:
        try:
            _, _, info = read_raster(target_raster)
            inputs["raster"] = info.to_dict()
            if not info.crs:
                quality_flags.append(flag("crs_missing", "Raster CRS is missing.", "warning"))
            if info.fallback_container:
                quality_flags.append(flag("fallback_raster_io", "Rasterio unavailable; fallback raster container was used.", "info"))
        except Exception as exc:
            quality_flags.append(flag("raster_read_failed", f"Could not read raster: {exc}", "error"))

    unique_flags = []
    seen = set()
    for item in quality_flags:
        key = (item.get("code"), item.get("message"))
        if key not in seen:
            seen.add(key)
            unique_flags.append(item)
    payload = {"quality_flags": unique_flags, "checked_outputs": outputs_to_check}
    quality_path = write_json(out_dir / "quality.json", payload)
    return write_manifest(
        task="quality_check",
        output_dir=out_dir,
        inputs=inputs,
        outputs={"quality_json": quality_path},
        parameters={},
        stats={"flag_count": len(unique_flags), "error_count": sum(1 for item in unique_flags if item.get("severity") == "error")},
        quality_flags=unique_flags,
        model={"id": "quality-checker", "backend": "rules", "framework": "python"},
    )
