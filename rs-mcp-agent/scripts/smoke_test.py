from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rs_service import services
from rs_service.core.raster import DEFAULT_TRANSFORM, create_synthetic_array, profile_from_array, write_raster


def _write_input(path: Path, changed: bool = False) -> str:
    array = create_synthetic_array(width=120, height=96, bands=4, changed=changed)
    profile = profile_from_array(array, transform=DEFAULT_TRANSFORM)
    write_raster(path, array, profile)
    return str(path)


def main() -> None:
    root = Path("workspace/smoke")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    before = _write_input(root / "before.tif", changed=False)
    after = _write_input(root / "after.tif", changed=True)

    inspect = services.inspect_raster(before)
    if inspect["width"] <= 0 or inspect["height"] <= 0:
        raise RuntimeError("inspect_raster returned invalid dimensions")
    preflight = services.preflight_plan(before, task="object_detection", tile_size=48, overlap=12)
    if preflight["tile_count"] <= 0:
        raise RuntimeError("preflight_plan returned no tiles")

    manifests = []
    manifests.append(services.run_object_detection(before, str(root / "object_detection"), tile_size=48, overlap=12))
    manifests.append(services.run_semantic_segmentation(before, str(root / "semantic_segmentation"), tile_size=48, overlap=12))
    manifests.append(services.run_instance_segmentation(before, str(root / "instance_segmentation"), tile_size=48, overlap=12))
    manifests.append(services.run_oriented_detection(before, str(root / "oriented_detection"), tile_size=48, overlap=12))
    manifests.append(services.run_change_detection(before, after, str(root / "change_detection"), tile_size=48, overlap=12, threshold=0.25))
    manifests.append(services.run_super_resolution(before, str(root / "super_resolution"), tile_size=48, overlap=12, scale=2))
    manifests.append(services.run_spectral_indices(before, str(root / "spectral_indices"), tile_size=48, overlap=12))

    analyzed_manifests = []
    for manifest in manifests:
        analyzed = services.analyze_job(manifest["job_id"])
        reported = services.generate_job_report(analyzed["job_id"])
        _validate_manifest(reported)
        analyzed_manifests.append(reported)
    manifests = analyzed_manifests

    print(
        json.dumps(
            {
                "ok": True,
                "inspect": {"width": inspect["width"], "height": inspect["height"], "count": inspect["count"]},
                "preflight": {"tile_count": preflight["tile_count"], "tile_size": preflight["tile_size"], "overlap": preflight["overlap"]},
                "manifest_paths": [item["manifest_path"] for item in manifests],
            },
            indent=2,
        )
    )


def _validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate the end-to-end output contract for one fake job."""
    job_id = manifest.get("job_id", "<missing>")
    if manifest.get("status") != "success":
        raise RuntimeError(f"Unexpected manifest status for {job_id}: {manifest.get('status')}")
    manifest_path = Path(str(manifest.get("manifest_path", "")))
    if not manifest_path.exists():
        raise RuntimeError(f"Missing manifest.json for {job_id}: {manifest_path}")
    outputs = manifest.get("outputs", {})
    output_values = [Path(str(value)) for value in outputs.values() if isinstance(value, str)]
    if not output_values or not any(path.exists() for path in output_values):
        raise RuntimeError(f"No existing output files for {job_id}")
    if not any(_is_core_output(path) for path in output_values):
        raise RuntimeError(f"Missing preview/raster/vector output for {job_id}")
    statistics = manifest.get("statistics") or manifest.get("stats")
    if not isinstance(statistics, dict) or not statistics:
        raise RuntimeError(f"Missing statistics for {job_id}")
    if not manifest.get("conclusion"):
        raise RuntimeError(f"Missing conclusion for {job_id}")
    report = outputs.get("report") or outputs.get("report_md")
    if not report or not Path(str(report)).exists():
        raise RuntimeError(f"Missing report.md for {job_id}")


def _is_core_output(path: Path) -> bool:
    """Return whether a path is one of the expected user-facing result types."""
    suffix = path.suffix.lower()
    return suffix in {".png", ".tif", ".tiff", ".geojson", ".gpkg", ".json", ".npy"}


if __name__ == "__main__":
    main()
