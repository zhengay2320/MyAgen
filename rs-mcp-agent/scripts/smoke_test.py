from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

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

    manifests = []
    manifests.append(services.run_object_detection(before, str(root / "object_detection"), tile_size=48, overlap=12))
    manifests.append(services.run_oriented_detection(before, str(root / "oriented_detection"), tile_size=48, overlap=12))
    manifests.append(services.run_semantic_segmentation(before, str(root / "semantic_segmentation"), tile_size=48, overlap=12))
    manifests.append(services.run_instance_segmentation(before, str(root / "instance_segmentation"), tile_size=48, overlap=12))
    manifests.append(services.run_change_detection(before, after, str(root / "change_detection"), tile_size=48, overlap=12, threshold=0.25))
    manifests.append(services.run_super_resolution(before, str(root / "super_resolution"), tile_size=48, overlap=12, scale=2))
    manifests.append(services.run_spectral_indices(before, str(root / "spectral_indices"), tile_size=48, overlap=12))

    analyzed_manifests = []
    for manifest in manifests:
        analyzed = services.analyze_job(manifest["job_id"])
        reported = services.generate_job_report(analyzed["job_id"])
        if reported.get("status") != "success":
            raise RuntimeError(f"Unexpected manifest status for {reported['job_id']}: {reported.get('status')}")
        if "report" not in reported.get("outputs", {}):
            raise RuntimeError(f"Missing report output for {reported['job_id']}")
        analyzed_manifests.append(reported)
    manifests = analyzed_manifests

    print(json.dumps({"ok": True, "manifest_paths": [item["manifest_path"] for item in manifests]}, indent=2))


if __name__ == "__main__":
    main()
