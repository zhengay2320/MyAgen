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
    """Write a small synthetic raster for subprocess smoke tests."""
    array = create_synthetic_array(width=96, height=72, bands=4, changed=changed)
    profile = profile_from_array(array, transform=DEFAULT_TRANSFORM)
    write_raster(path, array, profile)
    return str(path)


def main() -> None:
    """Run the fake external subprocess adapter through all tiled pipelines."""
    root = Path("workspace/smoke_subprocess")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    before = _write_input(root / "before.tif", changed=False)
    after = _write_input(root / "after.tif", changed=True)

    inspect = services.inspect_raster(before)
    preflight = services.preflight_plan(before, task="object_detection", tile_size=40, overlap=8)
    if inspect["width"] <= 0 or preflight["tile_count"] <= 0:
        raise RuntimeError("Subprocess smoke preflight failed.")

    manifests = [
        services.run_object_detection(
            before,
            str(root / "object_detection"),
            tile_size=40,
            overlap=8,
            model_id="fake_external_detection",
        ),
        services.run_semantic_segmentation(
            before,
            str(root / "semantic_segmentation"),
            tile_size=40,
            overlap=8,
            model_id="fake_external_segmentation",
        ),
        services.run_instance_segmentation(
            before,
            str(root / "instance_segmentation"),
            tile_size=40,
            overlap=8,
            model_id="fake_external_instance_segmentation",
        ),
        services.run_oriented_detection(
            before,
            str(root / "oriented_detection"),
            tile_size=40,
            overlap=8,
            model_id="fake_external_oriented_detection",
        ),
        services.run_change_detection(
            before,
            after,
            str(root / "change_detection"),
            tile_size=40,
            overlap=8,
            threshold=0.25,
            model_id="fake_external_change_detection",
        ),
        services.run_super_resolution(
            before,
            str(root / "super_resolution"),
            tile_size=40,
            overlap=8,
            scale=2,
            model_id="fake_external_super_resolution",
        ),
    ]

    for manifest in manifests:
        _validate_manifest(manifest)

    print(
        json.dumps(
            {
                "ok": True,
                "mode": "subprocess",
                "inspect": {"width": inspect["width"], "height": inspect["height"], "count": inspect["count"]},
                "preflight": {
                    "tile_count": preflight["tile_count"],
                    "tile_size": preflight["tile_size"],
                    "overlap": preflight["overlap"],
                },
                "manifest_paths": [item["manifest_path"] for item in manifests],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate the output contract for one subprocess fake job."""
    job_id = manifest.get("job_id", "<missing>")
    if manifest.get("status") != "success":
        raise RuntimeError(f"Unexpected subprocess manifest status for {job_id}: {manifest.get('status')}")
    manifest_path = Path(str(manifest.get("manifest_path", "")))
    if not manifest_path.exists():
        raise RuntimeError(f"Missing manifest.json for {job_id}: {manifest_path}")
    outputs = manifest.get("outputs", {})
    if not isinstance(outputs, dict) or not outputs:
        raise RuntimeError(f"Missing outputs for {job_id}")
    output_paths = [Path(str(value)) for value in outputs.values() if isinstance(value, str)]
    if not any(path.exists() for path in output_paths):
        raise RuntimeError(f"No existing output files for {job_id}")
    if not manifest.get("statistics") and not manifest.get("stats"):
        raise RuntimeError(f"Missing statistics for {job_id}")
    model = manifest.get("model", {})
    if model.get("backend") != "fake_external":
        raise RuntimeError(f"Expected fake_external backend for {job_id}, got {model.get('backend')!r}")


if __name__ == "__main__":
    main()
