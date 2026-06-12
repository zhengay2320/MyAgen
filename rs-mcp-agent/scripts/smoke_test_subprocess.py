from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rs_service import services
from rs_service.adapters.base import ChangePrediction, SegmentationPrediction, SuperResolutionPrediction
from rs_service.adapters.external_subprocess_adapter import ExternalSubprocessAdapter
from rs_service.core.raster import DEFAULT_TRANSFORM, create_synthetic_array, profile_from_array, write_raster
from rs_service.registry import get_adapter


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
    adapter_checks = _validate_adapters()

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
            model_id="fake_external_instance",
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
            model_id="fake_external_change",
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
                "adapter_checks": adapter_checks,
                "manifest_paths": [item["manifest_path"] for item in manifests],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _validate_adapters() -> dict[str, Any]:
    """Validate direct fake_external subprocess adapter calls."""
    tile = create_synthetic_array(width=32, height=24, bands=4, changed=False)
    changed = create_synthetic_array(width=32, height=24, bands=4, changed=True)
    checks: dict[str, Any] = {}

    detection = get_adapter("object_detection", model_id="fake_external_detection")
    if not isinstance(detection, ExternalSubprocessAdapter):
        raise RuntimeError("fake_external_detection did not resolve to ExternalSubprocessAdapter.")
    detections = detection.predict(tile, context={"tile": _TileInfo("direct_detection")})
    if not detections:
        raise RuntimeError("fake_external_detection returned no tile predictions.")
    checks["detection_count"] = len(detections)

    segmentation = get_adapter("semantic_segmentation", model_id="fake_external_segmentation")
    seg_prediction = segmentation.predict_tile(tile, {"tile_id": "direct_segmentation"})
    if not isinstance(seg_prediction, SegmentationPrediction) or seg_prediction.mask.shape != tile.shape[-2:]:
        raise RuntimeError("fake_external_segmentation did not return a valid mask.")
    checks["segmentation_mask_shape"] = list(seg_prediction.mask.shape)

    instance = get_adapter("instance_segmentation", model_id="fake_external_instance")
    instance_predictions = instance.predict_tile(tile, {"tile_id": "direct_instance"})
    if not instance_predictions or int(instance_predictions[0].mask.sum()) <= 0:
        raise RuntimeError("fake_external_instance did not return a non-empty instance mask.")
    checks["instance_mask_pixels"] = int(instance_predictions[0].mask.sum())

    change = get_adapter("change_detection", model_id="fake_external_change")
    change_prediction = change.predict_tile(tile, {"tile_id": "direct_change"}, tile_t2=changed, threshold=0.25)
    if not isinstance(change_prediction, ChangePrediction) or change_prediction.probability.shape != tile.shape[-2:]:
        raise RuntimeError("fake_external_change did not return a valid probability map.")
    checks["change_probability_shape"] = list(change_prediction.probability.shape)

    sr = get_adapter("super_resolution", model_id="fake_external_super_resolution", scale=2)
    sr_prediction = sr.predict_tile(tile, {"tile_id": "direct_sr"}, scale=2)
    if not isinstance(sr_prediction, SuperResolutionPrediction) or sr_prediction.image.shape[-2:] != (tile.shape[-2] * 2, tile.shape[-1] * 2):
        raise RuntimeError("fake_external_super_resolution did not return a valid SR image.")
    checks["super_resolution_shape"] = list(sr_prediction.image.shape)
    return checks


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


class _TileInfo:
    """Tiny tile context for adapter-level subprocess smoke checks."""

    def __init__(self, tile_id: str) -> None:
        self.tile_id = tile_id
        self.x0 = 0
        self.y0 = 0
        self.width = 32
        self.height = 24


if __name__ == "__main__":
    main()
