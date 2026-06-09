from __future__ import annotations

from typing import Any

from rs_service.adapters.fake import (
    FakeChangeDetectionAdapter,
    FakeDetectionAdapter,
    FakeInstanceSegmentationAdapter,
    FakeOrientedDetectionAdapter,
    FakeSemanticSegmentationAdapter,
    FakeSuperResolutionAdapter,
)
from rs_service.adapters.placeholders import framework_statuses


TASK_MODEL_IDS = {
    "object_detection": "fake_detection",
    "oriented_detection": "fake_oriented_detection",
    "semantic_segmentation": "fake_segmentation",
    "instance_segmentation": "fake_instance",
    "change_detection": "fake_change",
    "super_resolution": "fake_super_resolution",
}


def list_models() -> dict[str, Any]:
    adapters = [
        FakeDetectionAdapter().metadata.to_dict(),
        FakeOrientedDetectionAdapter().metadata.to_dict(),
        FakeSemanticSegmentationAdapter().metadata.to_dict(),
        FakeInstanceSegmentationAdapter().metadata.to_dict(),
        FakeChangeDetectionAdapter().metadata.to_dict(),
        FakeSuperResolutionAdapter().metadata.to_dict(),
    ]
    return {
        "models": adapters,
        "frameworks": framework_statuses(),
        "stage": "fake-adapters",
        "weights_policy": "Large weights are not committed. Use scripts/download_weights.py and workspace/models/*.",
    }


def get_adapter(task: str, model_id: str | None = None, **kwargs: Any) -> Any:
    selected = model_id or TASK_MODEL_IDS.get(task)
    if task == "object_detection" and selected in {"fake_detection", "fake-yolo-sahi"}:
        return FakeDetectionAdapter()
    if task == "oriented_detection" and selected in {"fake_oriented_detection", "fake-mmrotate"}:
        return FakeOrientedDetectionAdapter()
    if task == "semantic_segmentation" and selected in {"fake_segmentation", "fake-mmseg"}:
        return FakeSemanticSegmentationAdapter()
    if task == "instance_segmentation" and selected in {"fake_instance", "fake-mmdet-instance"}:
        return FakeInstanceSegmentationAdapter()
    if task == "change_detection" and selected in {"fake_change", "fake-opencd"}:
        return FakeChangeDetectionAdapter()
    if task == "super_resolution" and selected in {"fake_super_resolution", "fake-sr"}:
        return FakeSuperResolutionAdapter(scale=int(kwargs.get("scale", 2)))
    raise ValueError(f"No adapter registered for task={task!r}, model_id={selected!r}")
