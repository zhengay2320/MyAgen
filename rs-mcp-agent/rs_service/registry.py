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
    "object_detection": "fake-yolo-sahi",
    "oriented_detection": "fake-mmrotate",
    "semantic_segmentation": "fake-mmseg",
    "instance_segmentation": "fake-mmdet-instance",
    "change_detection": "fake-opencd",
    "super_resolution": "fake-sr",
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
    if task == "object_detection" and selected == "fake-yolo-sahi":
        return FakeDetectionAdapter()
    if task == "oriented_detection" and selected == "fake-mmrotate":
        return FakeOrientedDetectionAdapter()
    if task == "semantic_segmentation" and selected == "fake-mmseg":
        return FakeSemanticSegmentationAdapter()
    if task == "instance_segmentation" and selected == "fake-mmdet-instance":
        return FakeInstanceSegmentationAdapter()
    if task == "change_detection" and selected == "fake-opencd":
        return FakeChangeDetectionAdapter()
    if task == "super_resolution" and selected == "fake-sr":
        return FakeSuperResolutionAdapter(scale=int(kwargs.get("scale", 2)))
    raise ValueError(f"No adapter registered for task={task!r}, model_id={selected!r}")
