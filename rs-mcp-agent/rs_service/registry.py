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

REAL_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "yolo_detection": {
        "id": "yolo_detection",
        "task": "object_detection",
        "backend": "ultralytics",
        "framework": "ultralytics",
        "weights": "weights/yolo_detection.pt",
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "device": "cpu",
        "imgsz": 1024,
    },
    "sahi_yolo_detection": {
        "id": "sahi_yolo_detection",
        "task": "object_detection",
        "backend": "sahi",
        "framework": "sahi+ultralytics",
        "weights": "weights/yolo_detection.pt",
        "confidence_threshold": 0.25,
        "device": "cpu",
        "slice_height": 512,
        "slice_width": 512,
        "overlap_ratio": 0.2,
    },
    "yolo_instance_segmentation": {
        "id": "yolo_instance_segmentation",
        "task": "instance_segmentation",
        "backend": "ultralytics",
        "framework": "ultralytics",
        "weights": "weights/yolo_instance.pt",
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "device": "cpu",
        "imgsz": 1024,
    },
    "mmseg_segformer_landcover": {
        "id": "mmseg_segformer_landcover",
        "task": "semantic_segmentation",
        "backend": "mmseg",
        "framework": "mmsegmentation",
        "config": "external/mmsegmentation/configs/segformer/segformer_landcover.py",
        "checkpoint": "weights/mmseg_segformer_landcover.pth",
        "device": "cpu",
        "classes": ["background", "landcover"],
    },
    "mmseg_deeplab_building": {
        "id": "mmseg_deeplab_building",
        "task": "semantic_segmentation",
        "backend": "mmseg",
        "framework": "mmsegmentation",
        "config": "external/mmsegmentation/configs/deeplabv3/deeplab_building.py",
        "checkpoint": "weights/mmseg_deeplab_building.pth",
        "device": "cpu",
        "classes": ["background", "building"],
    },
    "mmdet_maskrcnn_instance": {
        "id": "mmdet_maskrcnn_instance",
        "task": "instance_segmentation",
        "backend": "mmdet",
        "framework": "mmdetection",
        "config": "external/mmdetection/configs/mask_rcnn/mask-rcnn_remote_sensing.py",
        "checkpoint": "weights/mmdet_maskrcnn_instance.pth",
        "device": "cpu",
        "classes": ["background", "target"],
    },
    "mmrotate_dota_oriented": {
        "id": "mmrotate_dota_oriented",
        "task": "oriented_detection",
        "backend": "mmrotate",
        "framework": "mmrotate",
        "config": "external/mmrotate/configs/oriented_rcnn/oriented-rcnn_dota.py",
        "checkpoint": "weights/mmrotate_dota_oriented.pth",
        "device": "cpu",
        "classes": ["plane", "ship", "storage-tank", "baseball-diamond", "tennis-court"],
        "angle_unit": "auto",
    },
    "opencd_changer_building": {
        "id": "opencd_changer_building",
        "task": "change_detection",
        "backend": "opencd",
        "framework": "open-cd",
        "config": "external/opencd/configs/changer/changer_building.py",
        "checkpoint": "weights/opencd_changer_building.pth",
        "device": "cpu",
        "threshold": 0.5,
    },
    "opencd_changeformer_landcover": {
        "id": "opencd_changeformer_landcover",
        "task": "change_detection",
        "backend": "opencd",
        "framework": "open-cd",
        "config": "external/opencd/configs/changeformer/changeformer_landcover.py",
        "checkpoint": "weights/opencd_changeformer_landcover.pth",
        "device": "cpu",
        "threshold": 0.5,
    },
    "swinir_x2": {
        "id": "swinir_x2",
        "task": "super_resolution",
        "backend": "swinir",
        "framework": "swinir/torch",
        "checkpoint": "weights/swinir_x2.pth",
        "scale": 2,
        "device": "cpu",
    },
    "swinir_x4": {
        "id": "swinir_x4",
        "task": "super_resolution",
        "backend": "swinir",
        "framework": "swinir/torch",
        "checkpoint": "weights/swinir_x4.pth",
        "scale": 4,
        "device": "cpu",
    },
    "basicsr_x4": {
        "id": "basicsr_x4",
        "task": "super_resolution",
        "backend": "basicsr",
        "framework": "basicsr",
        "config": "external/basicsr/options/sr_x4.yml",
        "checkpoint": "weights/basicsr_x4.pth",
        "scale": 4,
        "device": "cpu",
    },
    "mmagic_sr_stub": {
        "id": "mmagic_sr_stub",
        "task": "super_resolution",
        "backend": "mmagic",
        "framework": "mmagic",
        "config": "external/mmagic/configs/sr/sr_stub.py",
        "checkpoint": "weights/mmagic_sr_stub.pth",
        "scale": 4,
        "device": "cpu",
    },
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
    adapters.extend(
        {
            "id": config["id"],
            "task": config["task"],
            "backend": config["backend"],
            "framework": config["framework"],
            "weights": config.get("weights") or config.get("checkpoint"),
            "config": config.get("config"),
            "description": "Optional real model backend; loaded lazily only when selected.",
        }
        for config in REAL_MODEL_CONFIGS.values()
    )
    return {
        "models": adapters,
        "frameworks": framework_statuses(),
        "stage": "fake-adapters-with-optional-real-models",
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
    if task == "change_detection" and selected in {"fake_change", "fake_change_detection", "fake-opencd"}:
        return FakeChangeDetectionAdapter()
    if task == "super_resolution" and selected in {"fake_super_resolution", "fake-sr"}:
        return FakeSuperResolutionAdapter(scale=int(kwargs.get("scale", 2)))
    if task == "super_resolution" and selected in {"swinir_x2", "swinir_x4"}:
        from rs_service.adapters.swinir_adapter import SwinIRAdapter

        return SwinIRAdapter(_model_config(selected, kwargs))
    if task == "super_resolution" and selected == "basicsr_x4":
        from rs_service.adapters.basicsr_adapter import BasicSRAdapter

        return BasicSRAdapter(_model_config(selected, kwargs))
    if task == "super_resolution" and selected == "mmagic_sr_stub":
        from rs_service.adapters.mmagic_adapter import MMagicSuperResolutionAdapter

        return MMagicSuperResolutionAdapter(_model_config(selected, kwargs))
    if task == "object_detection" and selected == "yolo_detection":
        from rs_service.adapters.ultralytics_adapter import UltralyticsDetectionAdapter

        return UltralyticsDetectionAdapter(_model_config(selected, kwargs))
    if task == "object_detection" and selected == "sahi_yolo_detection":
        from rs_service.adapters.sahi_adapter import SahiDetectionAdapter

        return SahiDetectionAdapter(_model_config(selected, kwargs))
    if task == "instance_segmentation" and selected == "yolo_instance_segmentation":
        from rs_service.adapters.ultralytics_adapter import UltralyticsInstanceSegmentationAdapter

        return UltralyticsInstanceSegmentationAdapter(_model_config(selected, kwargs))
    if task == "semantic_segmentation" and selected in {"mmseg_segformer_landcover", "mmseg_deeplab_building"}:
        from rs_service.adapters.mmseg_adapter import MMSegmentationAdapter

        return MMSegmentationAdapter(_model_config(selected, kwargs))
    if task == "instance_segmentation" and selected == "mmdet_maskrcnn_instance":
        from rs_service.adapters.mmdet_adapter import MMDetectionInstanceAdapter

        return MMDetectionInstanceAdapter(_model_config(selected, kwargs))
    if task == "oriented_detection" and selected == "mmrotate_dota_oriented":
        from rs_service.adapters.mmrotate_adapter import MMRotateDetectionAdapter

        return MMRotateDetectionAdapter(_model_config(selected, kwargs))
    if task == "change_detection" and selected in {"opencd_changer_building", "opencd_changeformer_landcover"}:
        from rs_service.adapters.opencd_adapter import OpenCDAdapter

        return OpenCDAdapter(_model_config(selected, kwargs))
    raise ValueError(f"No adapter registered for task={task!r}, model_id={selected!r}")


def _model_config(model_id: str, overrides: dict[str, Any]) -> dict[str, Any]:
    """Return model config with runtime overrides and weight alias normalized."""
    config = dict(REAL_MODEL_CONFIGS[model_id])
    config.update({key: value for key, value in overrides.items() if value is not None})
    if "weight" not in config and "weights" in config:
        config["weight"] = config["weights"]
    return config
