from __future__ import annotations

import numpy as np

from rs_service.adapters.fake_change_adapter import FakeChangeDetectionAdapter
from rs_service.adapters.fake_detection_adapter import FakeDetectionAdapter, FakeOrientedDetectionAdapter
from rs_service.adapters.fake_instance_adapter import FakeInstanceSegmentationAdapter
from rs_service.adapters.fake_segmentation_adapter import FakeSemanticSegmentationAdapter
from rs_service.adapters.fake_super_resolution_adapter import FakeSuperResolutionAdapter
from rs_service.workers.common import (
    detection_payload,
    instance_payload,
    kwargs,
    load_array,
    model_config,
    tile_info,
    worker_main,
    write_array,
)


def handle(request: dict, response_path) -> dict:
    """Run one fake tile inference request in a subprocess worker."""
    task = str(request.get("task", model_config(request).get("task", "")))
    tile = load_array(request)
    info = tile_info(request)
    options = kwargs(request)
    if task == "object_detection":
        return detection_payload(FakeDetectionAdapter().predict_tile(tile, info, **options))
    if task == "oriented_detection":
        return detection_payload(FakeOrientedDetectionAdapter().predict_tile(tile, info, **options))
    if task == "instance_segmentation":
        return instance_payload(FakeInstanceSegmentationAdapter().predict_tile(tile, info, **options))
    if task == "semantic_segmentation":
        prediction = FakeSemanticSegmentationAdapter().predict_tile(tile, info, **options)
        payload = {
            "mask_path": write_array(response_path, "mask", prediction.mask.astype(np.uint8)),
            "class_names": prediction.class_names,
        }
        if prediction.probabilities is not None:
            payload["probability_path"] = write_array(response_path, "probability", prediction.probabilities.astype(np.float32))
        return payload
    if task == "change_detection":
        tile_t2 = load_array(request, "tile_t2")
        prediction = FakeChangeDetectionAdapter().predict_tile(tile, info, after_tile=tile_t2, **options)
        return {
            "mask_path": write_array(response_path, "mask", prediction.mask.astype(np.uint8)),
            "probability_path": write_array(response_path, "probability", prediction.probability.astype(np.float32)),
            "threshold": prediction.threshold,
        }
    if task == "super_resolution":
        scale = int(options.get("scale", model_config(request).get("scale", 2)))
        prediction = FakeSuperResolutionAdapter(scale=scale).predict_tile(tile, info, scale=scale)
        return {"image_path": write_array(response_path, "image", prediction.image), "scale": prediction.scale}
    raise ValueError(f"fake_external_infer does not support task={task!r}.")


if __name__ == "__main__":
    worker_main(handle)
