from __future__ import annotations

from rs_service.adapters.ultralytics_adapter import UltralyticsDetectionAdapter, UltralyticsInstanceSegmentationAdapter
from rs_service.workers.common import detection_payload, instance_payload, kwargs, load_array, model_config, tile_info, worker_main


def handle(request: dict, response_path) -> dict:
    """Run one Ultralytics YOLO tile inference request."""
    config = model_config(request)
    tile = load_array(request)
    task = str(request.get("task", config.get("task", "")))
    if task == "instance_segmentation":
        adapter = UltralyticsInstanceSegmentationAdapter(config)
        return instance_payload(adapter.predict_tile(tile, tile_info(request), **kwargs(request)))
    adapter = UltralyticsDetectionAdapter(config)
    return detection_payload(adapter.predict_tile(tile, tile_info(request), **kwargs(request)))


if __name__ == "__main__":
    worker_main(handle)
