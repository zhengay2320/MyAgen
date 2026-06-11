from __future__ import annotations

from rs_service.adapters.mmdet_adapter import MMDetectionInstanceAdapter
from rs_service.workers.common import instance_payload, kwargs, load_array, model_config, tile_info, worker_main


def handle(request: dict, response_path) -> dict:
    """Run one MMDetection instance segmentation tile inference request."""
    adapter = MMDetectionInstanceAdapter(model_config(request))
    return instance_payload(adapter.predict_tile(load_array(request), tile_info(request), **kwargs(request)), response_path)


if __name__ == "__main__":
    worker_main(handle)
