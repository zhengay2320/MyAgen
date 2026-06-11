from __future__ import annotations

from rs_service.adapters.mmrotate_adapter import MMRotateDetectionAdapter
from rs_service.workers.common import detection_payload, kwargs, load_array, model_config, tile_info, worker_main


def handle(request: dict, response_path) -> dict:
    """Run one MMRotate oriented detection tile inference request."""
    adapter = MMRotateDetectionAdapter(model_config(request))
    return detection_payload(adapter.predict_tile(load_array(request), tile_info(request), **kwargs(request)))


if __name__ == "__main__":
    worker_main(handle)
