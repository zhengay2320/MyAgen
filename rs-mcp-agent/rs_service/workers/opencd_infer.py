from __future__ import annotations

import numpy as np

from rs_service.adapters.opencd_adapter import OpenCDAdapter
from rs_service.workers.common import kwargs, load_array, model_config, tile_info, worker_main, write_array


def handle(request: dict, response_path) -> dict:
    """Run one Open-CD synchronized tile-pair inference request."""
    options = kwargs(request)
    adapter = OpenCDAdapter(model_config(request))
    prediction = adapter.predict_tile(
        load_array(request),
        tile_info(request),
        tile_t2=load_array(request, "tile_t2"),
        **options,
    )
    return {
        "mask_path": write_array(response_path, "mask", prediction.mask.astype(np.uint8)),
        "probability_path": write_array(response_path, "probability", prediction.probability.astype(np.float32)),
        "threshold": prediction.threshold,
    }


if __name__ == "__main__":
    worker_main(handle)
