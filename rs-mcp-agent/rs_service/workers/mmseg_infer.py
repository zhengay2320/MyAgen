from __future__ import annotations

import numpy as np

from rs_service.adapters.mmseg_adapter import MMSegmentationAdapter
from rs_service.workers.common import kwargs, load_array, model_config, tile_info, worker_main, write_array


def handle(request: dict, response_path) -> dict:
    """Run one MMSegmentation tile inference request."""
    adapter = MMSegmentationAdapter(model_config(request))
    prediction = adapter.predict_tile(load_array(request), tile_info(request), **kwargs(request))
    payload = {
        "mask_path": write_array(response_path, "mask", prediction.mask.astype(np.uint8)),
        "class_names": prediction.class_names,
        "metadata": prediction.metadata,
    }
    if prediction.probabilities is not None:
        payload["probability_path"] = write_array(response_path, "probability", prediction.probabilities.astype(np.float32))
    return payload


if __name__ == "__main__":
    worker_main(handle)
