from __future__ import annotations

from rs_service.adapters.swinir_adapter import SwinIRAdapter
from rs_service.workers.common import kwargs, load_array, model_config, tile_info, worker_main, write_array


def handle(request: dict, response_path) -> dict:
    """Run one SwinIR super-resolution tile inference request."""
    options = kwargs(request)
    adapter = SwinIRAdapter(model_config(request))
    prediction = adapter.predict_tile(load_array(request), tile_info(request), **options)
    return {"image_path": write_array(response_path, "image", prediction.image), "scale": prediction.scale}


if __name__ == "__main__":
    worker_main(handle)
