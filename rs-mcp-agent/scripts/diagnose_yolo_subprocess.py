from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rs_service import registry
from rs_service.registry import get_adapter, list_models
from rs_service.settings import get_settings


def main() -> None:
    """Print diagnostics for yolo_detection_subprocess registry routing."""
    model_id = "yolo_detection_subprocess"
    settings = get_settings()
    models_path = registry._models_config_path()
    raw_models = registry._read_models_yaml(models_path) if models_path.exists() else []
    counts = Counter(str(item.get("id")) for item in raw_models if item.get("id"))
    resolved = registry._resolve_model_config("object_detection", model_id, {})
    adapter = get_adapter("object_detection", model_id=model_id)
    list_entry = [item for item in list_models()["models"] if item.get("id") == model_id]
    payload: dict[str, Any] = {
        "cwd": str(Path.cwd()),
        "RS_MODELS_CONFIG": os.getenv("RS_MODELS_CONFIG"),
        "get_settings().models_config": str(settings.models_config),
        "_models_config_path().resolve()": str(models_path.resolve()),
        "models_yaml_exists": models_path.exists(),
        "duplicate_count": counts.get(model_id, 0),
        "raw_yaml_entries": [item for item in raw_models if item.get("id") == model_id],
        "_resolve_model_config": resolved,
        "adapter_type": f"{type(adapter).__module__}.{type(adapter).__name__}",
        "adapter_metadata": adapter.metadata.to_dict(),
        "list_models_entry": list_entry,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
