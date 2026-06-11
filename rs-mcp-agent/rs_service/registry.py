from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
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
from rs_service.settings import get_settings


TASK_MODEL_IDS = {
    "object_detection": "fake_detection",
    "oriented_detection": "fake_oriented_detection",
    "semantic_segmentation": "fake_segmentation",
    "instance_segmentation": "fake_instance",
    "change_detection": "fake_change",
    "super_resolution": "fake_super_resolution",
}

FAKE_ALIASES = {
    "fake_detection": "object_detection",
    "fake-yolo-sahi": "object_detection",
    "fake_oriented_detection": "oriented_detection",
    "fake-mmrotate": "oriented_detection",
    "fake_segmentation": "semantic_segmentation",
    "fake-mmseg": "semantic_segmentation",
    "fake_instance": "instance_segmentation",
    "fake-mmdet-instance": "instance_segmentation",
    "fake_change": "change_detection",
    "fake_change_detection": "change_detection",
    "fake-opencd": "change_detection",
    "fake_super_resolution": "super_resolution",
    "fake-sr": "super_resolution",
}


def _fake_model_configs() -> dict[str, dict[str, Any]]:
    """Return built-in fake model configs that are always available."""
    adapters = [
        FakeDetectionAdapter(),
        FakeOrientedDetectionAdapter(),
        FakeSemanticSegmentationAdapter(),
        FakeInstanceSegmentationAdapter(),
        FakeChangeDetectionAdapter(),
        FakeSuperResolutionAdapter(),
    ]
    configs: dict[str, dict[str, Any]] = {}
    for adapter in adapters:
        metadata = adapter.metadata.to_dict()
        configs[str(metadata["id"])] = {
            "id": metadata["id"],
            "task": metadata["task"],
            "backend": metadata["backend"],
            "framework": metadata["framework"],
            "weights": metadata.get("weights"),
            "description": metadata.get("description", ""),
        }
    for model_id, task in FAKE_ALIASES.items():
        configs.setdefault(
            model_id,
            {
                "id": model_id,
                "task": task,
                "backend": "fake",
                "framework": "numpy",
                "weights": None,
                "description": "Built-in fake adapter alias.",
            },
        )
    return configs


def list_models() -> dict[str, Any]:
    """List built-in fake models plus models declared in configs/models.yaml."""
    models = _merged_model_configs()
    return {
        "models": [_public_model_config(config) for config in models.values()],
        "frameworks": framework_statuses(),
        "stage": "yaml-configured-model-registry",
        "models_config": str(_models_config_path()),
        "weights_policy": "Large weights are not committed. Use scripts/download_weights.py and workspace/models/*.",
    }


def get_adapter(task: str, model_id: str | None = None, **kwargs: Any) -> Any:
    """Instantiate an adapter from YAML config, falling back to fake models."""
    selected = model_id or TASK_MODEL_IDS.get(task)
    config = _resolve_model_config(task, selected, kwargs)
    backend = str(config.get("backend", "fake")).lower()
    resolved_task = str(config.get("task") or task)
    runner = str(config.get("runner", "inprocess") or "inprocess").lower()

    if runner == "subprocess":
        from rs_service.adapters.external_subprocess_adapter import ExternalSubprocessAdapter

        return ExternalSubprocessAdapter(config)
    if runner == "http":
        raise ValueError("runner=http is reserved for a future remote model service runner and is not implemented yet.")
    if runner != "inprocess":
        raise ValueError(f"Unsupported model runner={runner!r} for model_id={selected!r}.")
    if backend == "fake":
        return _fake_adapter(resolved_task, config, kwargs)
    if resolved_task == "object_detection" and backend == "ultralytics":
        from rs_service.adapters.ultralytics_adapter import UltralyticsDetectionAdapter

        return UltralyticsDetectionAdapter(config)
    if resolved_task == "object_detection" and backend == "sahi":
        from rs_service.adapters.sahi_adapter import SahiDetectionAdapter

        return SahiDetectionAdapter(config)
    if resolved_task == "instance_segmentation" and backend == "ultralytics":
        from rs_service.adapters.ultralytics_adapter import UltralyticsInstanceSegmentationAdapter

        return UltralyticsInstanceSegmentationAdapter(config)
    if resolved_task == "instance_segmentation" and backend == "mmdet":
        from rs_service.adapters.mmdet_adapter import MMDetectionInstanceAdapter

        return MMDetectionInstanceAdapter(config)
    if resolved_task == "semantic_segmentation" and backend == "mmseg":
        from rs_service.adapters.mmseg_adapter import MMSegmentationAdapter

        return MMSegmentationAdapter(config)
    if resolved_task == "oriented_detection" and backend == "mmrotate":
        from rs_service.adapters.mmrotate_adapter import MMRotateDetectionAdapter

        return MMRotateDetectionAdapter(config)
    if resolved_task == "change_detection" and backend == "opencd":
        from rs_service.adapters.opencd_adapter import OpenCDAdapter

        return OpenCDAdapter(config)
    if resolved_task == "super_resolution" and backend == "swinir":
        from rs_service.adapters.swinir_adapter import SwinIRAdapter

        return SwinIRAdapter(config)
    if resolved_task == "super_resolution" and backend == "basicsr":
        from rs_service.adapters.basicsr_adapter import BasicSRAdapter

        return BasicSRAdapter(config)
    if resolved_task == "super_resolution" and backend == "mmagic":
        from rs_service.adapters.mmagic_adapter import MMagicSuperResolutionAdapter

        return MMagicSuperResolutionAdapter(config)

    fallback = TASK_MODEL_IDS.get(task)
    if fallback and fallback != selected:
        return get_adapter(task, model_id=fallback, **kwargs)
    raise ValueError(f"No adapter registered for task={task!r}, model_id={selected!r}, backend={backend!r}")


def _fake_adapter(task: str, config: dict[str, Any], overrides: dict[str, Any]) -> Any:
    """Instantiate a built-in fake adapter for a task."""
    if task == "object_detection":
        return FakeDetectionAdapter()
    if task == "oriented_detection":
        return FakeOrientedDetectionAdapter()
    if task == "semantic_segmentation":
        return FakeSemanticSegmentationAdapter()
    if task == "instance_segmentation":
        return FakeInstanceSegmentationAdapter()
    if task == "change_detection":
        return FakeChangeDetectionAdapter()
    if task == "super_resolution":
        return FakeSuperResolutionAdapter(scale=int(overrides.get("scale", config.get("scale", 2))))
    raise ValueError(f"No fake adapter for task={task!r}")


def _resolve_model_config(task: str, model_id: str | None, overrides: dict[str, Any]) -> dict[str, Any]:
    """Resolve model config from YAML or fake fallback and apply runtime overrides."""
    models = _merged_model_configs()
    selected = model_id or TASK_MODEL_IDS.get(task)
    config = dict(models.get(str(selected), {})) if selected else {}
    if not config:
        fallback = TASK_MODEL_IDS.get(task)
        config = dict(models.get(str(fallback), {})) if fallback else {}
    if not config:
        raise ValueError(f"No model config found for task={task!r}, model_id={selected!r}")
    if config.get("task") and task and config["task"] != task:
        fallback = TASK_MODEL_IDS.get(task)
        if fallback and fallback in models:
            config = dict(models[fallback])
        else:
            raise ValueError(f"Model {selected!r} is registered for task={config.get('task')!r}, not {task!r}")
    config.update({key: value for key, value in overrides.items() if value is not None})
    if "weight" not in config and "weights" in config:
        config["weight"] = config["weights"]
    if "weights" not in config and "weight" in config:
        config["weights"] = config["weight"]
    return config


def _public_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return stable public model fields for /models."""
    return {
        "id": config.get("id"),
        "task": config.get("task"),
        "backend": config.get("backend"),
        "framework": config.get("framework"),
        "weights": config.get("weights") or config.get("weight") or config.get("checkpoint"),
        "checkpoint": config.get("checkpoint"),
        "config": config.get("config"),
        "device": config.get("device"),
        "classes": config.get("classes"),
        "tile_size": config.get("tile_size"),
        "overlap": config.get("overlap"),
        "scale": config.get("scale"),
        "runner": config.get("runner", "inprocess"),
        "conda_env": config.get("conda_env"),
        "entrypoint": config.get("entrypoint"),
        "runner_timeout_sec": config.get("runner_timeout_sec"),
        "description": config.get("description", "Configured model."),
    }


def _merged_model_configs() -> dict[str, dict[str, Any]]:
    """Merge built-in fake configs with YAML configs, de-duplicated by id."""
    merged = _fake_model_configs()
    for model in _load_yaml_models():
        model_id = model.get("id")
        if model_id:
            merged[str(model_id)] = model
    return merged


@lru_cache(maxsize=4)
def _load_yaml_models_cached(path_text: str, mtime_ns: int) -> tuple[tuple[tuple[str, Any], ...], ...]:
    """Load model records from YAML as hashable tuples for cache stability."""
    path = Path(path_text)
    models = _read_models_yaml(path)
    return tuple(tuple(sorted(model.items())) for model in models)


def _load_yaml_models() -> list[dict[str, Any]]:
    """Load configured models from settings.RS_MODELS_CONFIG or configs/models.yaml."""
    path = _models_config_path()
    if not path.exists():
        return []
    cached = _load_yaml_models_cached(str(path.resolve()), path.stat().st_mtime_ns)
    return [dict(items) for items in cached]


def _models_config_path() -> Path:
    """Return the current models.yaml path from settings."""
    return Path(get_settings().models_config)


def _read_models_yaml(path: Path) -> list[dict[str, Any]]:
    """Read model entries using PyYAML when present, else a small project-local parser."""
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        payload = yaml.safe_load(text) or {}
        models = payload.get("models", [])
        return [dict(item) for item in models if isinstance(item, dict)]
    except Exception:
        return _parse_models_yaml_subset(text)


def _parse_models_yaml_subset(text: str) -> list[dict[str, Any]]:
    """Parse the models list from the repository's simple YAML without PyYAML."""
    models: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    pending_key: str | None = None
    in_models = False
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        if indent == 0:
            in_models = stripped == "models:"
            if not in_models and models:
                break
            continue
        if not in_models:
            continue
        if indent == 2 and stripped.startswith("- "):
            if current is not None:
                models.append(current)
            current = {}
            pending_key = None
            remainder = stripped[2:].strip()
            if remainder:
                key, value = _split_key_value(remainder)
                current[key] = _parse_scalar(value)
            continue
        if current is None:
            continue
        if indent == 4 and stripped.startswith("- ") and pending_key:
            current.setdefault(pending_key, []).append(_parse_scalar(stripped[2:].strip()))
            continue
        if indent >= 4 and ":" in stripped:
            key, value = _split_key_value(stripped)
            if value == "":
                current[key] = []
                pending_key = key
            else:
                current[key] = _parse_scalar(value)
                pending_key = None
    if current is not None:
        models.append(current)
    return models


def _split_key_value(text: str) -> tuple[str, str]:
    """Split a YAML key/value line once."""
    key, value = text.split(":", 1)
    return key.strip(), value.strip()


def _parse_scalar(value: str) -> Any:
    """Parse a simple YAML scalar or inline list."""
    if value in {"", "null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except Exception:
            return value
    try:
        if any(char in value for char in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")
