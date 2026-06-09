from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

try:  # pydantic-settings is a declared runtime dependency, but keep imports test-friendly.
    from pydantic_settings import BaseSettings as _SettingsBase
except Exception:  # pragma: no cover - exercised in minimal local environments
    _SettingsBase = BaseModel


class Settings(_SettingsBase):
    workspace: Path = Field(default=Path("./workspace"))
    models_config: Path = Field(default=Path("./configs/models.yaml"))
    analysis_rules: Path = Field(default=Path("./configs/analysis_rules.yaml"))
    service_host: str = Field(default="127.0.0.1")
    service_port: int = Field(default=8765)

    @property
    def inputs_dir(self) -> Path:
        return self.workspace / "inputs"

    @property
    def outputs_dir(self) -> Path:
        return self.workspace / "outputs"

    @property
    def previews_dir(self) -> Path:
        return self.workspace / "previews"

    @property
    def reports_dir(self) -> Path:
        return self.workspace / "reports"

    @property
    def jobs_dir(self) -> Path:
        return self.workspace / "jobs"

    @property
    def cache_dir(self) -> Path:
        return self.workspace / "cache"

    def ensure_workspace(self) -> None:
        for directory in [
            self.workspace,
            self.inputs_dir,
            self.outputs_dir,
            self.previews_dir,
            self.reports_dir,
            self.jobs_dir,
            self.cache_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace": str(self.workspace),
            "models_config": str(self.models_config),
            "analysis_rules": str(self.analysis_rules),
            "service_host": self.service_host,
            "service_port": self.service_port,
            "inputs_dir": str(self.inputs_dir),
            "outputs_dir": str(self.outputs_dir),
            "previews_dir": str(self.previews_dir),
            "reports_dir": str(self.reports_dir),
            "jobs_dir": str(self.jobs_dir),
            "cache_dir": str(self.cache_dir),
        }


def _env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings(
        workspace=_env_path("RS_WORKSPACE", "./workspace"),
        models_config=_env_path("RS_MODELS_CONFIG", "./configs/models.yaml"),
        analysis_rules=_env_path("RS_ANALYSIS_RULES", "./configs/analysis_rules.yaml"),
        service_host=os.getenv("RS_SERVICE_HOST", "127.0.0.1"),
        service_port=int(os.getenv("RS_SERVICE_PORT", "8765")),
    )
    settings.ensure_workspace()
    return settings
