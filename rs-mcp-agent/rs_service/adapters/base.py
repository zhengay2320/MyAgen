from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class AdapterMetadata:
    id: str
    task: str
    backend: str
    framework: str
    weights: str | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "backend": self.backend,
            "framework": self.framework,
            "weights": self.weights,
            "description": self.description,
        }


class DetectionAdapter(Protocol):
    metadata: AdapterMetadata

    def predict(self, tile: np.ndarray, context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        ...


class SegmentationAdapter(Protocol):
    metadata: AdapterMetadata

    def predict_proba(self, tile: np.ndarray, context: dict[str, Any] | None = None) -> np.ndarray:
        ...


class ChangeDetectionAdapter(Protocol):
    metadata: AdapterMetadata

    def predict_proba(
        self,
        tile_a: np.ndarray,
        tile_b: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> np.ndarray:
        ...


class SuperResolutionAdapter(Protocol):
    metadata: AdapterMetadata
    scale: int

    def upscale(self, tile: np.ndarray, context: dict[str, Any] | None = None) -> np.ndarray:
        ...


class ModelBackendUnavailable(RuntimeError):
    """Raised when a real framework adapter is requested without installed dependencies."""
