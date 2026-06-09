from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AdapterMetadata:
    """Static metadata that describes a model adapter."""

    id: str
    task: str
    backend: str
    framework: str
    weights: str | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize adapter metadata."""
        return asdict(self)


@dataclass
class DetectionPrediction:
    """Single detection prediction in tile-local pixel coordinates."""

    label: str
    score: float
    bbox: list[float]
    class_id: int = 0
    polygon: list[tuple[float, float]] | None = None
    rotated_box: dict[str, float] | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize detection prediction."""
        return {
            "label": self.label,
            "score": self.score,
            "bbox": self.bbox,
            "class_id": self.class_id,
            "polygon": self.polygon,
            "rotated_box": self.rotated_box,
            "attributes": self.attributes,
        }


@dataclass
class SegmentationPrediction:
    """Semantic segmentation prediction for one tile."""

    mask: np.ndarray
    probabilities: np.ndarray | None = None
    class_names: dict[int, str] = field(default_factory=dict)


@dataclass
class InstancePrediction:
    """Instance segmentation prediction for one tile."""

    label: str
    score: float
    bbox: list[float]
    mask: np.ndarray
    class_id: int = 0
    polygon: list[tuple[float, float]] | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_detection(self) -> DetectionPrediction:
        """Convert the instance metadata to a detection-like prediction."""
        return DetectionPrediction(
            label=self.label,
            score=self.score,
            bbox=self.bbox,
            class_id=self.class_id,
            polygon=self.polygon,
            attributes=self.attributes,
        )


@dataclass
class ChangePrediction:
    """Change detection prediction for one tile."""

    mask: np.ndarray
    probability: np.ndarray
    threshold: float = 0.5


@dataclass
class SuperResolutionPrediction:
    """Super-resolution prediction for one tile."""

    image: np.ndarray
    scale: int


class BaseAdapter(ABC):
    """Base class for all model adapters, including fake and real adapters."""

    metadata: AdapterMetadata

    def load(self) -> None:
        """Load model resources. Fake adapters are no-ops."""

    @abstractmethod
    def predict_tile(self, tile_array: np.ndarray, tile_info: dict[str, Any], **kwargs: Any) -> Any:
        """Run prediction on one tile and return a task-specific prediction dataclass."""

    def close(self) -> None:
        """Release model resources. Fake adapters are no-ops."""


class ModelBackendUnavailable(RuntimeError):
    """Raised when a real framework adapter is requested without installed dependencies."""
