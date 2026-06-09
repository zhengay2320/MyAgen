"""Model adapter interfaces and implementations."""

from rs_service.adapters.fake import (
    FakeChangeDetectionAdapter,
    FakeDetectionAdapter,
    FakeInstanceSegmentationAdapter,
    FakeOrientedDetectionAdapter,
    FakeSemanticSegmentationAdapter,
    FakeSuperResolutionAdapter,
)

__all__ = [
    "FakeChangeDetectionAdapter",
    "FakeDetectionAdapter",
    "FakeInstanceSegmentationAdapter",
    "FakeOrientedDetectionAdapter",
    "FakeSemanticSegmentationAdapter",
    "FakeSuperResolutionAdapter",
]
