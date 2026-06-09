from __future__ import annotations

from rs_service.adapters.fake_change_adapter import FakeChangeDetectionAdapter
from rs_service.adapters.fake_detection_adapter import FakeDetectionAdapter, FakeOrientedDetectionAdapter
from rs_service.adapters.fake_instance_adapter import FakeInstanceSegmentationAdapter
from rs_service.adapters.fake_segmentation_adapter import FakeSemanticSegmentationAdapter
from rs_service.adapters.fake_super_resolution_adapter import FakeSuperResolutionAdapter

__all__ = [
    "FakeChangeDetectionAdapter",
    "FakeDetectionAdapter",
    "FakeInstanceSegmentationAdapter",
    "FakeOrientedDetectionAdapter",
    "FakeSemanticSegmentationAdapter",
    "FakeSuperResolutionAdapter",
]
