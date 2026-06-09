from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from rs_service.adapters.base import ModelBackendUnavailable


@dataclass(frozen=True)
class FrameworkStatus:
    name: str
    import_name: str
    task: str
    installed: bool
    role: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "import_name": self.import_name,
            "task": self.task,
            "installed": self.installed,
            "role": self.role,
        }


FRAMEWORKS = [
    ("rasterio", "rasterio", "raster_io", "GeoTIFF IO and georeferencing"),
    ("geopandas", "geopandas", "vector_io", "GPKG/GeoJSON vector output"),
    ("shapely", "shapely", "vector_geometry", "Geometry operations"),
    ("rasterstats", "rasterstats", "zonal_statistics", "Raster zonal statistics"),
    ("ultralytics", "ultralytics", "object_detection", "YOLO model adapter"),
    ("sahi", "sahi", "object_detection", "Sliced detection orchestration"),
    ("mmsegmentation", "mmseg", "semantic_segmentation", "MMSegmentation adapter"),
    ("mmdetection", "mmdet", "instance_segmentation", "MMDetection adapter"),
    ("mmrotate", "mmrotate", "oriented_detection", "MMRotate adapter"),
    ("opencd", "opencd", "change_detection", "Open-CD adapter"),
    ("basicsr", "basicsr", "super_resolution", "BasicSR adapter"),
    ("mmagic", "mmagic", "super_resolution", "MMagic adapter"),
    ("samgeo", "samgeo", "prompt_segmentation", "SAMGeo optional adapter"),
    ("groundingdino", "groundingdino", "prompt_detection", "GroundingDINO optional adapter"),
]


def framework_statuses() -> list[dict]:
    statuses = []
    for name, import_name, task, role in FRAMEWORKS:
        statuses.append(
            FrameworkStatus(
                name=name,
                import_name=import_name,
                task=task,
                installed=importlib.util.find_spec(import_name) is not None,
                role=role,
            ).to_dict()
        )
    return statuses


def raise_unavailable(framework: str) -> None:
    raise ModelBackendUnavailable(
        f"{framework} adapter is not implemented in stage 1. Install the framework and add a concrete adapter "
        "behind rs_service.adapters without changing pipeline code."
    )
