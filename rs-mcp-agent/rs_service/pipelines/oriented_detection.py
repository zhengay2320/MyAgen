from __future__ import annotations

from pathlib import Path
from typing import Any

from rs_service.pipelines.detection import run_oriented_detection as _run_oriented_detection


def run_oriented_detection(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    tile_size: int = 512,
    overlap: int = 64,
    model_id: str | None = None,
    score_threshold: float = 0.0,
) -> dict[str, Any]:
    """Run oriented object detection using fake or MMRotate backends."""
    return _run_oriented_detection(
        input_path,
        output_dir=output_dir,
        tile_size=tile_size,
        overlap=overlap,
        model_id=model_id,
        score_threshold=score_threshold,
    )
