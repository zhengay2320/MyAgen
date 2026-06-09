from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


WEIGHT_DIRS = {
    "object_detection": "workspace/models/ultralytics",
    "oriented_detection": "workspace/models/mmrotate",
    "semantic_segmentation": "workspace/models/mmsegmentation",
    "instance_segmentation": "workspace/models/mmdetection",
    "change_detection": "workspace/models/opencd",
    "super_resolution": "workspace/models/super_resolution",
    "prompt_segmentation": "workspace/models/prompt_segmentation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare model weight directories. No large weights are committed.")
    parser.add_argument("--root", default="workspace/models", help="Model root directory.")
    parser.add_argument("--print-plan", action="store_true", help="Print the expected paths and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    for task, relative in WEIGHT_DIRS.items():
        path = root / Path(relative).relative_to("workspace/models")
        path.mkdir(parents=True, exist_ok=True)
        print(f"{task}: {path}")
    if args.print_plan:
        print("Place downloaded open-source checkpoints under the task-specific directories above.")
        print("Do not commit checkpoint files such as *.pt, *.pth, *.ckpt, *.safetensors, or model archives.")


if __name__ == "__main__":
    main()
