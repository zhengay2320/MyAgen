# Model Weight Policy

This repository does not commit large model weights.

Expected local paths:

- `workspace/models/ultralytics`: Ultralytics/YOLO checkpoints for object detection.
- `workspace/models/mmrotate`: MMRotate checkpoints for oriented detection.
- `workspace/models/mmsegmentation`: MMSegmentation checkpoints for semantic segmentation.
- `workspace/models/mmdetection`: MMDetection checkpoints for instance segmentation.
- `workspace/models/opencd`: Open-CD checkpoints for change detection.
- `workspace/models/super_resolution`: BasicSR, MMagic, or SwinIR checkpoints.
- `workspace/models/prompt_segmentation`: optional SAMGeo and GroundingDINO checkpoints.

Use:

```bash
python scripts/download_weights.py --print-plan
```

Concrete download commands should be added per deployment after selecting open-source checkpoints and validating their licenses.
