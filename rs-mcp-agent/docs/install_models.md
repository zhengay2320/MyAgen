# Model Dependency Installation

The base project is designed to install and start without model weights, GPU drivers, or heavy inference frameworks. Install model stacks only for the adapters you plan to enable.

## Base Install

```bash
pip install -e ".[dev]"
```

## Optional Direct Pip Extras

Ultralytics + SAHI detection:

```bash
pip install -e ".[detection]"
```

PyTorch runtime:

```bash
pip install -e ".[torch]"
```

OpenMMLab packages that are commonly available through pip wheels:

```bash
pip install -e ".[openmmlab]"
```

Combined first model stack:

```bash
pip install -e ".[ml]"
```

## Packages Best Installed With OpenMIM Or Source Instructions

Some remote sensing model packages are version- and CUDA-sensitive. They are intentionally not forced into the base dependency set.

- MMSegmentation: often installed with `mim install mmsegmentation`.
- MMRotate: often installed from OpenMMLab instructions matching the local CUDA/PyTorch/MMCV stack.
- Open-CD: often installed from its upstream repository or a pinned research environment.

Record the exact tested commands in deployment-specific documentation before enabling those adapters.

## Weight Paths

Do not commit checkpoints. Use:

```bash
python scripts/download_weights.py --print-plan
```

Expected local roots are under `workspace/models/*`.
