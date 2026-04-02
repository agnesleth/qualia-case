# LeRobot v3 Dataset Augmentation Tool

A CLI tool that augments [LeRobot v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) datasets from Hugging Face Hub — applying visual and geometric transforms to multiply training data for robot learning.

## What it does

- Downloads a source dataset, applies augmentations, creates new episodes, and pushes the result back to HF Hub
- **Visual augmentations**: ColorJitter, GaussianBlur, Sharpness, RandomErasing, StaticErasing (camera dirt), DriftingBlob (lens smudge)
- **Geometric augmentations**: HorizontalFlip — flips images *and* mirrors action/state vectors so the training signal stays consistent
- **Temporal augmentations**: FrameDecimator — drops every Nth frame
- Augmentations can be combined freely (e.g. `horizontal_flip color_jitter`)

## Setup

```bash
CONDA_SUBDIR=osx-arm64 conda env create -f environment.yml
conda activate lerobot-augment
huggingface-cli login
```

## Usage

```bash
# Visual augmentation (color jitter)
python augment_dataset.py \
    --source lerobot/aloha_static_cups_open \
    --output your-username/aloha_augmented \
    --augmentations color_jitter \
    --num-passes 2 --include-originals

# Horizontal flip with action/state mirroring (ALOHA bimanual preset)
python augment_dataset.py \
    --source lerobot/aloha_static_cups_open \
    --output your-username/aloha_flipped \
    --augmentations horizontal_flip \
    --robot-type aloha --num-passes 1 --no-push --force
```

Use `--no-push` to skip uploading to HF Hub and `--force` to overwrite existing local cache. Run `python augment_dataset.py --help` for all options. Augmentations can be combined (e.g. `--augmentations horizontal_flip color_jitter`).

## Available augmentations

| Name | Description | Key parameters |
|------|-------------|----------------|
| `color_jitter` | Random brightness, contrast, saturation, hue | `--brightness`, `--contrast`, `--saturation`, `--hue` |
| `gaussian_blur` | Random Gaussian blur | `--blur-kernel`, `--blur-sigma` |
| `sharpness` | Sharpness adjustment | `--sharpness-factor` |
| `random_erasing` | Random rectangular patches (per frame) | `--erasing-p`, `--erasing-scale` |
| `static_erasing` | Fixed rectangular patch (per episode) | `--erasing-scale` |
| `drifting_blob` | Soft blob drifting across frames | `--blob-radius`, `--blob-speed`, `--blob-softness`, `--blob-opacity` |
| `frame_decimate` | Remove every Nth frame | `--remove-every-n` |
| `horizontal_flip` | Flip images + mirror actions/states | `--robot-type` or `--action-mirror-mask` / `--state-mirror-mask` |

## Project structure

```
augment_dataset.py   # Main CLI: loads dataset, applies augmentations, saves result
transforms.py        # Custom transforms (StaticErasing, DriftingBlob, HorizontalFlipWithActionMirror, FrameDecimator)
explore_dataset.py   # Helper for inspecting datasets
environment.yml      # Conda environment
```

## How AI coding agents were used

 -This project was built with **Claude Code** (Anthropic's CLI agent). The workflow:                                  
1. **Research**: Claude Code fetched LeRobot v3 docs, inspected the `LeRobotDataset` API, and explored source code to understand data formats (CHW float32 tensors in, HWC numpy arrays out via `add_frame`).
2. **Design**: A planning agent designed the architecture — choosing the high-level `add_frame` loop over low-level video manipulation, and identifying the key constraint that geometric augmentations (like horizontal flip) must also transform action/state data to avoid contradictory training signals.                                               
3. **Implementation**: Claude Code wrote the tool iteratively, debugging conda/arm64 issues, frame format conversions (CHW to HWC), and cache conflicts along the way.
4. **Testing**: Verified each augmentation by comparing original vs. augmented frames visually and — for horizontal flip — numerically validating that action/state vectors were correctly mirrored.  




