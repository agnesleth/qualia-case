# LeRobot v3 Dataset Augmentation Tool

A CLI tool that takes an existing [LeRobot v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) dataset from Hugging Face Hub, applies visual augmentations to create new augmented episodes, and uploads the result as a new dataset.

## What it does

- **Downloads** a source LeRobot v3 dataset (e.g. `lerobot/aloha_static_cups_open`)
- **Applies visual augmentations** (ColorJitter, GaussianBlur, Sharpness, RandomErasing) to camera frames
- **Creates new episodes** with the augmented video frames while keeping states/actions unchanged
- **Pushes** the augmented dataset to Hugging Face Hub
- **Prints** a direct visualizer link to inspect the result

This is useful for **multiplying training data** in robot learning by creating visually varied copies of recorded episodes, improving model robustness to lighting, color, and visual noise variations.

## Setup

### 1. Create the conda environment

```bash
CONDA_SUBDIR=osx-arm64 conda env create -f environment.yml
conda activate lerobot-augment
```

> **Note:** The `CONDA_SUBDIR=osx-arm64` flag is needed if your conda base is x86 (Rosetta). On native arm64 conda or Linux, omit it.

### 2. Log in to Hugging Face

```bash
huggingface-cli login
```

You need a write token from https://huggingface.co/settings/tokens.

## Usage

### Basic: ColorJitter augmentation

```bash
python augment_dataset.py \
    --source lerobot/aloha_static_cups_open \
    --output your-username/aloha_cups_augmented \
    --num-passes 2 \
    --augmentations color_jitter \
    --include-originals
```

This creates a dataset with 50 original + 100 augmented episodes (2 passes x 50 episodes).

### Augment a subset of episodes

```bash
python augment_dataset.py \
    --source lerobot/aloha_static_cups_open \
    --output your-username/aloha_cups_small_aug \
    --episodes 0 1 2 \
    --num-passes 3
```

### Multiple augmentation types

```bash
python augment_dataset.py \
    --source lerobot/aloha_static_cups_open \
    --output your-username/aloha_cups_multi_aug \
    --augmentations color_jitter gaussian_blur \
    --brightness 0.7 1.3 \
    --contrast 0.7 1.3 \
    --blur-kernel 5 \
    --blur-sigma 0.5 2.0
```

### Local-only (no push)

```bash
python augment_dataset.py \
    --source lerobot/aloha_static_cups_open \
    --output your-username/test_local \
    --episodes 0 --num-passes 1 --no-push
```

## Available augmentations

| Name | Description | Key parameters |
|------|-------------|----------------|
| `color_jitter` | Random brightness, contrast, saturation, hue | `--brightness`, `--contrast`, `--saturation`, `--hue` |
| `gaussian_blur` | Random Gaussian blur | `--blur-kernel`, `--blur-sigma` |
| `sharpness` | Random sharpness adjustment | `--sharpness-factor` |
| `random_erasing` | Randomly erase rectangular patches | `--erasing-p`, `--erasing-scale` |

Multiple augmentations can be composed together using `--augmentations name1 name2 ...`.

## CLI reference

```
python augment_dataset.py --help
```

Key flags:
- `--source`: Source dataset repo_id on HF Hub
- `--output`: Output dataset repo_id
- `--num-passes`: Number of augmented copies per episode (default: 2)
- `--augmentations`: Which augmentations to apply (default: color_jitter)
- `--include-originals`: Include unmodified episodes in the output
- `--episodes`: Only process specific episode indices
- `--seed`: Random seed for reproducibility
- `--force`: Overwrite existing local cache
- `--no-push`: Skip uploading to HF Hub
- `--vcodec`: Video codec (default: libsvtav1)

## How AI coding agents were used

This project was built entirely with AI coding assistance using **Claude Code** (Anthropic's CLI agent). The development process:

1. **Research phase**: Claude Code fetched and analyzed the LeRobot v3 documentation, inspected the `LeRobotDataset` API signatures, explored sample datasets on HF Hub, and examined the source code of `lerobot.datasets.lerobot_dataset`, `dataset_tools`, `transforms`, and `image_writer` modules.

2. **Design phase**: A specialized planning agent designed the architecture, considering trade-offs between high-level API usage (`add_frame` loop) vs. low-level video manipulation. The plan identified key API contracts (CHW float32 tensors from `__getitem__`, HWC numpy arrays for `add_frame`, auto-populated DEFAULT_FEATURES).

3. **Implementation**: Claude Code wrote the complete CLI tool in a single pass, then iteratively debugged it:
   - Fixed conda environment issues (Python version, arm64 architecture, ffmpeg dependency)
   - Fixed frame format conversion (CHW->HWC for `add_frame`, scalar reshaping for `next.done`)
   - Added `--force` flag after encountering cache directory conflicts during testing

4. **Testing**: Smoke-tested with 1 episode / 1 pass, verified the output dataset loads correctly via `LeRobotDataset`, then ran full augmentation with hub upload.

The entire process from empty directory to working tool took approximately 1 hour of wall-clock time with Claude Code driving exploration, planning, implementation, and debugging.

## Architecture

The tool uses LeRobot's high-level `LeRobotDataset` API:

```
Source dataset (HF Hub)
    ↓ LeRobotDataset(repo_id) + __getitem__
Decoded frames (float32 CHW tensors)
    ↓ torchvision.transforms.v2 augmentations
Augmented frames
    ↓ CHW→HWC conversion + add_frame()
    ↓ save_episode() → video encoding
    ↓ finalize() → parquet footer metadata
Output dataset
    ↓ push_to_hub()
Hugging Face Hub
```

States and actions pass through unchanged. Only camera features (`observation.images.*`) are augmented.
