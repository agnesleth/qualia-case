#!/usr/bin/env python3
"""
LeRobot v3 Dataset Augmentation Tool

Takes an existing LeRobot v3 dataset from Hugging Face Hub, applies visual
augmentations to create new augmented episodes, and pushes the result back
as a new dataset. Prints a direct visualizer link for verification.

Usage:
    python augment_dataset.py \
        --source lerobot/aloha_static_cups_open \
        --output your-username/aloha_cups_augmented \
        --num-passes 2 \
        --augmentations color_jitter \
        --include-originals
"""

import argparse
import sys
import time
from urllib.parse import quote

import numpy as np
import torch
from torchvision.transforms import v2
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES

DEFAULT_FEATURE_KEYS = set(DEFAULT_FEATURES.keys())


# ---------------------------------------------------------------------------
# Augmentation registry
# ---------------------------------------------------------------------------

def build_color_jitter(args): #adjusts brightness, contrast, saturation, hue
    return v2.ColorJitter(
        brightness=tuple(args.brightness),
        contrast=tuple(args.contrast),
        saturation=tuple(args.saturation),
        hue=tuple(args.hue),
    )


def build_gaussian_blur(args): #applies a Gaussian blur to the image
    return v2.GaussianBlur(
        kernel_size=args.blur_kernel,
        sigma=tuple(args.blur_sigma),
    )


def build_sharpness(args): #adjusts the sharpness of the image
    return v2.RandomAdjustSharpness(
        sharpness_factor=args.sharpness_factor,
        p=1.0,
    )


def build_random_erasing(args):
    return v2.RandomErasing(
        p=args.erasing_p,
        scale=tuple(args.erasing_scale),
    )


class StaticErasing:
    """Erases a fixed rectangle - sampled once, applied to every frame.

    Call resample() at the start of each episode to pick a new patch position.
    Simulates static occlusions like dirt on the camera lens.
    """

    def __init__(self, scale=(0.02, 0.15), value=0.0):
        self.scale = scale
        self.value = value
        self.i = self.j = self.h = self.w = 0

    def resample(self, img_h, img_w):
        """Pick a new random rectangle for this episode."""
        import random
        area = img_h * img_w
        erase_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect = random.uniform(0.5, 2.0)
        self.h = int(round((erase_area * aspect) ** 0.5))
        self.w = int(round((erase_area / aspect) ** 0.5))
        self.h = min(self.h, img_h)
        self.w = min(self.w, img_w)
        self.i = random.randint(0, img_h - self.h)
        self.j = random.randint(0, img_w - self.w)

    def __call__(self, img):
        img = img.clone()
        img[:, self.i:self.i + self.h, self.j:self.j + self.w] = self.value
        return img

    def __repr__(self):
        return f"StaticErasing(scale={self.scale})"


def build_static_erasing(args):
    return StaticErasing(scale=tuple(args.erasing_scale))


AUGMENTATION_BUILDERS = {
    "color_jitter": build_color_jitter,
    "gaussian_blur": build_gaussian_blur,
    "sharpness": build_sharpness,
    "random_erasing": build_random_erasing,
    "static_erasing": build_static_erasing,
}


def build_transform(args):
    """Build a composed transform from the requested augmentations."""
    transforms = []
    for name in args.augmentations:
        if name not in AUGMENTATION_BUILDERS:
            print(f"Unknown augmentation: {name}. Available: {list(AUGMENTATION_BUILDERS.keys())}")
            sys.exit(1)
        transforms.append(AUGMENTATION_BUILDERS[name](args))

    if len(transforms) == 1:
        return transforms[0]
    return v2.Compose(transforms)


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------

def build_frame_dict(item, feature_keys, features_meta):
    """Convert a __getitem__ result to an add_frame-compatible dict.

    Excludes DEFAULT_FEATURES (auto-populated by add_frame).
    Converts CHW image tensors to HWC numpy arrays (add_frame expects HWC).
    Reshapes scalar tensors to match expected shapes.
    """
    frame = {"task": item["task"]}
    for key in feature_keys:
        if key in DEFAULT_FEATURE_KEYS:
            continue
        if key in item:
            val = item[key]
            if isinstance(val, torch.Tensor):
                val = val.numpy()
            # CHW -> HWC for image/video features
            if val.ndim == 3 and val.shape[0] == 3:
                val = np.transpose(val, (1, 2, 0))
            # Reshape scalars to expected shape
            if key in features_meta:
                expected_shape = tuple(features_meta[key]["shape"])
                if val.ndim == 0 and expected_shape == (1,):
                    val = val.reshape(1)
            frame[key] = val
    return frame


def get_episode_range(source, ep_idx):
    """Return (from_idx, to_idx) absolute frame indices for an episode."""
    ep = source.meta.episodes[ep_idx]
    return ep["dataset_from_index"], ep["dataset_to_index"]


def get_episode_task(source, ep_idx):
    """Return the task string for an episode."""
    ep = source.meta.episodes[ep_idx]
    tasks = ep["tasks"]
    if isinstance(tasks, list):
        return tasks[0]
    return str(tasks)


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def copy_episode(source, output, ep_idx, feature_keys, features_meta):
    """Copy an episode from source to output without modification."""
    from_idx, to_idx = get_episode_range(source, ep_idx)
    for global_idx in range(from_idx, to_idx):
        item = source[global_idx]
        frame = build_frame_dict(item, feature_keys, features_meta)
        output.add_frame(frame)
    output.save_episode()


def augment_episode(source, output, ep_idx, transform, feature_keys, camera_keys, features_meta):
    """Create an augmented copy of an episode."""
    from_idx, to_idx = get_episode_range(source, ep_idx)

    # For static transforms, sample patch position once per episode
    # by peeking at the first frame's image size
    if isinstance(transform, StaticErasing):
        first = source[from_idx]
        for cam_key in camera_keys:
            if cam_key in first:
                _, h, w = first[cam_key].shape
                transform.resample(h, w)
                break

    for global_idx in range(from_idx, to_idx):
        item = source[global_idx]
        # Apply augmentation only to camera (visual) features
        for cam_key in camera_keys:
            if cam_key in item:
                img = item[cam_key]  # float32 CHW tensor in [0, 1]
                item[cam_key] = transform(img)
        frame = build_frame_dict(item, feature_keys, features_meta)
        output.add_frame(frame)
    output.save_episode()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Augment a LeRobot v3 dataset and push to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    parser.add_argument("--source", required=True, help="Source dataset repo_id (e.g. lerobot/aloha_static_cups_open)")
    parser.add_argument("--output", required=True, help="Output dataset repo_id (e.g. user/dataset_augmented)")

    # Augmentation config
    parser.add_argument("--num-passes", type=int, default=2, help="Number of augmented copies per episode (default: 2)")
    parser.add_argument(
        "--augmentations", nargs="+", default=["color_jitter"],
        choices=list(AUGMENTATION_BUILDERS.keys()),
        help="Augmentations to apply (default: color_jitter)",
    )
    parser.add_argument("--include-originals", action="store_true", help="Include original episodes in output dataset")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--episodes", nargs="+", type=int, default=None, help="Only augment these episode indices (default: all)")

    # ColorJitter params
    parser.add_argument("--brightness", nargs=2, type=float, default=[0.8, 1.2], metavar=("MIN", "MAX"))
    parser.add_argument("--contrast", nargs=2, type=float, default=[0.8, 1.2], metavar=("MIN", "MAX"))
    parser.add_argument("--saturation", nargs=2, type=float, default=[0.5, 1.5], metavar=("MIN", "MAX"))
    parser.add_argument("--hue", nargs=2, type=float, default=[-0.05, 0.05], metavar=("MIN", "MAX"))

    # GaussianBlur params
    parser.add_argument("--blur-kernel", type=int, default=5, help="Gaussian blur kernel size (default: 5)")
    parser.add_argument("--blur-sigma", nargs=2, type=float, default=[0.1, 2.0], metavar=("MIN", "MAX"))

    # Sharpness params
    parser.add_argument("--sharpness-factor", type=float, default=2.0, help="Sharpness adjustment factor (default: 2.0)")

    # Random erasing params
    parser.add_argument("--erasing-p", type=float, default=0.5, help="Random erasing probability (default: 0.5)")
    parser.add_argument("--erasing-scale", nargs=2, type=float, default=[0.02, 0.15], metavar=("MIN", "MAX"))

    # Output config
    parser.add_argument("--vcodec", default="libsvtav1", help="Video codec (default: libsvtav1)")
    parser.add_argument("--image-writer-threads", type=int, default=4, help="Threads for image writing (default: 4)")
    parser.add_argument("--no-push", action="store_true", help="Skip pushing to Hub (local only)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing local cache for output dataset")

    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    # Seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # ---- Clean up existing cache if --force ----
    if args.force:
        import shutil
        from pathlib import Path
        cache_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / args.output
        if cache_dir.exists():
            print(f"Removing existing cache: {cache_dir}")
            shutil.rmtree(cache_dir)

    # ---- Load source dataset ----
    print(f"Loading source dataset: {args.source}")
    source = LeRobotDataset(args.source)

    camera_keys = source.meta.camera_keys
    feature_keys = list(source.meta.features.keys())
    episode_indices = args.episodes if args.episodes else list(range(source.meta.total_episodes))

    print(f"  Episodes: {source.meta.total_episodes} ({len(episode_indices)} selected)")
    print(f"  Frames: {source.meta.total_frames}, FPS: {source.fps}")
    print(f"  Cameras: {camera_keys}")
    print(f"  Robot: {source.meta.robot_type}")

    # ---- Compute expected output size ----
    n_original = len(episode_indices) if args.include_originals else 0
    n_augmented = len(episode_indices) * args.num_passes
    print(f"\nOutput plan:")
    print(f"  Original episodes: {n_original}")
    print(f"  Augmented episodes: {n_augmented} ({args.num_passes} passes)")
    print(f"  Total episodes: {n_original + n_augmented}")
    print(f"  Augmentations: {args.augmentations}")

    # ---- Build user features (exclude DEFAULT_FEATURES) ----
    features_meta = source.meta.features
    user_features = {
        k: v for k, v in features_meta.items()
        if k not in DEFAULT_FEATURE_KEYS
    }

    # ---- Create output dataset ----
    print(f"\nCreating output dataset: {args.output}")
    output = LeRobotDataset.create(
        repo_id=args.output,
        fps=source.fps,
        features=user_features,
        robot_type=source.meta.robot_type,
        use_videos=len(source.meta.camera_keys) > 0,
        vcodec=args.vcodec,
        image_writer_threads=args.image_writer_threads,
    )

    try:
        # ---- Copy originals ----
        if args.include_originals:
            print("\nCopying original episodes...")
            for ep_idx in tqdm(episode_indices, desc="Originals"):
                copy_episode(source, output, ep_idx, feature_keys, features_meta)

        # ---- Augmentation passes ----
        transform = build_transform(args)
        print(f"\nAugmentation transform: {transform}")

        for pass_idx in range(args.num_passes):
            if args.seed is not None:
                torch.manual_seed(args.seed + pass_idx + 1)

            desc = f"Pass {pass_idx + 1}/{args.num_passes}"
            for ep_idx in tqdm(episode_indices, desc=desc):
                augment_episode(source, output, ep_idx, transform, feature_keys, camera_keys, features_meta)

        # ---- Finalize ----
        print("\nFinalizing dataset...")
        output.finalize()

        # ---- Push to Hub ----
        if not args.no_push:
            print("Pushing to Hugging Face Hub...")
            output.push_to_hub()

    except Exception:
        # Ensure parquet writers are closed even on error
        output.finalize()
        raise

    # ---- Print results ----
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Total episodes: {n_original + n_augmented}")

    encoded_path = quote(args.output, safe="")
    viz_url = f"https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2F{encoded_path}%2Fepisode_0"
    print(f"\nVisualizer link:\n  {viz_url}")


if __name__ == "__main__":
    main()
