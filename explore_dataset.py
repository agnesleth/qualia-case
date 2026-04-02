#!/usr/bin/env python3
"""
Explore the structure and metadata of the lerobot/aloha_static_cups_open dataset.
"""

from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = "lerobot/aloha_static_cups_open"

print(f"Loading dataset: {repo_id}")
dataset = LeRobotDataset(repo_id)

# --- General info ---
print("\n=== General Info ===")
print(f"  Total episodes:  {dataset.meta.total_episodes}")
print(f"  Total frames:    {dataset.meta.total_frames}")
print(f"  FPS:             {dataset.fps}")
print(f"  Robot type:      {dataset.meta.robot_type}")
print(f"  Camera keys:     {dataset.meta.camera_keys}")

# --- Features (schema) ---
print("\n=== Features ===")
for name, meta in dataset.meta.features.items():
    dtype = meta.get("dtype", "?")
    shape = meta.get("shape", "?")
    print(f"  {name:40s}  dtype={dtype:12s}  shape={shape}")

# --- Tasks ---
print("\n=== Tasks ===")
tasks = dataset.meta.tasks
if hasattr(tasks, "iterrows"):
    for _, row in tasks.iterrows():
        print(f"  {dict(row)}")
elif tasks is not None:
    print(f"  {tasks}")
else:
    print("  (no task metadata found)")

# --- Episodes ---
print("\n=== Episodes (first 5) ===")
episodes = dataset.meta.episodes
if hasattr(episodes, "head"):
    print(episodes.head())
else:
    for i, ep in enumerate(episodes[:5]):
        print(f"  Episode {i}: {ep}")

# --- Stats ---
print("\n=== Stats ===")
if hasattr(dataset.meta, "stats") and dataset.meta.stats:
    for key, stat in dataset.meta.stats.items():
        parts = []
        for stat_name in ("mean", "std", "min", "max"):
            if stat_name in stat:
                val = stat[stat_name]
                # Truncate long arrays for readability
                s = str(val)
                if len(s) > 80:
                    s = s[:77] + "..."
                parts.append(f"{stat_name}={s}")
        print(f"  {key}: {', '.join(parts)}")
else:
    print("  (no stats found)")

# --- Sample item ---
print("\n=== Sample item (index 0) ===")
sample = dataset[0]
for key, val in sample.items():
    if hasattr(val, "shape"):
        print(f"  {key:40s}  shape={str(val.shape):20s}  dtype={val.dtype}")
    else:
        print(f"  {key:40s}  value={val}")

print("\nDone.")
