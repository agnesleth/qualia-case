"""
Custom transforms for LeRobot datasets.

Contains both image-level transforms (StaticErasing, DriftingBlob) and
episode-level transforms (FrameDecimator) that are used by augment_dataset.py.
"""

import math
import random

import torch
from torchvision.transforms import functional as F


class FrameDecimator:
    """Remove every Nth frame from an episode.

    Example: remove_every_n=5 keeps 4 out of every 5 frames,
    dropping frames at positions 4, 9, 14, ... (0-indexed).
    """

    def __init__(self, remove_every_n: int = 5):
        if remove_every_n < 2:
            raise ValueError(f"remove_every_n must be >= 2, got {remove_every_n}")
        self.remove_every_n = remove_every_n

    def should_keep(self, frame_index: int) -> bool:
        return (frame_index + 1) % self.remove_every_n != 0

    def __repr__(self):
        return f"FrameDecimator(remove_every_n={self.remove_every_n})"


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


class DriftingBlob:
    """A soft, round blob that drifts smoothly across frames.

    The blob is filled with the local average color under the mask,
    creating a low-contrast smudge rather than a hard occlusion.
    Call resample() once per episode to pick a random start position
    and direction. Each __call__ applies the blob and advances its position.
    """

    def __init__(self, radius: int = 30, speed: float = 2.0, softness: float = 0.6, opacity: float = 0.5):
        self.radius = radius
        self.speed = speed
        self.softness = softness
        self.opacity = opacity
        # State (set by resample)
        self.cy = 0.0
        self.cx = 0.0
        self.vy = 0.0
        self.vx = 0.0
        self.img_h = 0
        self.img_w = 0
        # Precompute the gaussian mask kernel
        self._mask = self._make_mask(radius, softness)

    @staticmethod
    def _make_mask(radius, softness):
        """Create a 2D gaussian soft-circle mask of shape (2r+1, 2r+1)."""
        size = 2 * radius + 1
        center = radius
        # sigma controls how soft the edge is; lower softness = harder edge
        sigma = radius * max(softness, 0.05)
        y = torch.arange(size, dtype=torch.float32) - center
        x = torch.arange(size, dtype=torch.float32) - center
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        dist_sq = yy ** 2 + xx ** 2
        mask = torch.exp(-dist_sq / (2 * sigma ** 2))
        # Zero out anything outside the circle radius
        mask[dist_sq > radius ** 2] = 0.0
        # Normalize peak to 1
        if mask.max() > 0:
            mask = mask / mask.max()
        return mask  # shape: (2r+1, 2r+1)

    def resample(self, img_h, img_w):
        """Pick a random start position and drift direction for this episode."""
        self.img_h = img_h
        self.img_w = img_w
        self.cy = random.uniform(self.radius, img_h - self.radius)
        self.cx = random.uniform(self.radius, img_w - self.radius)
        angle = random.uniform(0, 2 * math.pi)
        self.vy = self.speed * math.sin(angle)
        self.vx = self.speed * math.cos(angle)

    def __call__(self, img):
        """Apply blob at current position, then advance position.

        Args:
            img: float32 CHW tensor in [0, 1]
        Returns:
            Modified CHW tensor
        """
        img = img.clone()
        C, H, W = img.shape
        r = self.radius
        mask = self._mask  # (2r+1, 2r+1)

        # Integer center
        cy_int = int(round(self.cy))
        cx_int = int(round(self.cx))

        # Compute overlap between mask and image
        # Mask coordinates
        m_top = max(0, r - cy_int)
        m_left = max(0, r - cx_int)
        m_bottom = min(2 * r + 1, H - cy_int + r)
        m_right = min(2 * r + 1, W - cx_int + r)

        # Image coordinates
        i_top = max(0, cy_int - r)
        i_left = max(0, cx_int - r)
        i_bottom = min(H, cy_int + r + 1)
        i_right = min(W, cx_int + r + 1)

        if i_top >= i_bottom or i_left >= i_right:
            self._advance()
            return img

        # Extract mask patch and image patch
        mask_patch = mask[m_top:m_bottom, m_left:m_right]  # (ph, pw)
        img_patch = img[:, i_top:i_bottom, i_left:i_right]  # (C, ph, pw)

        # Compute average color under the mask
        mask_sum = mask_patch.sum()
        if mask_sum > 0:
            avg_color = (img_patch * mask_patch.unsqueeze(0)).sum(dim=(1, 2)) / mask_sum  # (C,)
        else:
            self._advance()
            return img

        # Blend: pixel = pixel * (1 - alpha) + avg_color * alpha
        # opacity < 1.0 means only partial blend, so original detail shows through
        alpha = mask_patch.unsqueeze(0) * self.opacity  # (1, ph, pw)
        img[:, i_top:i_bottom, i_left:i_right] = (
            img_patch * (1 - alpha) + avg_color.view(C, 1, 1) * alpha
        )

        self._advance()
        return img

    def _advance(self):
        """Move the blob and slightly perturb its direction."""
        self.cy += self.vy
        self.cx += self.vx

        # Bounce off edges
        if self.cy < self.radius or self.cy > self.img_h - self.radius:
            self.vy = -self.vy
            self.cy = max(self.radius, min(self.img_h - self.radius, self.cy))
        if self.cx < self.radius or self.cx > self.img_w - self.radius:
            self.vx = -self.vx
            self.cx = max(self.radius, min(self.img_w - self.radius, self.cx))

        # Small random perturbation to direction (drift, don't jump)
        self.vy += random.gauss(0, self.speed * 0.1)
        self.vx += random.gauss(0, self.speed * 0.1)

        # Clamp speed so it doesn't accelerate unboundedly
        current_speed = math.sqrt(self.vy ** 2 + self.vx ** 2)
        if current_speed > self.speed * 2:
            scale = (self.speed * 2) / current_speed
            self.vy *= scale
            self.vx *= scale

    def __repr__(self):
        return f"DriftingBlob(radius={self.radius}, speed={self.speed}, softness={self.softness}, opacity={self.opacity})"


# ---------------------------------------------------------------------------
# Robot presets for horizontal flip action/state mirroring
# ---------------------------------------------------------------------------

ROBOT_PRESETS = {
    "aloha": {
        # ALOHA has 14-dim actions: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
        # Joints 1,3 (0-indexed) in each arm control lateral (left-right) movement,
        # so they must be negated when the image is flipped horizontally.
        # Additionally, because ALOHA is bimanual, flipping the image swaps the
        # visual appearance of the left and right arms, so we must also swap the
        # left-arm and right-arm action/state blocks entirely.
        "action_mirror_mask": [1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1],
        "state_mirror_mask": [1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1],
        "swap_action_ranges": [(slice(0, 7), slice(7, 14))],
        "swap_state_ranges": [(slice(0, 7), slice(7, 14))],
    },
}


class HorizontalFlipWithActionMirror:
    """Horizontally flip camera images AND mirror the corresponding action/state vectors.

    Bimanual robots (e.g. ALOHA):
        For bimanual setups, flipping also swaps which physical arm appears on
        which side of the image. The left arm's actions must become the right
        arm's actions and vice versa. This is handled by swap_action_ranges /
        swap_state_ranges, which exchange entire blocks of the action vector.

    Usage:
        The class exposes separate methods for images vs. numerical data because
        augment_dataset.py processes them at different points in the pipeline:
        - flip_image(img)       -> horizontally flipped CHW tensor
        - mirror_actions(action) -> mirrored action tensor
        - mirror_state(state)   -> mirrored state tensor
    """

    def __init__(self, action_mirror_mask, state_mirror_mask,
                 swap_action_ranges=None, swap_state_ranges=None):
        self.action_mirror_mask = torch.tensor(action_mirror_mask, dtype=torch.float32)
        self.state_mirror_mask = torch.tensor(state_mirror_mask, dtype=torch.float32)
        self.swap_action_ranges = swap_action_ranges or []
        self.swap_state_ranges = swap_state_ranges or []

    def flip_image(self, img):
        """Horizontally flip a CHW image tensor."""
        return F.hflip(img)

    def _mirror_vector(self, vec, mask, swap_ranges):
        """Apply element-wise mask and optional range swaps to a 1-D tensor."""
        out = vec.clone() * mask.to(vec.device)
        for range_a, range_b in swap_ranges:
            tmp = out[range_a].clone()
            out[range_a] = out[range_b]
            out[range_b] = tmp
        return out

    def mirror_actions(self, action):
        """Mirror an action vector: negate lateral dims, swap arm blocks."""
        return self._mirror_vector(action, self.action_mirror_mask, self.swap_action_ranges)

    def mirror_state(self, state):
        """Mirror a state vector: negate lateral dims, swap arm blocks."""
        return self._mirror_vector(state, self.state_mirror_mask, self.swap_state_ranges)

    def __repr__(self):
        return (f"HorizontalFlipWithActionMirror("
                f"action_mask={self.action_mirror_mask.tolist()}, "
                f"state_mask={self.state_mirror_mask.tolist()}, "
                f"swap_action_ranges={self.swap_action_ranges}, "
                f"swap_state_ranges={self.swap_state_ranges})")
