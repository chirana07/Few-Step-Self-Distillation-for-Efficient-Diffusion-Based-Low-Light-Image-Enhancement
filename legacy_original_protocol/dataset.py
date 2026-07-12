"""
dataset.py — LLIE paired dataset loader.

Changes vs. prior version (2026-04-24):
- Supports three dataset layouts:
    * LOL v1           : <root>/(train|eval15)/(low|high)/*.png
    * LOL-v2 Real      : <root>/Real_captured/(Train|Test)/(Low|Normal)/*.png
    * LOL-v2 Synthetic : <root>/Synthetic/(Train|Test)/(Low|Normal)/*.png
- Training uses RANDOM CROPS of CROP_SIZE x CROP_SIZE (default 256) — no more
  full-image resize to 128x128 that was destroying detail.
- Eval/test loader returns full-resolution images padded to a multiple of 8.
  (If you prefer to keep things simple you can still use evaluation.py directly;
  this path is here for validation-loop use during training.)
- Deterministic augmentation flag so ablation runs are reproducible.
"""
import os
import random
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from config import Config


LAYOUTS = {
    "lol_v1": {
        "train": {"low": "our485/low",     "high": "our485/high"},
        "test":  {"low": "eval15/low",     "high": "eval15/high"},
        "aliases": ["lol", "lolv1", "lol_v1"],
    },
    "lolv2_real": {
        "train": {"low": "Real_captured/Train/Low", "high": "Real_captured/Train/Normal"},
        "test":  {"low": "Real_captured/Test/Low",  "high": "Real_captured/Test/Normal"},
        "aliases": ["lolv2_real", "lol_v2_real", "lolv2real"],
    },
    "lolv2_syn": {
        "train": {"low": "Synthetic/Train/Low", "high": "Synthetic/Train/Normal"},
        "test":  {"low": "Synthetic/Test/Low",  "high": "Synthetic/Test/Normal"},
        "aliases": ["lolv2_syn", "lol_v2_syn", "lolv2synth", "lolv2_synthetic"],
    },
    # Generic flat layout: <root>/low, <root>/high
    "flat": {
        "train": {"low": "low", "high": "high"},
        "test":  {"low": "low", "high": "high"},
        "aliases": ["flat"],
    },
}


def _resolve_layout(name):
    name = name.lower()
    for key, meta in LAYOUTS.items():
        if name in meta["aliases"] or name == key:
            return key
    raise ValueError(f"Unknown dataset layout '{name}'. Known: {list(LAYOUTS.keys())}")


def _find_dataset_root(preferred):
    """Try <preferred>, then fall back to Kaggle/local defaults."""
    candidates = [preferred] if preferred else []
    candidates += [Config.DATASET_ROOT, "./input", "."]
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    return candidates[0] if candidates else "."


class LOLDataset(Dataset):
    """
    Generic paired LLIE dataset.

    Examples:
        LOLDataset("train")                            # default: lol_v1 layout
        LOLDataset("train", layout="lolv2_real")
        LOLDataset("test",  layout="lolv2_syn", root="/kaggle/input/lolv2")
    """

    def __init__(
        self,
        mode="train",
        layout="lol_v1",
        root=None,
        crop_size=None,
        augment=True,
        pad_multiple=8,
    ):
        self.mode = mode
        self.augment = augment and (mode == "train")
        self.pad_multiple = pad_multiple

        self.crop_size = crop_size or getattr(Config, "CROP_SIZE", getattr(Config, "IMG_SIZE", 128))
        self.layout_name = _resolve_layout(layout)
        layout_meta = LAYOUTS[self.layout_name]

        split = "train" if mode in ["train", "val"] else "test"
        rel_low = layout_meta[split]["low"]
        rel_high = layout_meta[split]["high"]

        root = _find_dataset_root(root)
        # Two ways to resolve: root contains the layout directly, OR root has the layout as a subdir.
        # We try both and pick whichever exists.
        candidates = [
            (os.path.join(root, rel_low), os.path.join(root, rel_high)),
            (os.path.join(root, self.layout_name, rel_low), os.path.join(root, self.layout_name, rel_high)),
        ]
        # Also scan any subdirectory that happens to contain the expected path
        found = None
        for lo, hi in candidates:
            if os.path.isdir(lo) and os.path.isdir(hi):
                found = (lo, hi)
                break
        if found is None:
            # last-ditch: glob
            import glob
            lo_matches = glob.glob(os.path.join(root, "**", os.path.basename(rel_low)), recursive=True)
            hi_matches = glob.glob(os.path.join(root, "**", os.path.basename(rel_high)), recursive=True)
            if lo_matches and hi_matches:
                found = (lo_matches[0], hi_matches[0])

        if found is None:
            raise FileNotFoundError(
                f"Could not resolve {self.layout_name}/{split} under root={root}. "
                f"Tried: {candidates}"
            )

        self.low_dir, self.high_dir = found
        exts = (".png", ".jpg", ".jpeg", ".bmp")
        self.pairs = []
        high_files = set(os.listdir(self.high_dir))
        for n in sorted(os.listdir(self.low_dir)):
            if not n.lower().endswith(exts):
                continue
            if n in high_files:
                self.pairs.append((n, n))
            else:
                # Try to match Kaggle renamed files like low00001.png -> normal00001.png
                n_normal = n.replace("low", "normal").replace("Low", "Normal")
                n_high = n.replace("low", "high").replace("Low", "High")
                if n_normal in high_files:
                    self.pairs.append((n, n_normal))
                elif n_high in high_files:
                    self.pairs.append((n, n_high))

        if not self.pairs:
            raise RuntimeError(
                f"No paired images under {self.low_dir} <-> {self.high_dir}"
            )
            
        # --- Create a deterministic validation split ---
        self.pairs = sorted(self.pairs, key=lambda x: x[0])
        if split == "train":
            val_size = min(max(int(len(self.pairs) * 0.1), 1), 50) # Use 10% up to 50 images
            if mode == "val":
                self.pairs = self.pairs[-val_size:]
                self.augment = False
            elif mode == "train":
                self.pairs = self.pairs[:-val_size]

        print(f"[LOLDataset] layout={self.layout_name} mode={mode} n={len(self.pairs)} "
              f"low={self.low_dir} high={self.high_dir}")

    def __len__(self):
        return len(self.pairs)

    def _pad_to_multiple(self, pil_low, pil_high):
        w, h = pil_low.size
        pad_w = (self.pad_multiple - w % self.pad_multiple) % self.pad_multiple
        pad_h = (self.pad_multiple - h % self.pad_multiple) % self.pad_multiple
        if pad_w or pad_h:
            pil_low = TF.pad(pil_low, [0, 0, pad_w, pad_h], fill=0)
            pil_high = TF.pad(pil_high, [0, 0, pad_w, pad_h], fill=0)
        return pil_low, pil_high

    def __getitem__(self, idx):
        low_name, high_name = self.pairs[idx]
        low = Image.open(os.path.join(self.low_dir, low_name)).convert("RGB")
        high = Image.open(os.path.join(self.high_dir, high_name)).convert("RGB")

        if self.mode == "train":
            # Random crop at CROP_SIZE
            cw = ch = int(self.crop_size)
            w, h = low.size
            if w < cw or h < ch:
                # image smaller than crop — upsample to crop size (rare on LOL)
                scale = max(cw / w, ch / h) * 1.01
                low = low.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
                high = high.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
                w, h = low.size
            x0 = random.randint(0, w - cw)
            y0 = random.randint(0, h - ch)
            low = low.crop((x0, y0, x0 + cw, y0 + ch))
            high = high.crop((x0, y0, x0 + cw, y0 + ch))

            if self.augment:
                if random.random() < 0.5:
                    low = TF.hflip(low); high = TF.hflip(high)
                if random.random() < 0.5:
                    low = TF.vflip(low); high = TF.vflip(high)
                k = random.randint(0, 3)
                if k:
                    low = low.rotate(90 * k)
                    high = high.rotate(90 * k)
        else:
            low, high = self._pad_to_multiple(low, high)

        low_t = (TF.to_tensor(low) - 0.5) * 2.0
        high_t = (TF.to_tensor(high) - 0.5) * 2.0
        return low_t, high_t
