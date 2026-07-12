#!/usr/bin/env python3
"""Create a deterministic validation split from LOL-v2 Real training pairs.

The official Test directory is listed in the manifest for auditability but is never
used to select a checkpoint or ARR parameter.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def resolve_dataset_root(root):
    root = Path(root).expanduser().resolve()
    candidates = [root, root / "LOL-v2", root / "LOL_v2"]
    for candidate in candidates:
        if (candidate / "Real_captured" / "Train" / "Low").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find Real_captured/Train/Low below {root}. "
        "Pass the directory that contains Real_captured."
    )


def collect_pairs(low_dir, high_dir):
    high_names = {p.name for p in high_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    pairs = []
    missing = []
    for low_path in sorted(low_dir.iterdir()):
        if low_path.suffix.lower() not in IMAGE_EXTS:
            continue
        candidates = [
            low_path.name,
            low_path.name.replace("low", "normal", 1),
            low_path.name.replace("Low", "Normal", 1),
            low_path.name.replace("low", "high", 1),
            low_path.name.replace("Low", "High", 1),
        ]
        high_name = next((name for name in candidates if name in high_names), None)
        if high_name is None:
            missing.append(low_path.name)
        else:
            pairs.append({"low": low_path.name, "high": high_name})
    if missing:
        raise RuntimeError(f"Missing targets for {len(missing)} low images: {missing[:5]}")
    if not pairs:
        raise RuntimeError(f"No pairs found under {low_dir} and {high_dir}")
    return pairs


def rank(pair, seed):
    key = f"{seed}:{pair['low']}:{pair['high']}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    args = parser.parse_args()

    if not 0.05 <= args.val_fraction <= 0.30:
        raise ValueError("--val-fraction must be between 0.05 and 0.30")

    root = resolve_dataset_root(args.dataset_root)
    train_root = root / "Real_captured" / "Train"
    test_root = root / "Real_captured" / "Test"
    all_train = collect_pairs(train_root / "Low", train_root / "Normal")
    official_test = collect_pairs(test_root / "Low", test_root / "Normal")

    ordered = sorted(all_train, key=lambda item: rank(item, args.seed))
    val_count = max(1, round(len(ordered) * args.val_fraction))
    val_pairs = sorted(ordered[:val_count], key=lambda item: item["low"])
    train_pairs = sorted(ordered[val_count:], key=lambda item: item["low"])

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "LOL-v2 Real",
        "dataset_root_at_creation": str(root),
        "split_method": "SHA-256 ranking of paired training filenames",
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "counts": {
            "train": len(train_pairs),
            "val": len(val_pairs),
            "test": len(official_test),
        },
        "policy": {
            "checkpoint_selection": "val only",
            "arr_selection": "val only",
            "official_test": "evaluate once after all choices are frozen",
        },
        "splits": {
            "train": train_pairs,
            "val": val_pairs,
            "test": official_test,
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"Dataset root: {root}")
    print(f"Train: {len(train_pairs)} | Validation: {len(val_pairs)} | Test: {len(official_test)}")
    print(f"Manifest: {output.resolve()}")
    print(f"Manifest SHA-256: {digest}")


if __name__ == "__main__":
    main()
