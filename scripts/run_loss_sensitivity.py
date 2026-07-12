#!/usr/bin/env python3
"""Run a validation-only, compute-matched teacher-loss sensitivity study."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch


EXPECTED_SOURCE_SHA256 = "e7f86ad8ba2baf11384e784346ef7f777cd0bd3abb2e4a33cc00809e6d16f374"
EXPECTED_MANIFEST_SHA256 = "464f54299480414b61376a2d6388afa7f15499af3bbad56c2b33527ddf6e0ca9"

# Exact transferred-teacher weights, with one non-core term removed at a time.
VARIANTS = {
    "full": (1.0, 0.3, 0.1, 0.05, 0.2, 0.02),
    "no_ssim": (1.0, 0.0, 0.1, 0.05, 0.2, 0.02),
    "no_vgg": (1.0, 0.3, 0.0, 0.05, 0.2, 0.02),
    "no_color": (1.0, 0.3, 0.1, 0.0, 0.2, 0.02),
    "no_gradient": (1.0, 0.3, 0.1, 0.05, 0.0, 0.02),
    "no_tv": (1.0, 0.3, 0.1, 0.05, 0.2, 0.0),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--val-seed", type=int, default=99173)
    parser.add_argument("--val-every", type=int, default=2)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    return parser.parse_args()


def main():
    args = parse()
    work = Path(__file__).resolve().parent
    src = work / "src"
    output_root = Path(args.output_root).resolve()
    checkpoints = output_root / "checkpoints"
    logs_dir = output_root / "logs"
    checkpoints.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_sha = sha256(args.source_checkpoint)
    if checkpoint_sha != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"Wrong source final.pth SHA-256: {checkpoint_sha}")
    manifest_sha = sha256(args.manifest)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"Wrong split manifest SHA-256: {manifest_sha}")

    unknown = sorted(set(args.variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")

    rows = []
    env = os.environ.copy()
    env["VAL_EVERY"] = str(args.val_every)

    for variant in args.variants:
        weights = VARIANTS[variant]
        variant_dir = checkpoints / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        tag = f"loss_{variant}_seed{args.seed}"
        command = [
            sys.executable, "train.py",
            "--layout", "lolv2_real",
            "--dataset-root", str(Path(args.dataset_root).resolve()),
            "--manifest", str(Path(args.manifest).resolve()),
            "--init-from", str(Path(args.source_checkpoint).resolve()),
            "--use-illum-prior", "0",
            "--epochs", str(args.epochs),
            "--crop-size", "256",
            "--batch-size", "4",
            "--lr", "1e-5",
            "--inference-steps", "5",
            "--val-seed", str(args.val_seed),
            "--seed", str(args.seed),
            "--num-workers", "2",
            "--output-dir", str(variant_dir),
            "--tag", tag,
            "--w-char", str(weights[0]),
            "--w-ssim", str(weights[1]),
            "--w-perc", str(weights[2]),
            "--w-color", str(weights[3]),
            "--w-grad", str(weights[4]),
            "--w-tv", str(weights[5]),
        ]
        print("\nRunning:", " ".join(command), flush=True)
        subprocess.run(command, cwd=src, env=env, check=True)

        best_path = variant_dir / f"best_{tag}.pth"
        log_path = variant_dir / f"train_log_{tag}.csv"
        if not best_path.is_file() or not log_path.is_file():
            raise FileNotFoundError(f"Missing outputs for {variant}")
        raw = torch.load(best_path, map_location="cpu")
        rows.append({
            "variant": variant,
            "w_char": weights[0],
            "w_ssim": weights[1],
            "w_vgg": weights[2],
            "w_color": weights[3],
            "w_gradient": weights[4],
            "w_tv": weights[5],
            "best_epoch": int(raw["epoch"]),
            "val_psnr": float(raw["val_psnr"]),
            "val_ssim": float(raw["val_ssim"]),
            "checkpoint_sha256": sha256(best_path),
            "training_seed": args.seed,
            "validation_seed": args.val_seed,
            "validation_steps": 5,
            "epochs_budget": args.epochs,
        })
        (logs_dir / log_path.name).write_bytes(log_path.read_bytes())

    summary_path = output_root / "loss_sensitivity_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    protocol = {
        "study": "validation-only continued-training loss sensitivity",
        "official_test_used": False,
        "source_checkpoint": str(Path(args.source_checkpoint).resolve()),
        "source_checkpoint_sha256": checkpoint_sha,
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": manifest_sha,
        "split_counts": {"train": 620, "validation": 69},
        "training_seed": args.seed,
        "validation_seed": args.val_seed,
        "validation_steps": 5,
        "validation_every_epochs": args.val_every,
        "epochs_budget": args.epochs,
        "selection_metric": "validation PSNR",
        "variants": {name: list(VARIANTS[name]) for name in args.variants},
        "interpretation": (
            "This measures marginal sensitivity during matched continued training. "
            "It is not a from-scratch causal attribution of the pretrained checkpoint."
        ),
    }
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
