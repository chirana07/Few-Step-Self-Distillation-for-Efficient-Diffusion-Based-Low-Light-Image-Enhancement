#!/usr/bin/env python3
"""Run the leakage-free, paired-control experiment protocol with resume support."""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
SRC = WORKSPACE / "src"
EXPECTED_TEACHER_SHA256 = "a92c2988ced5cb0fb8e403fb143913d8fc2837fdb31dbeb1634aa3ca031697a2"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command, cwd=SRC):
    print("\nRunning:", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)


def parse_seeds(value):
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def dataset_paths(root):
    root = Path(root).expanduser().resolve()
    candidates = [root, root / "LOL-v2", root / "LOL_v2"]
    for candidate in candidates:
        if (candidate / "Real_captured" / "Train" / "Low").is_dir():
            return {
                "root": candidate,
                "train_low": candidate / "Real_captured" / "Train" / "Low",
                "train_high": candidate / "Real_captured" / "Train" / "Normal",
                "test_low": candidate / "Real_captured" / "Test" / "Low",
                "test_high": candidate / "Real_captured" / "Test" / "Normal",
            }
    raise FileNotFoundError(f"LOL-v2 Real layout not found below {root}")


def ensure_split(args, paths):
    if args.manifest.exists():
        print(f"Using existing immutable manifest: {args.manifest}")
        return
    run([
        sys.executable,
        HERE / "make_lolv2_split.py",
        "--dataset-root", paths["root"],
        "--output", args.manifest,
        "--seed", args.split_seed,
        "--val-fraction", args.val_fraction,
    ], cwd=WORKSPACE)


def train_branches(args, seeds, paths):
    for seed in seeds:
        for run_kind, w_distill in (("anchor_only", "0.0"), ("fsd", "1.0")):
            tag = f"{run_kind}_seed{seed}"
            out_dir = args.output_root / "checkpoints" / f"seed_{seed}"
            checkpoint = out_dir / f"best_{tag}.pth"
            if checkpoint.exists():
                print(f"Skipping completed training checkpoint: {checkpoint}")
                continue
            run([
                sys.executable, "train_distill.py",
                "--layout", "lolv2_real",
                "--dataset-root", paths["root"],
                "--manifest", args.manifest,
                "--init-from", args.init_checkpoint,
                "--teacher-from", args.init_checkpoint,
                "--run-kind", run_kind,
                "--epochs", args.epochs,
                "--teacher-steps", args.teacher_steps,
                "--student-inference-steps", args.inference_steps,
                "--lr", args.lr,
                "--w-distill", w_distill,
                "--w-anchor", args.w_anchor,
                "--batch-size", args.batch_size,
                "--crop-size", args.crop_size,
                "--val-every", "1",
                "--seed", seed,
                "--val-seed", args.eval_seed,
                "--output-dir", out_dir,
                "--tag", tag,
            ])


def eval_command(args, checkpoint, split_name, low, high, output, alpha, manifest_split, lpips):
    command = [
        sys.executable, "evaluation.py",
        "--splits", f"{split_name}:{low}:{high}",
        "--checkpoint", checkpoint,
        "--inference-steps", args.inference_steps,
        "--sampler", "ddim",
        "--gate-alpha", alpha,
        "--gate-floor", args.gate_floor,
        "--seed", args.eval_seed,
        "--manifest", args.manifest,
        "--manifest-split", manifest_split,
        "--results-root", output,
    ]
    if not lpips:
        command.append("--no-lpips")
    return command


def tune_arr(args, seeds, paths):
    rows = []
    alpha_values = [float(item) for item in args.alphas.split(",")]
    for seed in seeds:
        checkpoint = (
            args.output_root / "checkpoints" / f"seed_{seed}" /
            f"best_fsd_seed{seed}.pth"
        )
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing FSD checkpoint: {checkpoint}")
        for alpha in alpha_values:
            label = f"a{int(round(alpha * 100)):03d}"
            out = args.output_root / "validation" / f"seed_{seed}" / label
            summary = out / "summary.csv"
            if not summary.exists():
                run(eval_command(
                    args, checkpoint, "lolv2_real_val",
                    paths["train_low"], paths["train_high"], out,
                    alpha, "val", lpips=False,
                ))
            with summary.open(newline="", encoding="utf-8") as f:
                result = next(csv.DictReader(f))
            rows.append({
                "seed": seed,
                "alpha": alpha,
                "psnr": float(result["psnr_mean"]),
                "ssim": float(result["ssim_mean"]),
                "n": int(result["n"]),
            })

    aggregates = []
    for alpha in alpha_values:
        selected = [row for row in rows if row["alpha"] == alpha]
        aggregates.append({
            "alpha": alpha,
            "mean_psnr": sum(row["psnr"] for row in selected) / len(selected),
            "mean_ssim": sum(row["ssim"] for row in selected) / len(selected),
            "training_seeds": len(selected),
        })
    winner = max(aggregates, key=lambda row: (row["mean_psnr"], row["mean_ssim"], -row["alpha"]))

    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "arr_validation_runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_root / "arr_validation_aggregate.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    (args.output_root / "selected_alpha.json").write_text(
        json.dumps({
            "selected_alpha": winner["alpha"],
            "selection_metric": "mean validation PSNR",
            "eval_seed": args.eval_seed,
            "manifest": str(args.manifest.resolve()),
            "training_seeds": seeds,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Selected alpha on validation only: {winner['alpha']}")


def official_test(args, seeds, paths):
    selection_path = args.output_root / "selected_alpha.json"
    if not selection_path.exists():
        raise FileNotFoundError("Run the tune stage before official-test")
    alpha = json.loads(selection_path.read_text(encoding="utf-8"))["selected_alpha"]
    jobs = [("transferred_teacher", None, args.init_checkpoint, 0.0)]
    for seed in seeds:
        checkpoint_dir = args.output_root / "checkpoints" / f"seed_{seed}"
        jobs.extend([
            ("anchor_only", seed, checkpoint_dir / f"best_anchor_only_seed{seed}.pth", 0.0),
            ("fsd", seed, checkpoint_dir / f"best_fsd_seed{seed}.pth", 0.0),
            ("fsd_arr", seed, checkpoint_dir / f"best_fsd_seed{seed}.pth", alpha),
        ])

    registry_rows = []
    for method, seed, checkpoint, gate_alpha in jobs:
        suffix = method if seed is None else f"{method}_seed{seed}"
        out = args.output_root / "official_test" / suffix
        summary = out / "summary.csv"
        if not summary.exists():
            run(eval_command(
                args, checkpoint, "lolv2_real_test",
                paths["test_low"], paths["test_high"], out,
                gate_alpha, "test", lpips=True,
            ))
        with summary.open(newline="", encoding="utf-8") as f:
            result = next(csv.DictReader(f))
        registry_rows.append({
            "method": method,
            "training_seed": "" if seed is None else seed,
            "checkpoint": str(Path(checkpoint).resolve()),
            "alpha": gate_alpha,
            "psnr": result["psnr_mean"],
            "ssim": result["ssim_mean"],
            "lpips": result["lpips_mean"],
            "n": result["n"],
            "evaluation_seed": args.eval_seed,
        })
    with (args.output_root / "official_test_registry.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(registry_rows[0]))
        writer.writeheader()
        writer.writerows(registry_rows)
    print(f"Official test registry: {args.output_root / 'official_test_registry.csv'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["split", "train", "tune", "official-test", "all"], default="all")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=WORKSPACE / "configs" / "lolv2_real_split_seed3407.json",
    )
    parser.add_argument("--output-root", type=Path, default=WORKSPACE / "results")
    parser.add_argument("--seeds", default="3407",
                        help="Comma-separated training seeds. Use 3407,3408,3409 for the strong protocol.")
    parser.add_argument("--split-seed", type=int, default=3407)
    parser.add_argument("--eval-seed", type=int, default=99173)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--teacher-steps", type=int, default=5)
    parser.add_argument("--inference-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--lr", default="1e-5")
    parser.add_argument("--w-anchor", default="0.5")
    parser.add_argument("--gate-floor", default="0.5")
    parser.add_argument("--alphas", default="0.0,0.1,0.2,0.3,0.4,0.5")
    args = parser.parse_args()

    args.init_checkpoint = args.init_checkpoint.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if not args.init_checkpoint.is_file():
        raise FileNotFoundError(args.init_checkpoint)
    checkpoint_hash = sha256(args.init_checkpoint)
    if checkpoint_hash != EXPECTED_TEACHER_SHA256:
        raise ValueError(
            f"Wrong transferred teacher SHA-256: {checkpoint_hash}; "
            f"expected {EXPECTED_TEACHER_SHA256}"
        )
    seeds = parse_seeds(args.seeds)
    paths = dataset_paths(args.dataset_root)

    stages = ["split", "train", "tune", "official-test"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "split":
            ensure_split(args, paths)
        elif stage == "train":
            ensure_split(args, paths)
            train_branches(args, seeds, paths)
        elif stage == "tune":
            ensure_split(args, paths)
            tune_arr(args, seeds, paths)
        elif stage == "official-test":
            official_test(args, seeds, paths)


if __name__ == "__main__":
    main()
