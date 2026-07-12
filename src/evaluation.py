"""
evaluation.py — full-resolution evaluation on one or more LLIE test sets.

Changes vs. prior version (2026-04-24):
- CRITICAL FIX: no longer resizes test images to 128x128 before computing PSNR/SSIM.
  That resize was the main reason reported metrics were low. We now pad each image
  to the next multiple of 8, run the model, then crop back to the original size
  (exactly the protocol used in inference.py).
- Added LPIPS (perceptual metric) when the `lpips` package is available.
- Accepts multiple evaluation directories via --splits (e.g. eval15, LOLv2-Real-Test,
  LOLv2-Synthetic-Test) and writes one results block per split.
- Accepts --inference-steps to support the sampling-step ablation.
- Accepts --sampler (ddim | dpm_posterior) to compare samplers.
- Accepts --checkpoint to let the ablation driver point at different checkpoints.
- Logs per-image metrics in CSV so we can compute stdev / confidence intervals later.
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import random
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func
from tqdm import tqdm

from config import Config
from diffusion import DiffusionEngine
from model import ResidualConditionedUNet

# LPIPS is optional — skip gracefully if the package isn't installed.
try:
    import lpips as lpips_pkg  # noqa: F401
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False


# ----------------------------- helpers -----------------------------

def pad_to_multiple_of_8(img):
    """PIL image -> padded PIL image + original (w, h). Matches inference.py."""
    w, h = img.size
    new_w = ((w + 7) // 8) * 8
    new_h = ((h + 7) // 8) * 8
    pad_w = new_w - w
    pad_h = new_h - h
    img = TF.pad(img, [0, 0, pad_w, pad_h], fill=0)
    return img, (w, h)


def load_checkpoint(model, checkpoint_path, device, prefer_ema=True):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and prefer_ema and "ema" in checkpoint:
        print(f"[{checkpoint_path}] loading EMA weights")
        model.load_state_dict(checkpoint["ema"])
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        print(f"[{checkpoint_path}] loading standard model weights")
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    return model


def calculate_metrics(pred_pil, gt_pil, lpips_fn=None, device="cpu"):
    pred_np = np.array(pred_pil)
    gt_np = np.array(gt_pil)

    p = float(psnr_func(gt_np, pred_np, data_range=255))
    s = float(ssim_func(gt_np, pred_np, data_range=255, channel_axis=2))

    lp = None
    if lpips_fn is not None:
        # LPIPS expects tensors in [-1, 1], shape (1, 3, H, W)
        pred_t = TF.to_tensor(pred_pil).unsqueeze(0).to(device) * 2 - 1
        gt_t = TF.to_tensor(gt_pil).unsqueeze(0).to(device) * 2 - 1
        with torch.no_grad():
            lp = float(lpips_fn(pred_t, gt_t).item())

    return p, s, lp


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def stable_latent(tensor, base_seed, split_name, image_name):
    key = f"{base_seed}:{split_name}:{image_name}".encode("utf-8")
    latent_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2 ** 63 - 1)
    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(latent_seed)
    latent = torch.randn(
        tensor.shape, generator=generator, device=tensor.device, dtype=tensor.dtype
    )
    return latent, latent_seed


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def discover_splits(splits_arg):
    """
    Split descriptor format: NAME:LOW_DIR:HIGH_DIR
    e.g. "eval15:./eval15/low:./eval15/high"
         "lolv2_real:./LOLv2/Real_captured/Test/Low:./LOLv2/Real_captured/Test/Normal"
    If no colon present, treat as a simple folder name under ./ that contains low/ and high/.
    """
    out = []
    for s in splits_arg:
        parts = s.split(":")
        if len(parts) == 3:
            out.append(tuple(parts))
        elif len(parts) == 1:
            name = parts[0]
            out.append((name, os.path.join(name, "low"), os.path.join(name, "high")))
        else:
            raise ValueError(f"Bad split descriptor: {s}")
    return out


# ----------------------------- main eval -----------------------------

def evaluate_split(
    model,
    diff,
    split_name,
    low_dir,
    high_dir,
    results_dir,
    inference_steps,
    sampler,
    device,
    lpips_fn=None,
    gate_alpha=0.0,
    gate_floor=0.5,
    seed=3407,
    pair_map=None,
):
    os.makedirs(results_dir, exist_ok=True)

    if pair_map is not None:
        names = sorted(pair_map)
    else:
        names = sorted(
            os.path.basename(p) for p in glob.glob(os.path.join(low_dir, "*"))
            if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        )
    if not names:
        print(f"[{split_name}] no images found in {low_dir}")
        return None

    per_image = []
    print(f"\n=== {split_name} ({len(names)} images, {inference_steps} steps, {sampler}) ===")

    for name in tqdm(names):
        low_path = os.path.join(low_dir, name)
        high_path = os.path.join(high_dir, pair_map[name] if pair_map is not None else name)
        
        if not os.path.exists(high_path):
            # Fallback 1: try replacing 'low' with 'normal' (LOL-v2 Real)
            if name.startswith("low"):
                fallback_name = name.replace("low", "normal", 1)
                high_path = os.path.join(high_dir, fallback_name)
            
            if not os.path.exists(high_path):
                # Fallback 2: try case-insensitive or other common patterns if needed
                # For now, just skip if still not found
                continue

        low_pil = Image.open(low_path).convert("RGB")
        high_pil = Image.open(high_path).convert("RGB")
        if low_pil.size != high_pil.size:
            print(
                f"[{split_name}] skipping {name}: low size {low_pil.size} "
                f"!= high size {high_pil.size}"
            )
            continue

        low_padded, (orig_w, orig_h) = pad_to_multiple_of_8(low_pil)
        low_tensor = (TF.to_tensor(low_padded) - 0.5) * 2.0
        low_tensor = low_tensor.unsqueeze(0).to(device)
        initial_noise, latent_seed = stable_latent(low_tensor, seed, split_name, name)

        start = time.time()
        with torch.no_grad():
            if sampler == "ddim":
                gen_tensor = diff.ddim_sample(
                    model, low_tensor, inference_steps=inference_steps,
                    gate_alpha=gate_alpha, gate_floor=gate_floor,
                    initial_noise=initial_noise,
                )
            elif sampler == "dpm_posterior":
                gen_tensor = diff.sample(
                    model, low_tensor, inference_steps=inference_steps,
                    initial_noise=initial_noise,
                )
            else:
                raise ValueError(f"Unknown sampler: {sampler}")
        elapsed = time.time() - start

        gen_tensor = (gen_tensor + 1.0) / 2.0
        gen_tensor = torch.clamp(gen_tensor, 0.0, 1.0)
        gen_pil = TF.to_pil_image(gen_tensor.squeeze(0).cpu())
        # crop back to original size (undo the padding)
        gen_pil = gen_pil.crop((0, 0, orig_w, orig_h))

        p, s, lp = calculate_metrics(gen_pil, high_pil, lpips_fn=lpips_fn, device=device)

        per_image.append({
            "split": split_name,
            "image": name,
            "psnr": p,
            "ssim": s,
            "lpips": lp if lp is not None else "",
            "runtime_s": elapsed,
            "latent_seed": latent_seed,
        })

        out_path = os.path.join(results_dir, f"{split_name}_{name}")
        gen_pil.save(out_path)

    psnr_vals = np.array([r["psnr"] for r in per_image])
    ssim_vals = np.array([r["ssim"] for r in per_image])
    rt_vals = np.array([r["runtime_s"] for r in per_image])
    lpips_vals = np.array([r["lpips"] for r in per_image if r["lpips"] != ""]) if any(
        r["lpips"] != "" for r in per_image
    ) else None

    summary = {
        "split": split_name,
        "n": len(per_image),
        "psnr_mean": float(psnr_vals.mean()),
        "psnr_std": float(psnr_vals.std()),
        "ssim_mean": float(ssim_vals.mean()),
        "ssim_std": float(ssim_vals.std()),
        "lpips_mean": float(lpips_vals.mean()) if lpips_vals is not None else None,
        "runtime_mean": float(rt_vals.mean()),
        "inference_steps": inference_steps,
        "sampler": sampler,
        "seed": seed,
        "gate_alpha": gate_alpha,
        "gate_floor": gate_floor,
    }
    return summary, per_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["eval15:./eval15/low:./eval15/high"],
        help="Space-separated split descriptors: NAME:LOW_DIR:HIGH_DIR",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to .pth. Default: ./checkpoints/last_pth_only/final.pth or ./checkpoints/final.pth",
    )
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument(
        "--sampler",
        choices=["ddim", "dpm_posterior"],
        default="dpm_posterior",
        help="ddim = proper DDIM; dpm_posterior = the legacy deterministic posterior-mean chain.",
    )
    parser.add_argument(
        "--results-root",
        default="./eval_results",
        help="Where to write per-split image outputs and the summary CSV",
    )
    parser.add_argument("--no-lpips", action="store_true", help="Skip LPIPS even if installed")
    parser.add_argument("--tag", default="", help="Optional tag appended to the output dir name")
    parser.add_argument("--seed", type=int, default=3407,
                        help="Base seed; each image gets a stable derived starting latent.")
    parser.add_argument("--manifest", default=None,
                        help="Optional integrity split JSON. Required for validation tuning.")
    parser.add_argument("--manifest-split", default=None,
                        choices=["train", "val", "test"],
                        help="Manifest split to evaluate; use val for ARR selection.")
    parser.add_argument(
        "--gate-alpha", type=float, default=0.0,
        help="Adaptive Residual Rescaling strength. 0 = vanilla DDIM (unchanged); "
             "values in [0, 1] dampen residual at high noise levels. Tune on val.",
    )
    parser.add_argument(
        "--gate-floor", type=float, default=0.5,
        help="Floor on the ARR factor (prevents zeroing the residual at noisy steps).",
    )
    args = parser.parse_args()

    set_deterministic(args.seed)
    if bool(args.manifest) != bool(args.manifest_split):
        raise ValueError("--manifest and --manifest-split must be provided together")

    conf = Config()
    device = conf.DEVICE
    print(f"Device: {device}")

    # Resolve checkpoint path
    ckpt = args.checkpoint
    if ckpt is None:
        for cand in [
            os.path.join(conf.SAVE_DIR, "last_pth_only", "final.pth"),
            os.path.join(conf.SAVE_DIR, "final.pth"),
            os.path.join(conf.SAVE_DIR, "best.pth"),
        ]:
            if os.path.exists(cand):
                ckpt = cand
                break
    if ckpt is None or not os.path.exists(ckpt):
        raise FileNotFoundError(f"No checkpoint found. Pass --checkpoint or place one at {conf.SAVE_DIR}")

    # Determine architecture from tensors, not stale checkpoint metadata. All
    # checkpoints used by this protocol have a six-channel [x_t || y] head.
    peek = torch.load(ckpt, map_location="cpu")
    state = peek.get("ema", peek.get("model", peek)) if isinstance(peek, dict) else peek
    head_channels = int(state["head.weight"].shape[1])
    if head_channels != 6:
        raise ValueError(f"Expected six-channel denoiser head, found {head_channels}")
    if isinstance(peek, dict) and peek.get("use_illum_prior", False):
        print("[checkpoint] ignoring stale use_illum_prior=True metadata; head has 6 channels")
    del peek

    model = ResidualConditionedUNet(use_illum_prior=False).to(device)
    model = load_checkpoint(model, ckpt, device)
    model.eval()

    diff = DiffusionEngine()

    # Optional LPIPS
    lpips_fn = None
    if not args.no_lpips and _LPIPS_AVAILABLE:
        lpips_fn = lpips_pkg.LPIPS(net="alex").to(device)
        lpips_fn.eval()
        print("LPIPS (alex) enabled")
    else:
        if args.no_lpips:
            print("LPIPS disabled (--no-lpips)")
        else:
            print("LPIPS not installed, skipping. `pip install lpips` to enable.")

    results_root = args.results_root
    if args.tag:
        results_root = f"{results_root}_{args.tag}"
    os.makedirs(results_root, exist_ok=True)

    pair_map = None
    if args.manifest:
        with open(args.manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        entries = manifest["splits"][args.manifest_split]
        pair_map = {entry["low"]: entry["high"] for entry in entries}

    summaries = []
    all_per_image = []
    for name, low_dir, high_dir in discover_splits(args.splits):
        split_results_dir = os.path.join(results_root, name)
        result = evaluate_split(
            model, diff,
            split_name=name,
            low_dir=low_dir,
            high_dir=high_dir,
            results_dir=split_results_dir,
            inference_steps=args.inference_steps,
            sampler=args.sampler,
            device=device,
            lpips_fn=lpips_fn,
            gate_alpha=args.gate_alpha,
            gate_floor=args.gate_floor,
            seed=args.seed,
            pair_map=pair_map,
        )
        if result is None:
            continue
        summary, per_image = result
        summaries.append(summary)
        all_per_image.extend(per_image)

    # Write summary CSV
    summary_csv = os.path.join(results_root, "summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)

    # Write per-image CSV
    per_image_csv = os.path.join(results_root, "per_image.csv")
    with open(per_image_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_per_image[0].keys()))
        writer.writeheader()
        for row in all_per_image:
            writer.writerow(row)

    metadata = {
        "checkpoint": os.path.abspath(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
        "seed": args.seed,
        "manifest": os.path.abspath(args.manifest) if args.manifest else None,
        "manifest_split": args.manifest_split,
        "inference_steps": args.inference_steps,
        "sampler": args.sampler,
        "gate_alpha": args.gate_alpha,
        "gate_floor": args.gate_floor,
        "lpips_enabled": lpips_fn is not None,
        "splits": args.splits,
    }
    with open(os.path.join(results_root, "protocol.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    # Legacy text file (keeps backward compat with old eval_metrics.txt)
    with open("eval_metrics.txt", "w") as f:
        for s in summaries:
            f.write(f"[{s['split']}] n={s['n']} steps={s['inference_steps']} sampler={s['sampler']}\n")
            f.write(f"  PSNR:  {s['psnr_mean']:.4f} (std {s['psnr_std']:.4f})\n")
            f.write(f"  SSIM:  {s['ssim_mean']:.4f} (std {s['ssim_std']:.4f})\n")
            if s["lpips_mean"] is not None:
                f.write(f"  LPIPS: {s['lpips_mean']:.4f}\n")
            f.write(f"  Runtime/image: {s['runtime_mean']:.4f}s\n\n")

    print("\n==== Summary ====")
    for s in summaries:
        line = f"[{s['split']}] PSNR {s['psnr_mean']:.4f} | SSIM {s['ssim_mean']:.4f}"
        if s["lpips_mean"] is not None:
            line += f" | LPIPS {s['lpips_mean']:.4f}"
        line += f" | {s['runtime_mean']:.3f}s/img | n={s['n']} | {s['inference_steps']}steps/{s['sampler']}"
        print(line)
    print(f"\nSaved summary CSV: {summary_csv}")
    print(f"Saved per-image CSV: {per_image_csv}")


if __name__ == "__main__":
    main()
