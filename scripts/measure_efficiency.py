"""
measure_efficiency.py — produce the efficiency table reviewers demanded.

Reports:
  - Parameter count (trainable + total)
  - FLOPs per forward pass of the denoiser (fvcore; falls back to manual if missing)
  - End-to-end latency per image (GPU + CPU) across {5, 10, 20, 50, 100} sampling steps
  - Peak GPU memory per run
  - Throughput (images / second) at batch 1

Run on Kaggle (T4):
    python measure_efficiency.py --resolution 400 600 --device cuda
Run locally on M4 Pro (CPU-only):
    python measure_efficiency.py --resolution 400 600 --device cpu --skip-gpu

Output: a CSV at ./eval_results/efficiency.csv and a markdown table printed to stdout
ready to paste into the paper.
"""
import argparse
import csv
import os
import time

import torch

from config import Config
from diffusion import DiffusionEngine
from model import ResidualConditionedUNet


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_flops(model, dummy_x, dummy_t, dummy_low):
    """Returns (flops_per_forward, method_string). Tries fvcore first, falls back to thop, then manual."""
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, (dummy_x, dummy_t, dummy_low))
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        return float(flops.total()), "fvcore"
    except Exception as e:
        print(f"fvcore failed ({e}); trying thop")

    try:
        from thop import profile
        flops, _ = profile(model, inputs=(dummy_x, dummy_t, dummy_low), verbose=False)
        return float(flops), "thop"
    except Exception as e:
        print(f"thop failed ({e}); no FLOPs measurement")
        return None, "unavailable"


def bench_latency(diff, model, low_tensor, inference_steps, sampler, warmup=2, iters=10, device="cuda"):
    fn = diff.ddim_sample if sampler == "ddim" else diff.sample

    # warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = fn(model, low_tensor, inference_steps=inference_steps)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            _ = fn(model, low_tensor, inference_steps=inference_steps)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    mean_ms = 1000.0 * elapsed / iters
    peak_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if device.startswith("cuda") else None
    )
    return mean_ms, peak_mb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", nargs=2, type=int, default=[400, 600],
                        help="H W for benchmarking (LOL native is ~400x600)")
    parser.add_argument("--steps", nargs="+", type=int, default=[5, 10, 20, 50, 100])
    parser.add_argument("--sampler", choices=["ddim", "dpm_posterior"], default="ddim")
    parser.add_argument("--device", default=None, help="cuda / cpu / mps. Default: auto")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out-csv", default="./eval_results/efficiency.csv")
    args = parser.parse_args()

    device = args.device or Config.DEVICE
    if args.skip_gpu and device.startswith("cuda"):
        device = "cpu"
    print(f"Benchmarking on device={device}, resolution={args.resolution[0]}x{args.resolution[1]}")

    # Rebuild config with correct device
    conf = Config()
    conf.DEVICE = device  # allow override

    model = ResidualConditionedUNet().to(device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        if "ema" in ckpt:
            model.load_state_dict(ckpt["ema"])
        elif "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
    model.eval()

    # Match DiffusionEngine to the chosen device
    diff = DiffusionEngine()
    # ensure buffers on the right device (important when --device cpu but CUDA is available)
    for attr in [
        "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
        "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
        "posterior_mean_coef1", "posterior_mean_coef2", "posterior_variance",
    ]:
        setattr(diff, attr, getattr(diff, attr).to(device))
    diff.device = device

    total_params, trainable_params = count_params(model)
    print(f"Params: total={total_params/1e6:.3f}M, trainable={trainable_params/1e6:.3f}M")

    H, W = args.resolution
    dummy_x = torch.randn(1, 3, H, W, device=device)
    dummy_t = torch.tensor([0], device=device, dtype=torch.long)
    dummy_low = torch.randn(1, 3, H, W, device=device)

    flops, flops_method = measure_flops(model, dummy_x, dummy_t, dummy_low)
    if flops is not None:
        print(f"FLOPs per denoiser forward ({flops_method}): {flops/1e9:.2f} G")

    rows = []
    for n_steps in args.steps:
        mean_ms, peak_mb = bench_latency(
            diff, model, dummy_low, n_steps, args.sampler,
            iters=args.iters, device=device,
        )
        total_flops = (flops * n_steps) if flops is not None else None
        rows.append({
            "device": device,
            "resolution": f"{H}x{W}",
            "sampler": args.sampler,
            "steps": n_steps,
            "latency_ms": round(mean_ms, 2),
            "throughput_imgs_per_s": round(1000.0 / mean_ms, 3),
            "peak_gpu_mb": round(peak_mb, 1) if peak_mb is not None else "",
            "total_flops_g": round(total_flops / 1e9, 2) if total_flops is not None else "",
            "params_m": round(total_params / 1e6, 3),
        })
        tail = f"{peak_mb:.1f}MB" if peak_mb is not None else "-"
        print(f"  steps={n_steps:<3d}  {mean_ms:8.2f} ms/img  peak={tail}")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Markdown table for the paper
    print("\n### Markdown table (copy into paper)\n")
    print("| Device | Res | Sampler | Steps | Latency (ms) | Throughput (img/s) | Peak mem (MB) | FLOPs (G) | Params (M) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['device']} | {r['resolution']} | {r['sampler']} | {r['steps']} | "
            f"{r['latency_ms']} | {r['throughput_imgs_per_s']} | {r['peak_gpu_mb']} | "
            f"{r['total_flops_g']} | {r['params_m']} |"
        )
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
