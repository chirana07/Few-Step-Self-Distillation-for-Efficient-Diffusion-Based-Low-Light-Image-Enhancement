"""
train_distill.py — few-step self-distillation for residual-space diffusion LLIE.

Setup:
  Teacher (frozen): the existing Day 1 checkpoint, run with K DDIM sub-steps from
    a noise level t_start down to t=0. Produces a refined residual prediction.
  Student (trainable): initialized from the same checkpoint. Learns to match the
    teacher's refined output in a SINGLE forward pass at t_start.

After training the student is plugged into the standard 5-step DDIM sampler. Each
of the 5 steps is now a distilled single-shot prediction trained against teacher's
multi-step refinement, so the composite 5-step output should beat the original
5-step output without any extra inference cost.

The training loop:
  for batch (low, high):
     r = clip(high - low)                  # ground-truth residual
     t = single value sampled from {coarse 5-step DDIM schedule}
     x_t = forward_diffuse(r, t)            # noisy state at t
     teacher_x0 = teacher.ddim_substeps(x_t, t -> 0, K substeps)   # no_grad
     student_x0 = student(x_t, t, low)      # single forward pass
     loss = w_distill * char(student_x0, teacher_x0)
          + w_anchor  * char(student_x0, r)               # ground-truth anchor

Usage on Kaggle:
    python train_distill.py \
        --layout lolv2_real \
        --dataset-root /kaggle/working/data \
        --init-from /kaggle/input/lumidiff-checkpoint/final.pth \
        --epochs 20 --teacher-steps 5 --lr 1e-5 \
        --tag distill_K5
"""
import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import LOLDataset
from diffusion import DiffusionEngine
from model import ResidualConditionedUNet
from train import smart_load_checkpoint, set_seed, EMA, validate, count_parameters


# --------------------------- losses ---------------------------

def charbonnier(pred, target, eps=1e-3):
    return torch.sqrt((pred - target) ** 2 + eps ** 2).mean()


# ------------------------ teacher refinement ------------------------

@torch.no_grad()
def teacher_refine(teacher, diff, x_t, low, t_start, num_substeps):
    """Run num_substeps DDIM steps from x_t (at noise level t_start) down to t=0.

    Args:
        teacher: frozen model (eval mode, requires_grad=False on params)
        diff: DiffusionEngine
        x_t: (B, 3, H, W) noisy residual at noise level t_start
        low: (B, 3, H, W) low-light conditioning
        t_start: int, the starting noise-level index (e.g. 99)
        num_substeps: how many DDIM substeps to take from t_start down to 0

    Returns:
        teacher_x0: (B, 3, H, W) refined residual prediction (detached)
    """
    teacher.eval()
    b = x_t.shape[0]

    if t_start <= 0 or num_substeps <= 0:
        # Trivial case: already at t=0 or no substeps requested. Just call teacher once.
        t = torch.zeros(b, device=x_t.device, dtype=torch.long)
        _, x0 = teacher(x_t, t, low, return_residual=True)
        return torch.clamp(x0, -1.0, 1.0).detach()

    # Sub-schedule from t_start down through 0, num_substeps + 1 points
    schedule = torch.linspace(t_start, 0, num_substeps + 1, device=x_t.device).long()
    schedule = torch.unique_consecutive(schedule)
    idx_list = schedule.tolist()

    residual = x_t.clone()
    for step_i in range(len(idx_list) - 1):
        i = idx_list[step_i]
        next_i = idx_list[step_i + 1]
        t = torch.full((b,), int(i), device=x_t.device, dtype=torch.long)

        _, x0_pred = teacher(residual, t, low, return_residual=True)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        a_t = diff.alphas_cumprod[t][:, None, None, None]
        a_prev = diff.alphas_cumprod[
            torch.full((b,), int(next_i), device=x_t.device, dtype=torch.long)
        ][:, None, None, None]

        eps_pred = (residual - torch.sqrt(a_t) * x0_pred) / torch.sqrt(1.0 - a_t + 1e-8)
        # eta = 0 deterministic DDIM update
        residual = (
            torch.sqrt(a_prev) * x0_pred
            + torch.sqrt(torch.clamp(1.0 - a_prev, min=0.0)) * eps_pred
        )

    # The final residual after stepping through to t=0 IS the teacher's refined x0.
    return residual.detach()


# ------------------------ training loop ------------------------

def train_distill(args):
    conf = Config()
    set_seed(args.seed or conf.SEED)

    if args.crop_size:
        conf.CROP_SIZE = args.crop_size
    if args.batch_size:
        conf.BATCH_SIZE = args.batch_size

    use_prior = bool(args.use_illum_prior)

    print(f"=== Few-step self-distillation ===")
    print(f"  init_from       = {args.init_from}")
    print(f"  layout          = {args.layout}")
    print(f"  dataset_root    = {args.dataset_root}")
    print(f"  use_illum_prior = {use_prior}")
    print(f"  teacher_steps   = {args.teacher_steps} (DDIM sub-steps the teacher takes)")
    print(f"  student_steps   = 1 (forward pass)")
    print(f"  lr              = {args.lr}")
    print(f"  epochs          = {args.epochs}")
    print(f"  w_distill       = {args.w_distill}")
    print(f"  w_anchor        = {args.w_anchor}")

    # --- data ---
    train_ds = LOLDataset(mode="train", layout=args.layout, root=args.dataset_root,
                          crop_size=conf.CROP_SIZE, augment=True)
    val_ds = LOLDataset(mode="test", layout=args.layout, root=args.dataset_root, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=conf.BATCH_SIZE, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # --- models ---
    diff = DiffusionEngine()

    # Student (trainable). Architecture follows --use-illum-prior.
    student = ResidualConditionedUNet(use_illum_prior=use_prior).to(conf.DEVICE)
    if args.init_from:
        smart_load_checkpoint(student, args.init_from, conf.DEVICE,
                              prefer="ema" if not args.init_from_model_key else "model")
    else:
        raise SystemExit("--init-from is REQUIRED for self-distillation. Pass the Day 1 checkpoint path.")

    # Teacher (frozen). Architecture must match the source checkpoint exactly,
    # so it always uses use_illum_prior=False if the source did. Detect from the ckpt.
    raw_ckpt = torch.load(args.init_from, map_location="cpu")
    if isinstance(raw_ckpt, dict):
        teacher_use_prior = bool(raw_ckpt.get("use_illum_prior", False))
    else:
        teacher_use_prior = False
    del raw_ckpt
    teacher = ResidualConditionedUNet(use_illum_prior=teacher_use_prior).to(conf.DEVICE)
    smart_load_checkpoint(teacher, args.init_from, conf.DEVICE,
                          prefer="ema" if not args.init_from_model_key else "model")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"[teacher] use_illum_prior={teacher_use_prior}, frozen, {count_parameters(teacher):,} params")
    print(f"[student] use_illum_prior={use_prior}, trainable, {count_parameters(student):,} params")

    ema = EMA(student)

    optimizer = optim.AdamW(student.parameters(), lr=args.lr, betas=conf.BETAS,
                             weight_decay=conf.WEIGHT_DECAY)
    # Cosine LR (no warmup since we're starting from a trained checkpoint)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=conf.LR_MIN)

    # --- coarse student schedule (5-step DDIM by default) ---
    n_student_steps = args.student_inference_steps
    student_schedule = torch.linspace(conf.TIMESTEPS - 1, 0, n_student_steps,
                                      device=conf.DEVICE).long()
    student_schedule = torch.unique_consecutive(student_schedule)
    print(f"[student] coarse 5-step schedule (timesteps): {student_schedule.tolist()}")

    # --- logging ---
    os.makedirs(conf.SAVE_DIR, exist_ok=True)
    log_path = os.path.join(conf.SAVE_DIR, f"distill_log_{args.tag}.csv")
    log_f = open(log_path, "w", newline="")
    log_writer = csv.writer(log_f)
    log_writer.writerow([
        "epoch", "lr", "loss_total", "loss_distill", "loss_anchor",
        "val_psnr_5step", "val_ssim_5step", "sec",
    ])

    best_psnr = -1.0

    # --- training loop ---
    for epoch in range(args.epochs):
        student.train()
        t0 = time.time()
        running = {"total": 0.0, "distill": 0.0, "anchor": 0.0}
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Distill epoch {epoch}")
        for low, high in pbar:
            low = low.to(conf.DEVICE, non_blocking=True)
            high = high.to(conf.DEVICE, non_blocking=True)
            target_residual = torch.clamp(high - low, -1.0, 1.0)
            B = low.size(0)

            # Sample a SINGLE t for the whole batch from the coarse student schedule.
            # This makes the teacher refinement efficient (one schedule per batch).
            t_idx_scalar = int(torch.randint(0, len(student_schedule), (1,)).item())
            t_actual = int(student_schedule[t_idx_scalar].item())
            t_batch = torch.full((B,), t_actual, device=conf.DEVICE, dtype=torch.long)

            # Forward diffuse the GT residual to noise level t_actual.
            noisy_residual, _ = diff.q_sample(target_residual, t_batch,
                                               offset_noise_strength=0.1)

            # Teacher refines (frozen, no_grad). Always conditions on `low` only
            # (uses illum_prior internally if it was trained with it).
            teacher_x0 = teacher_refine(
                teacher, diff, noisy_residual, low,
                t_start=t_actual, num_substeps=args.teacher_steps,
            )

            # Student single forward pass. Same x_t, same t, same low.
            _pred_img, student_x0 = student(noisy_residual, t_batch, low,
                                              return_residual=True)
            student_x0 = torch.clamp(student_x0, -1.0, 1.0)

            # Losses
            loss_distill = charbonnier(student_x0, teacher_x0)
            loss_anchor  = charbonnier(student_x0, target_residual)
            loss = args.w_distill * loss_distill + args.w_anchor * loss_anchor

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), conf.GRAD_CLIP)
            optimizer.step()
            ema.update(student)

            running["total"]   += float(loss.item())
            running["distill"] += float(loss_distill.item())
            running["anchor"]  += float(loss_anchor.item())
            n_batches += 1

            pbar.set_postfix({
                "t":       f"{t_actual:>3d}",
                "total":   f"{loss.item():.3f}",
                "distill": f"{loss_distill.item():.3f}",
                "anchor":  f"{loss_anchor.item():.3f}",
            })

        scheduler.step()
        avg = {k: v / max(n_batches, 1) for k, v in running.items()}

        # --- 5-step validation using EMA weights ---
        val_psnr, val_ssim = -1.0, -1.0
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            ema_model = ResidualConditionedUNet(use_illum_prior=use_prior).to(conf.DEVICE)
            ema_model.load_state_dict(ema.shadow)
            # Validate at the student inference steps (default 5)
            old_inf = conf.INFERENCE_STEPS
            conf.INFERENCE_STEPS = n_student_steps
            try:
                val_psnr, val_ssim = validate(ema_model, diff, val_loader, conf.DEVICE,
                                               max_batches=args.val_max_batches)
            finally:
                conf.INFERENCE_STEPS = old_inf
            print(f"[epoch {epoch}] 5-step val PSNR {val_psnr:.3f}  SSIM {val_ssim:.4f}")

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save({
                    "epoch": epoch,
                    "model": student.state_dict(),
                    "ema":   ema.shadow,
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim,
                    "use_illum_prior": use_prior,
                    "init_from": args.init_from,
                    "teacher_steps": args.teacher_steps,
                    "student_inference_steps": n_student_steps,
                }, os.path.join(conf.SAVE_DIR, f"best_{args.tag}.pth"))
                print(f"  -> new best, saved best_{args.tag}.pth")

        # Save last (overwrite) every epoch
        torch.save({
            "epoch": epoch,
            "model": student.state_dict(),
            "ema":   ema.shadow,
            "use_illum_prior": use_prior,
            "init_from": args.init_from,
            "teacher_steps": args.teacher_steps,
            "student_inference_steps": n_student_steps,
        }, os.path.join(conf.SAVE_DIR, f"last_{args.tag}.pth"))

        log_writer.writerow([
            epoch, optimizer.param_groups[0]["lr"],
            avg["total"], avg["distill"], avg["anchor"],
            val_psnr, val_ssim, time.time() - t0,
        ])
        log_f.flush()

    log_f.close()
    print(f"Done. Best 5-step val PSNR: {best_psnr:.3f}")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--layout", default=os.environ.get("DATASET_LAYOUT", "lolv2_real"),
                   help="Dataset layout: lol_v1 | lolv2_real | lolv2_syn | flat")
    p.add_argument("--dataset-root", default=None,
                   help="Path to the layout's parent directory (e.g. /kaggle/working/data).")
    p.add_argument("--init-from", required=True,
                   help="Path to the Day 1 checkpoint. Used for BOTH teacher and student init.")
    p.add_argument("--init-from-model-key", action="store_true",
                   help="Use 'model' key instead of 'ema' key from the source checkpoint.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5,
                   help="Student learning rate. 1e-5 is conservative; 5e-5 may converge faster.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--use-illum-prior", type=int, default=0, choices=[0, 1],
                   help="Whether the STUDENT uses the illumination prior. Teacher always matches "
                        "the source checkpoint. Default 0 = student matches teacher architecture.")
    p.add_argument("--teacher-steps", type=int, default=5,
                   help="DDIM sub-steps the teacher takes from t_start to 0. Higher = more refined "
                        "target but slower training. K=5 is the sweet spot.")
    p.add_argument("--student-inference-steps", type=int, default=5,
                   help="The student's intended inference step count. Validation uses this. "
                        "Coarse training schedule has this many points.")
    p.add_argument("--w-distill", type=float, default=1.0,
                   help="Weight on charbonnier(student_x0, teacher_x0).")
    p.add_argument("--w-anchor", type=float, default=0.5,
                   help="Weight on charbonnier(student_x0, gt_residual). Stability anchor.")
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--val-max-batches", type=int, default=None)
    p.add_argument("--tag", default=os.environ.get("RUN_TAG", "distill"))
    return p.parse_args()


if __name__ == "__main__":
    train_distill(parse())
