import os
import subprocess
import sys
import csv

def run_cmd(cmd):
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)

def read_summary(path):
    if not os.path.exists(path): return []
    return list(csv.DictReader(open(path)))

def fmt(r, variant):
    lp = r.get('lpips_mean')
    lp_s = f"{float(lp):.4f}" if lp not in (None, '', 'None') else '-'
    return (f"| {variant} | {r['split']} | {r['n']} | {r['inference_steps']} | "
            f"{float(r['psnr_mean']):.3f} | {float(r['ssim_mean']):.4f} | {lp_s} | "
            f"{float(r['runtime_mean']):.3f} |")

def main():
    CHECKPOINT = "./checkpoints/last_pth_only/final.pth"
    EVAL_RESULTS = "./eval_results/empirical_study"
    os.makedirs(EVAL_RESULTS, exist_ok=True)
    
    # We will only use eval15 as the local dataset since LOL-v2 is not present locally.
    SPLITS = [
        "eval15:./eval15/low:./eval15/high"
    ]
    
    print("=== 1. ARR Ablation (Alpha Grid) ===")
    ARR_GRID = []
    
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        tag = f"arr_a{int(alpha*100):03d}"
        run_cmd([
            sys.executable, "evaluation.py",
            "--splits", *SPLITS,
            "--inference-steps", "5",
            "--sampler", "ddim",
            "--checkpoint", CHECKPOINT,
            "--results-root", os.path.join(EVAL_RESULTS, "arr_grid"),
            "--tag", tag,
            "--gate-alpha", str(alpha),
            "--gate-floor", "0.5",
            "--no-lpips",
        ])
        summary_path = os.path.join(EVAL_RESULTS, "arr_grid_" + tag, "summary.csv")
        for row in read_summary(summary_path):
            row["alpha"] = alpha
            ARR_GRID.append(row)
            
    print("=== 2. Sampler Ablation (DPM-Posterior) ===")
    run_cmd([
        sys.executable, "evaluation.py",
        "--splits", *SPLITS,
        "--inference-steps", "5",
        "--sampler", "dpm_posterior",
        "--checkpoint", CHECKPOINT,
        "--results-root", os.path.join(EVAL_RESULTS, "sampler_ablation"),
        "--tag", "dpm",
        "--no-lpips",
    ])
    
    with open("paper_tables.md", "w") as f:
        f.write("\n\n==== PAPER TABLES ====\n\n")
        
        # Table A: Headline (already in phase3_day1_outputs)
        f.write("### Table A — Headline (Day 1 Vanilla DDIM)\n")
        f.write("| Variant | Split | n | Steps | PSNR | SSIM | LPIPS | Latency/img (s) |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for tag in ("s5", "s20"):
            for r in read_summary(f"./phase3_day1_outputs/headline_{tag}/summary.csv"):
                if r['split'] == 'eval15': # Only print eval15 for consistency if we want, or both.
                    f.write(fmt(r, "Vanilla DDIM") + "\n")
            for r in read_summary(f"./phase3_day1_outputs/headline_{tag}/summary.csv"):
                if r['split'] == 'lolv2_real': # It's good to show the existing LOL-v2 results!
                    f.write(fmt(r, "Vanilla DDIM") + "\n")
                
        f.write("\n### Table B — Method ablation (Adaptive Residual Rescaling @ 5 DDIM steps, eval15)\n")
        f.write("ARR introduces an inference-time dynamic scaling of the residual.\n")
        f.write("| alpha | PSNR | SSIM |\n")
        f.write("|---|---|---|\n")
        for r in ARR_GRID:
            f.write(f"| {float(r['alpha']):.2f} | {float(r['psnr_mean']):.3f} | {float(r['ssim_mean']):.4f} |\n")
            
        f.write("\n### Table C — Sampler Comparison (@ 5 steps, eval15)\n")
        f.write("| Variant | Split | n | Steps | PSNR | SSIM | LPIPS |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in read_summary("./phase3_day1_outputs/headline_s5/summary.csv"):
            if r['split'] == 'eval15':
                lp = r.get('lpips_mean', '-')
                f.write(f"| DDIM | {r['split']} | {r['n']} | {r['inference_steps']} | "
                      f"{float(r['psnr_mean']):.3f} | {float(r['ssim_mean']):.4f} | {float(lp) if lp not in ('', None, '-') else '-'} |\n")
        for r in read_summary(os.path.join(EVAL_RESULTS, "sampler_ablation_dpm/summary.csv")):
            lp = r.get('lpips_mean', '-')
            f.write(f"| DPM-Posterior | {r['split']} | {r['n']} | {r['inference_steps']} | "
                  f"{float(r['psnr_mean']):.3f} | {float(r['ssim_mean']):.4f} | {float(lp) if lp not in ('', None, '-') else '-'} |\n")

        f.write("\n### Table D — Step ablation (Day 1 Vanilla DDIM)\n")
        f.write("| Steps | Split | PSNR | SSIM | Latency/img (s) |\n")
        f.write("|---|---|---|---|---|\n")
        for N in (5, 10, 20, 50, 100):
            for r in read_summary(f"./phase3_day1_outputs/step_ablation_full_ddim_s{N}/summary.csv"):
                f.write(f"| {N} | {r['split']} | {float(r['psnr_mean']):.3f} | {float(r['ssim_mean']):.4f} | {float(r['runtime_mean']):.3f} |\n")

        f.write("\n### Table E — Efficiency (T4, 400x600, Day 1 Vanilla DDIM)\n")
        try:
            with open("./phase3_day1_outputs/efficiency_t4.csv") as eff:
                f.write(eff.read() + "\n")
        except Exception as e:
            f.write(f"Could not read efficiency: {e}\n")
    
    # Table A: Headline (already in phase3_day1_outputs)
    print("### Table A — Headline (Day 1 Vanilla DDIM)")
    print("| Variant | Split | n | Steps | PSNR | SSIM | LPIPS | Latency/img (s) |")
    print("|---|---|---|---|---|---|---|---|")
    for tag in ("s5", "s20"):
        for r in read_summary(f"./phase3_day1_outputs/headline_{tag}/summary.csv"):
            if r['split'] == 'eval15': # Only print eval15 for consistency if we want, or both.
                print(fmt(r, "Vanilla DDIM"))
        for r in read_summary(f"./phase3_day1_outputs/headline_{tag}/summary.csv"):
            if r['split'] == 'lolv2_real': # It's good to show the existing LOL-v2 results!
                print(fmt(r, "Vanilla DDIM"))
            
    print("\n### Table B — Method ablation (Adaptive Residual Rescaling @ 5 DDIM steps, eval15)")
    print("ARR introduces an inference-time dynamic scaling of the residual.")
    print("| alpha | PSNR | SSIM |")
    print("|---|---|---|")
    for r in ARR_GRID:
        print(f"| {float(r['alpha']):.2f} | {float(r['psnr_mean']):.3f} | {float(r['ssim_mean']):.4f} |")
        
    print("\n### Table C — Sampler Comparison (@ 5 steps, eval15)")
    print("| Variant | Split | n | Steps | PSNR | SSIM | LPIPS |")
    print("|---|---|---|---|---|---|---|")
    for r in read_summary("./phase3_day1_outputs/headline_s5/summary.csv"):
        if r['split'] == 'eval15':
            lp = r.get('lpips_mean', '-')
            print(f"| DDIM | {r['split']} | {r['n']} | {r['inference_steps']} | "
                  f"{float(r['psnr_mean']):.3f} | {float(r['ssim_mean']):.4f} | {float(lp) if lp not in ('', None, '-') else '-'} |")
    for r in read_summary(os.path.join(EVAL_RESULTS, "sampler_ablation_dpm/summary.csv")):
        lp = r.get('lpips_mean', '-')
        print(f"| DPM-Posterior | {r['split']} | {r['n']} | {r['inference_steps']} | "
              f"{float(r['psnr_mean']):.3f} | {float(r['ssim_mean']):.4f} | {float(lp) if lp not in ('', None, '-') else '-'} |")

    print("\n### Table D — Step ablation (Day 1 Vanilla DDIM)")
    print("| Steps | Split | PSNR | SSIM | Latency/img (s) |")
    print("|---|---|---|---|---|")
    for N in (5, 10, 20, 50, 100):
        for r in read_summary(f"./phase3_day1_outputs/step_ablation_full_ddim_s{N}/summary.csv"):
            print(f"| {N} | {r['split']} | {float(r['psnr_mean']):.3f} | {float(r['ssim_mean']):.4f} | {float(r['runtime_mean']):.3f} |")

    print("\n### Table E — Efficiency (T4, 400x600, Day 1 Vanilla DDIM)")
    try:
        with open("./phase3_day1_outputs/efficiency_t4.csv") as f:
            print(f.read())
    except Exception as e:
        print("Could not read efficiency:", e)

if __name__ == "__main__":
    main()
