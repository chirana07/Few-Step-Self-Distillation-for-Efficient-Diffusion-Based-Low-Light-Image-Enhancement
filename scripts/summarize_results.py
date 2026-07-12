#!/usr/bin/env python3
"""Summarize controlled test runs and decompose continued-training/FSD/ARR effects."""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = ("psnr", "ssim", "lpips")


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for metric in METRICS:
            row[metric] = float(row[metric])
        row["training_seed"] = int(row["training_seed"]) if row["training_seed"] else None
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = read_rows(args.registry)
    source = next(
        row for row in rows
        if row["method"] in {"source_teacher", "transferred_teacher"}
    )
    by_method = defaultdict(list)
    by_seed = defaultdict(dict)
    for row in rows:
        by_method[row["method"]].append(row)
        if row["training_seed"] is not None:
            by_seed[row["training_seed"]][row["method"]] = row

    summary = []
    for method, method_rows in sorted(by_method.items()):
        item = {"method": method, "n_training_seeds": len(method_rows)}
        for metric in METRICS:
            values = [row[metric] for row in method_rows]
            item[f"{metric}_mean"] = statistics.mean(values)
            item[f"{metric}_std_across_training_seeds"] = statistics.stdev(values) if len(values) > 1 else ""
        summary.append(item)

    effects = []
    for seed, methods in sorted(by_seed.items()):
        required = {"anchor_only", "fsd", "fsd_arr"}
        if not required.issubset(methods):
            raise RuntimeError(f"Seed {seed} is missing methods: {sorted(required - set(methods))}")
        item = {"training_seed": seed}
        for metric in METRICS:
            item[f"continued_training_{metric}"] = methods["anchor_only"][metric] - source[metric]
            item[f"fsd_over_anchor_{metric}"] = methods["fsd"][metric] - methods["anchor_only"][metric]
            item[f"arr_over_fsd_{metric}"] = methods["fsd_arr"][metric] - methods["fsd"][metric]
            item[f"total_over_source_{metric}"] = methods["fsd_arr"][metric] - source[metric]
        effects.append(item)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "method_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (args.output_dir / "contribution_decomposition.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(effects[0]))
        writer.writeheader()
        writer.writerows(effects)
    print(args.output_dir / "method_summary.csv")
    print(args.output_dir / "contribution_decomposition.csv")


if __name__ == "__main__":
    main()
