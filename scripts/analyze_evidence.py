#!/usr/bin/env python3
"""Paired analysis of official-test evidence stored in integrity_evidence.zip."""

import argparse
import csv
import io
import zipfile

import numpy as np


METHODS = ("source_teacher", "anchor_only_seed3407", "fsd_seed3407")
METRICS = ("psnr", "ssim", "lpips")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    args = parser.parse_args()

    data = {}
    with zipfile.ZipFile(args.archive) as archive:
        for method in METHODS:
            path = f"results/official_test/{method}/per_image.csv"
            with archive.open(path) as raw:
                rows = list(csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8")))
            data[method] = {
                row["image"]: {metric: float(row[metric]) for metric in METRICS}
                for row in rows
            }

    comparisons = (
        ("source_teacher", "fsd_seed3407"),
        ("anchor_only_seed3407", "fsd_seed3407"),
        ("source_teacher", "anchor_only_seed3407"),
    )
    rng = np.random.default_rng(3407)
    for baseline, candidate in comparisons:
        common = sorted(set(data[baseline]) & set(data[candidate]))
        print(f"\n{baseline} -> {candidate} (n={len(common)})")
        for metric in METRICS:
            delta = np.array([
                data[candidate][image][metric] - data[baseline][image][metric]
                for image in common
            ])
            samples = rng.choice(
                delta, size=(args.bootstrap_samples, len(delta)), replace=True
            ).mean(axis=1)
            low, high = np.quantile(samples, [0.025, 0.975])
            improved = int((delta < 0).sum()) if metric == "lpips" else int((delta > 0).sum())
            print(
                f"{metric}: mean_delta={delta.mean():.6f}, median={np.median(delta):.6f}, "
                f"improved={improved}/{len(delta)}, bootstrap_95%=[{low:.6f}, {high:.6f}]"
            )


if __name__ == "__main__":
    main()
