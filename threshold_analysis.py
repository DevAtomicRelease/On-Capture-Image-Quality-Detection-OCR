"""
Threshold Analysis
==================
Generates signal distribution histograms and suggests optimal thresholds
by analysing the separation between usable and retake classes.

This script produces the threshold justification required by the assessment:
- Histogram plots of each signal per class
- Suggested threshold values based on inter-class separation
- Overlap analysis

Usage:
    python threshold_analysis.py --input_dir test_images/ --labels labels.csv --output_dir analysis/

Requires: matplotlib (pip install matplotlib)
"""

import os
import sys
import csv
import json
import argparse
import numpy as np
from collections import defaultdict
from detector import detect

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Histograms will not be generated.")
    print("Install with: pip install matplotlib")


SIGNALS_TO_ANALYZE = [
    "laplacian_variance",
    "tenengrad",
    "fft_hf_ratio",
    "local_sharpness_std",
    "brightness_mean",
    "brightness_std",
    "glare_ratio",
    "overexposed_ratio",
    "glare_cluster_ratio",
    "border_emptiness",
    "content_coverage",
    "dark_pixel_ratio",
]


def load_labels(path: str) -> dict:
    """Load labels CSV into {filename: {ground_truth, failure_mode}} map."""
    labels = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"].strip()
            labels[fname] = {
                "ground_truth": row["ground_truth"].strip().lower(),
                "failure_mode": row.get("failure_mode", "").strip().lower(),
            }
    return labels


def collect_signals(input_dir: str, labels: dict) -> dict:
    """Run detector on all labelled images and collect signals by class."""
    class_signals = defaultdict(lambda: defaultdict(list))
    all_results = []

    for fname, label_info in sorted(labels.items()):
        img_path = os.path.join(input_dir, fname)
        if not os.path.exists(img_path):
            print(f"  Skipping {fname} — file not found")
            continue

        result = detect(img_path)
        gt = label_info["ground_truth"]

        for sig_name in SIGNALS_TO_ANALYZE:
            if sig_name in result["signals"]:
                class_signals[gt][sig_name].append(result["signals"][sig_name])

        all_results.append({
            "filename": fname,
            "ground_truth": gt,
            "failure_mode": label_info["failure_mode"],
            **result["signals"],
        })
        print(f"  Processed {fname} (gt={gt})")

    return class_signals, all_results


def find_optimal_threshold(usable_vals: list, retake_vals: list) -> dict:
    """
    Find the threshold that minimises weighted classification error.
    Weights false-rejects 2x more than false-accepts (per assessment priority).
    """
    if not usable_vals or not retake_vals:
        return {"threshold": None, "error": None}

    all_vals = sorted(set(usable_vals + retake_vals))
    best_threshold = None
    best_score = float("inf")
    false_reject_weight = 2.0  # false-rejects are worse for UX

    for i in range(len(all_vals) - 1):
        t = (all_vals[i] + all_vals[i + 1]) / 2

        # Convention: below threshold → retake, above → usable
        false_rejects = sum(1 for v in usable_vals if v < t)
        false_accepts = sum(1 for v in retake_vals if v >= t)

        fr_rate = false_rejects / len(usable_vals)
        fa_rate = false_accepts / len(retake_vals)
        score = false_reject_weight * fr_rate + fa_rate

        if score < best_score:
            best_score = score
            best_threshold = t

    return {
        "threshold": round(best_threshold, 4) if best_threshold else None,
        "weighted_error": round(best_score, 4) if best_score != float("inf") else None,
    }


def plot_histogram(
    usable_vals: list,
    retake_vals: list,
    signal_name: str,
    threshold: float,
    output_path: str,
):
    """Plot overlapping histograms for usable vs retake classes."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    bins = 30
    if usable_vals:
        ax.hist(usable_vals, bins=bins, alpha=0.6, label="usable",
                color="#2196F3", edgecolor="black", linewidth=0.5)
    if retake_vals:
        ax.hist(retake_vals, bins=bins, alpha=0.6, label="retake",
                color="#F44336", edgecolor="black", linewidth=0.5)

    if threshold is not None:
        ax.axvline(x=threshold, color="black", linestyle="--", linewidth=2,
                   label=f"threshold = {threshold:.4f}")

    ax.set_xlabel(signal_name, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Distribution of '{signal_name}' by class", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze signal distributions and derive thresholds."
    )
    parser.add_argument("--input_dir", "-i", required=True,
                        help="Path to test images folder.")
    parser.add_argument("--labels", "-l", required=True,
                        help="Path to labels CSV.")
    parser.add_argument("--output_dir", "-o", default="analysis",
                        help="Directory for output plots and analysis.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading labels...")
    labels = load_labels(args.labels)
    print(f"Found {len(labels)} labelled images.\n")

    print("Running detector on all images...")
    class_signals, all_results = collect_signals(args.input_dir, labels)

    print("\n--- Threshold Analysis ---\n")

    threshold_report = {}
    for sig_name in SIGNALS_TO_ANALYZE:
        usable_vals = class_signals.get("usable", {}).get(sig_name, [])
        retake_vals = class_signals.get("retake", {}).get(sig_name, [])

        if not usable_vals and not retake_vals:
            continue

        result = find_optimal_threshold(usable_vals, retake_vals)
        threshold_report[sig_name] = {
            "usable_count": len(usable_vals),
            "retake_count": len(retake_vals),
            "usable_mean": round(np.mean(usable_vals), 4) if usable_vals else None,
            "usable_std": round(np.std(usable_vals), 4) if usable_vals else None,
            "retake_mean": round(np.mean(retake_vals), 4) if retake_vals else None,
            "retake_std": round(np.std(retake_vals), 4) if retake_vals else None,
            "optimal_threshold": result["threshold"],
            "weighted_error": result["weighted_error"],
        }

        print(f"{sig_name}:")
        print(f"  usable: mean={threshold_report[sig_name]['usable_mean']}, "
              f"std={threshold_report[sig_name]['usable_std']}")
        print(f"  retake: mean={threshold_report[sig_name]['retake_mean']}, "
              f"std={threshold_report[sig_name]['retake_std']}")
        print(f"  optimal threshold: {result['threshold']}")
        print()

        # Plot
        plot_path = os.path.join(args.output_dir, f"hist_{sig_name}.png")
        plot_histogram(usable_vals, retake_vals, sig_name,
                       result["threshold"], plot_path)

    # Save threshold report
    report_path = os.path.join(args.output_dir, "threshold_report.json")
    with open(report_path, "w") as f:
        json.dump(threshold_report, f, indent=2)
    print(f"Threshold report saved to {report_path}")

    # Save raw signal data
    data_path = os.path.join(args.output_dir, "signal_data.csv")
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(data_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"Raw signal data saved to {data_path}")

    if HAS_MATPLOTLIB:
        print(f"Histograms saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
