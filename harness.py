"""
Test Harness
============
Runs the detector over a folder of labelled images and produces:
    - Per-image verdicts (CSV output)
    - Confusion matrix
    - False-reject and false-accept rates
    - Per-failure-mode performance breakdown
    - Latency percentiles (p50, p95, max)
    - Signal distribution histograms for threshold derivation

Usage:
    python harness.py --images test_images/ --labels labels.csv
    python harness.py --images test_images/ --labels labels.csv --histograms
    python harness.py --images test_images/ --labels labels.csv --output results.csv
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from detector import detect, Thresholds


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

def load_labels(labels_path: str) -> Dict[str, dict]:
    """
    Load ground-truth labels from CSV.

    Expected CSV columns:
        filename        - image filename (must match files in the images folder)
        ground_truth    - "usable" or "retake" (or "borderline")
        failure_mode    - optional: "motion_blur", "defocus", "glare",
                          "framing", "lighting", "none", or blank

    Returns dict: filename -> {"ground_truth": str, "failure_mode": str}
    """
    labels = {}
    with open(labels_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"].strip()
            gt = row["ground_truth"].strip().lower()
            fm = row.get("failure_mode", "none").strip().lower()
            if fm == "":
                fm = "none"
            labels[fname] = {"ground_truth": gt, "failure_mode": fm}
    return labels


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batch(
    images_dir: str,
    labels: Dict[str, dict],
    thresholds: Thresholds = None,
) -> List[dict]:
    """
    Run detector on all labelled images.
    Returns list of result dicts with ground truth attached.
    """
    results = []
    filenames = sorted(labels.keys())

    for i, fname in enumerate(filenames):
        path = os.path.join(images_dir, fname)
        if not os.path.isfile(path):
            print(f"  WARNING: {fname} not found in {images_dir}, skipping")
            continue

        result = detect(path, thresholds)
        result["filename"] = fname
        result["ground_truth"] = labels[fname]["ground_truth"]
        result["failure_mode"] = labels[fname]["failure_mode"]
        results.append(result)

        status = "OK" if result["verdict"] == result["ground_truth"] else "MISMATCH"
        print(
            f"  [{i+1}/{len(filenames)}] {fname}: "
            f"verdict={result['verdict']}  gt={result['ground_truth']}  "
            f"latency={result['latency_ms']}ms  [{status}]"
        )

    return results


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_confusion_matrix(results: List[dict]) -> dict:
    """
    Compute confusion matrix mapping predicted verdict to ground truth.

    For the purpose of false-reject/false-accept calculation, we treat
    "borderline" predictions as "retake" (conservative) and "borderline"
    ground truths as "retake" (they are unusable).
    """
    # Map to binary for rate calculation
    def to_binary(v):
        return "usable" if v == "usable" else "retake"

    matrix = defaultdict(lambda: defaultdict(int))
    for r in results:
        pred = r["verdict"]
        gt = r["ground_truth"]
        matrix[gt][pred] += 1

    # Binary confusion matrix
    tp = 0  # correctly accepted usable
    tn = 0  # correctly rejected retake
    fp = 0  # false accept (retake passed as usable)
    fn = 0  # false reject (usable flagged for retake)

    for r in results:
        pred_bin = to_binary(r["verdict"])
        gt_bin = to_binary(r["ground_truth"])

        if gt_bin == "usable" and pred_bin == "usable":
            tp += 1
        elif gt_bin == "retake" and pred_bin == "retake":
            tn += 1
        elif gt_bin == "retake" and pred_bin == "usable":
            fp += 1
        elif gt_bin == "usable" and pred_bin == "retake":
            fn += 1

    total_usable = tp + fn
    total_retake = tn + fp

    false_reject_rate = fn / total_usable if total_usable > 0 else 0.0
    false_accept_rate = fp / total_retake if total_retake > 0 else 0.0
    accuracy = (tp + tn) / len(results) if results else 0.0

    return {
        "full_matrix": dict(matrix),
        "binary": {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive_accept": fp,
            "false_negative_reject": fn,
        },
        "false_reject_rate": round(false_reject_rate, 4),
        "false_accept_rate": round(false_accept_rate, 4),
        "accuracy": round(accuracy, 4),
        "total_images": len(results),
    }


def compute_per_failure_mode(results: List[dict]) -> dict:
    """
    Break down performance by failure mode.
    For each mode, report total images, correct rejections, and miss rate.
    """
    modes = defaultdict(lambda: {"total": 0, "correctly_handled": 0, "missed": 0})

    for r in results:
        fm = r["failure_mode"]
        if fm == "none":
            continue  # skip usable images; they don't have a failure mode

        modes[fm]["total"] += 1
        # A retake image is "correctly handled" if verdict != usable
        if r["verdict"] != "usable":
            modes[fm]["correctly_handled"] += 1
        else:
            modes[fm]["missed"] += 1

    # Compute detection rate per mode
    summary = {}
    for mode, counts in sorted(modes.items()):
        detection_rate = (
            counts["correctly_handled"] / counts["total"]
            if counts["total"] > 0
            else 0.0
        )
        summary[mode] = {
            "total": counts["total"],
            "detected": counts["correctly_handled"],
            "missed": counts["missed"],
            "detection_rate": round(detection_rate, 4),
        }

    return summary


def compute_latency_stats(results: List[dict]) -> dict:
    """Compute p50, p95, and max latency in ms."""
    latencies = [r["latency_ms"] for r in results]
    if not latencies:
        return {"p50": 0, "p95": 0, "max": 0, "mean": 0}

    return {
        "p50": round(float(np.percentile(latencies, 50)), 2),
        "p95": round(float(np.percentile(latencies, 95)), 2),
        "max": round(float(max(latencies)), 2),
        "mean": round(float(np.mean(latencies)), 2),
    }


def find_misclassified(results: List[dict]) -> List[dict]:
    """Return all misclassified images for failure analysis."""
    def to_binary(v):
        return "usable" if v == "usable" else "retake"

    misclassified = []
    for r in results:
        if to_binary(r["verdict"]) != to_binary(r["ground_truth"]):
            misclassified.append({
                "filename": r["filename"],
                "predicted": r["verdict"],
                "ground_truth": r["ground_truth"],
                "reason": r["reason"],
                "failure_mode": r["failure_mode"],
                "signals": r["signals"],
            })
    return misclassified


# ---------------------------------------------------------------------------
# Signal distribution (for threshold derivation)
# ---------------------------------------------------------------------------

def print_signal_distributions(results: List[dict]):
    """
    Print signal distributions per class for threshold derivation.
    Shows min, 25th, median, 75th, max for each signal, split by class.
    """
    signal_names = [
        "laplacian_var", "laplacian_p95", "tenengrad", "fft_hf_ratio",
        "glare_ratio", "mean_brightness", "contrast", "border_content_ratio",
    ]

    def to_binary(v):
        return "usable" if v == "usable" else "retake"

    usable = [r for r in results if to_binary(r["ground_truth"]) == "usable"]
    retake = [r for r in results if to_binary(r["ground_truth"]) == "retake"]

    print("\n" + "=" * 80)
    print("SIGNAL DISTRIBUTIONS (for threshold derivation)")
    print("=" * 80)

    for sig in signal_names:
        u_vals = [r["signals"].get(sig, 0) for r in usable]
        r_vals = [r["signals"].get(sig, 0) for r in retake]

        print(f"\n--- {sig} ---")
        if u_vals:
            print(
                f"  USABLE  (n={len(u_vals):3d}): "
                f"min={min(u_vals):10.4f}  p25={np.percentile(u_vals, 25):10.4f}  "
                f"median={np.median(u_vals):10.4f}  p75={np.percentile(u_vals, 75):10.4f}  "
                f"max={max(u_vals):10.4f}"
            )
        if r_vals:
            print(
                f"  RETAKE  (n={len(r_vals):3d}): "
                f"min={min(r_vals):10.4f}  p25={np.percentile(r_vals, 25):10.4f}  "
                f"median={np.median(r_vals):10.4f}  p75={np.percentile(r_vals, 75):10.4f}  "
                f"max={max(r_vals):10.4f}"
            )


def generate_histogram_data(results: List[dict], output_dir: str):
    """
    Save signal distributions as CSV files for plotting histograms externally.
    Each file has two columns: value, class (usable/retake).
    """
    os.makedirs(output_dir, exist_ok=True)

    signal_names = [
        "laplacian_var", "laplacian_p95", "tenengrad", "fft_hf_ratio",
        "glare_ratio", "mean_brightness", "contrast", "border_content_ratio",
    ]

    def to_binary(v):
        return "usable" if v == "usable" else "retake"

    for sig in signal_names:
        filepath = os.path.join(output_dir, f"dist_{sig}.csv")
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["value", "class"])
            for r in results:
                writer.writerow([
                    r["signals"].get(sig, 0),
                    to_binary(r["ground_truth"]),
                ])
        print(f"  Saved: {filepath}")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results_csv(results: List[dict], output_path: str):
    """Save per-image results to CSV."""
    if not results:
        return

    signal_keys = sorted(results[0].get("signals", {}).keys())
    fieldnames = [
        "filename", "verdict", "ground_truth", "failure_mode",
        "reason", "latency_ms",
    ] + signal_keys

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {
                "filename": r["filename"],
                "verdict": r["verdict"],
                "ground_truth": r["ground_truth"],
                "failure_mode": r["failure_mode"],
                "reason": r["reason"],
                "latency_ms": r["latency_ms"],
            }
            row.update(r.get("signals", {}))
            writer.writerow(row)


def print_report(
    confusion: dict,
    per_mode: dict,
    latency: dict,
    misclassified: List[dict],
):
    """Print a formatted metrics report to stdout."""
    print("\n" + "=" * 80)
    print("EVALUATION REPORT")
    print("=" * 80)

    print(f"\nTotal images evaluated: {confusion['total_images']}")

    # Confusion matrix
    print("\n--- Confusion Matrix (full) ---")
    matrix = confusion["full_matrix"]
    all_verdicts = ["usable", "borderline", "retake"]
    gt_pred_label = "GT \\ Pred"
    header = f"{gt_pred_label:<15}" + "".join(f"{v:<12}" for v in all_verdicts)
    print(header)
    for gt in all_verdicts:
        row_data = matrix.get(gt, {})
        row = f"{gt:<15}" + "".join(f"{row_data.get(v, 0):<12}" for v in all_verdicts)
        print(row)

    # Binary metrics
    b = confusion["binary"]
    print("\n--- Binary Metrics (borderline treated as retake) ---")
    print(f"  True Positives  (usable correctly accepted):  {b['true_positive']}")
    print(f"  True Negatives  (retake correctly rejected):  {b['true_negative']}")
    print(f"  False Accepts   (retake passed as usable):    {b['false_positive_accept']}")
    print(f"  False Rejects   (usable flagged for retake):  {b['false_negative_reject']}")
    print(f"\n  False-Reject Rate: {confusion['false_reject_rate']:.2%}")
    print(f"  False-Accept Rate: {confusion['false_accept_rate']:.2%}")
    print(f"  Overall Accuracy:  {confusion['accuracy']:.2%}")

    # Per-failure-mode
    print("\n--- Per-Failure-Mode Performance ---")
    if per_mode:
        print(f"  {'Mode':<20} {'Total':<8} {'Detected':<10} {'Missed':<8} {'Det. Rate':<10}")
        for mode, stats in per_mode.items():
            print(
                f"  {mode:<20} {stats['total']:<8} {stats['detected']:<10} "
                f"{stats['missed']:<8} {stats['detection_rate']:.2%}"
            )
    else:
        print("  No failure modes labelled in dataset.")

    # Latency
    print("\n--- Latency (ms) ---")
    print(f"  p50:  {latency['p50']} ms")
    print(f"  p95:  {latency['p95']} ms")
    print(f"  max:  {latency['max']} ms")
    print(f"  mean: {latency['mean']} ms")

    # Misclassified images
    print(f"\n--- Misclassified Images ({len(misclassified)} total) ---")
    for m in misclassified[:10]:  # show first 10
        print(f"\n  File: {m['filename']}")
        print(f"    Predicted:    {m['predicted']}")
        print(f"    Ground Truth: {m['ground_truth']}")
        print(f"    Failure Mode: {m['failure_mode']}")
        print(f"    Reason:       {m['reason']}")
        print(f"    Key signals:")
        for k, v in m["signals"].items():
            print(f"      {k}: {v}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run image quality detector over labelled test set and report metrics."
    )
    parser.add_argument(
        "--images", required=True,
        help="Directory containing test images",
    )
    parser.add_argument(
        "--labels", required=True,
        help="CSV file with columns: filename, ground_truth, failure_mode",
    )
    parser.add_argument(
        "--output", default="results.csv",
        help="Path to save per-image results CSV (default: results.csv)",
    )
    parser.add_argument(
        "--histograms", action="store_true",
        help="Generate signal distribution CSVs for histogram plotting",
    )
    parser.add_argument(
        "--hist-dir", default="histograms",
        help="Directory for histogram CSV files (default: histograms/)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isdir(args.images):
        print(f"ERROR: Images directory not found: {args.images}")
        sys.exit(1)
    if not os.path.isfile(args.labels):
        print(f"ERROR: Labels file not found: {args.labels}")
        sys.exit(1)

    # Load labels
    print(f"Loading labels from {args.labels}...")
    labels = load_labels(args.labels)
    print(f"  Loaded {len(labels)} labels")

    # Run batch
    print(f"\nRunning detector on {args.images}...")
    results = run_batch(args.images, labels)
    print(f"\n  Processed {len(results)} images")

    if not results:
        print("ERROR: No images were processed.")
        sys.exit(1)

    # Save per-image results
    save_results_csv(results, args.output)
    print(f"\nPer-image results saved to: {args.output}")

    # Compute and print metrics
    confusion = compute_confusion_matrix(results)
    per_mode = compute_per_failure_mode(results)
    latency = compute_latency_stats(results)
    misclassified = find_misclassified(results)

    print_report(confusion, per_mode, latency, misclassified)

    # Signal distributions for threshold derivation
    print_signal_distributions(results)

    # Histogram data
    if args.histograms:
        print(f"\nGenerating histogram CSVs in {args.hist_dir}/...")
        generate_histogram_data(results, args.hist_dir)

    # Save summary as JSON
    summary_path = args.output.replace(".csv", "_summary.json")
    summary = {
        "confusion_matrix": confusion,
        "per_failure_mode": per_mode,
        "latency": latency,
        "misclassified_count": len(misclassified),
        "misclassified": misclassified[:10],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
