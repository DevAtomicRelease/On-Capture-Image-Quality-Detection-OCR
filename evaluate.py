"""
Evaluation Script
=================
Computes all required metrics by comparing detector results against
ground-truth labels.

Produces:
    - Confusion matrix
    - False-reject rate (usable images flagged for retake)
    - False-accept rate (retake images let through)
    - Per-failure-mode breakdown
    - Latency percentiles

Usage:
    python evaluate.py --results results.csv --labels labels.csv

Labels CSV format:
    filename, ground_truth, failure_mode
    img001.jpg, usable,
    img002.jpg, retake, motion_blur
    img003.jpg, retake, glare
"""

import os
import sys
import csv
import json
import argparse
from collections import defaultdict


def load_csv(path: str) -> list:
    """Load a CSV file as a list of dicts."""
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def merge_results_labels(results: list, labels: list) -> list:
    """Merge results with labels on filename."""
    label_map = {}
    for row in labels:
        fname = row.get("filename", "").strip()
        gt = row.get("ground_truth", "").strip().lower()
        mode = row.get("failure_mode", "").strip().lower()
        label_map[fname] = {"ground_truth": gt, "failure_mode": mode}

    merged = []
    for r in results:
        fname = r["filename"].strip()
        if fname in label_map:
            r["ground_truth"] = label_map[fname]["ground_truth"]
            r["failure_mode"] = label_map[fname]["failure_mode"]
            merged.append(r)
        else:
            print(f"Warning: {fname} in results but not in labels — skipping.")
    return merged


def compute_confusion_matrix(data: list) -> dict:
    """
    3x3 confusion matrix: usable / borderline / retake.
    For false-reject and false-accept, we collapse borderline into retake
    (borderline prompts a user action, same as retake from UX perspective).
    """
    classes = ["usable", "borderline", "retake"]
    matrix = {gt: {pred: 0 for pred in classes} for gt in classes}

    for row in data:
        gt = row["ground_truth"]
        pred = row["verdict"]
        if gt in classes and pred in classes:
            matrix[gt][pred] += 1

    return matrix


def print_confusion_matrix(matrix: dict):
    """Pretty-print the confusion matrix."""
    classes = ["usable", "borderline", "retake"]
    print("\nConfusion Matrix (rows=ground truth, cols=predicted):")
    print(f"{'':>15s} | {'usable':>10s} | {'borderline':>10s} | {'retake':>10s} | {'total':>6s}")
    print("-" * 65)
    for gt in classes:
        row = matrix.get(gt, {})
        total = sum(row.values())
        print(
            f"{gt:>15s} | {row.get('usable', 0):>10d} | "
            f"{row.get('borderline', 0):>10d} | "
            f"{row.get('retake', 0):>10d} | {total:>6d}"
        )


def compute_error_rates(matrix: dict) -> dict:
    """
    False-reject rate: usable images classified as retake OR borderline.
    False-accept rate: retake images classified as usable.
    """
    usable_row = matrix.get("usable", {})
    total_usable = sum(usable_row.values())
    false_rejects = usable_row.get("retake", 0) + usable_row.get("borderline", 0)

    retake_row = matrix.get("retake", {})
    total_retake = sum(retake_row.values())
    false_accepts = retake_row.get("usable", 0)

    # Include borderline ground truth images that were accepted
    borderline_row = matrix.get("borderline", {})
    total_borderline = sum(borderline_row.values())

    frr = false_rejects / total_usable if total_usable > 0 else 0.0
    far = false_accepts / total_retake if total_retake > 0 else 0.0

    return {
        "false_reject_rate": round(frr, 4),
        "false_accept_rate": round(far, 4),
        "false_rejects_count": false_rejects,
        "total_usable": total_usable,
        "false_accepts_count": false_accepts,
        "total_retake": total_retake,
    }


def compute_per_failure_mode(data: list) -> dict:
    """
    Per-failure-mode accuracy: for each failure mode in the labels,
    what fraction was correctly classified as retake or borderline?
    """
    mode_stats = defaultdict(lambda: {"total": 0, "detected": 0, "missed": 0})

    for row in data:
        mode = row.get("failure_mode", "")
        if not mode or row["ground_truth"] == "usable":
            continue
        mode_stats[mode]["total"] += 1
        if row["verdict"] in ("retake", "borderline"):
            mode_stats[mode]["detected"] += 1
        else:
            mode_stats[mode]["missed"] += 1

    result = {}
    for mode, stats in sorted(mode_stats.items()):
        detection_rate = stats["detected"] / stats["total"] if stats["total"] > 0 else 0
        result[mode] = {
            "total": stats["total"],
            "detected": stats["detected"],
            "missed": stats["missed"],
            "detection_rate": round(detection_rate, 4),
        }
    return result


def compute_latency_stats(data: list) -> dict:
    """Compute latency percentiles from results."""
    latencies = []
    for row in data:
        lat = row.get("latency_ms")
        if lat is not None:
            latencies.append(float(lat))

    if not latencies:
        return {"p50": 0, "p95": 0, "max": 0}

    latencies.sort()
    n = len(latencies)
    return {
        "p50_ms": round(latencies[int(n * 0.50)], 2),
        "p95_ms": round(latencies[min(int(n * 0.95), n - 1)], 2),
        "max_ms": round(latencies[-1], 2),
        "mean_ms": round(sum(latencies) / n, 2),
    }


def identify_misclassifications(data: list) -> list:
    """Return list of misclassified images for failure analysis."""
    misclassified = []
    for row in data:
        gt = row["ground_truth"]
        pred = row["verdict"]

        is_wrong = False
        error_type = ""

        if gt == "usable" and pred in ("retake", "borderline"):
            is_wrong = True
            error_type = "FALSE_REJECT"
        elif gt == "retake" and pred == "usable":
            is_wrong = True
            error_type = "FALSE_ACCEPT"
        elif gt == "borderline" and pred == "usable":
            is_wrong = True
            error_type = "FALSE_ACCEPT_BORDERLINE"

        if is_wrong:
            misclassified.append({
                "filename": row["filename"],
                "ground_truth": gt,
                "predicted": pred,
                "error_type": error_type,
                "reason": row.get("reason", ""),
                "failure_mode": row.get("failure_mode", ""),
                # Include key signals for analysis
                "laplacian_variance": row.get("laplacian_variance", ""),
                "tenengrad": row.get("tenengrad", ""),
                "fft_hf_ratio": row.get("fft_hf_ratio", ""),
                "brightness_mean": row.get("brightness_mean", ""),
                "glare_ratio": row.get("glare_ratio", ""),
            })
    return misclassified


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate detector results against ground-truth labels."
    )
    parser.add_argument(
        "--results", "-r",
        required=True,
        help="Path to results CSV from harness.py"
    )
    parser.add_argument(
        "--labels", "-l",
        required=True,
        help="Path to labels CSV (filename, ground_truth, failure_mode)"
    )
    parser.add_argument(
        "--output", "-o",
        default="evaluation_report.json",
        help="Output JSON path for full evaluation report."
    )
    args = parser.parse_args()

    # Load data
    results = load_csv(args.results)
    labels = load_csv(args.labels)
    data = merge_results_labels(results, labels)

    if not data:
        print("Error: No matching images between results and labels.")
        sys.exit(1)

    print(f"Evaluating {len(data)} images...\n")

    # Confusion matrix
    matrix = compute_confusion_matrix(data)
    print_confusion_matrix(matrix)

    # Error rates
    error_rates = compute_error_rates(matrix)
    print(f"\nFalse-Reject Rate: {error_rates['false_reject_rate']:.2%} "
          f"({error_rates['false_rejects_count']}/{error_rates['total_usable']})")
    print(f"False-Accept Rate: {error_rates['false_accept_rate']:.2%} "
          f"({error_rates['false_accepts_count']}/{error_rates['total_retake']})")

    # Per-failure-mode
    mode_stats = compute_per_failure_mode(data)
    if mode_stats:
        print("\nPer-Failure-Mode Detection Rates:")
        print(f"{'Mode':>20s} | {'Total':>6s} | {'Detected':>8s} | {'Missed':>6s} | {'Rate':>8s}")
        print("-" * 60)
        for mode, stats in mode_stats.items():
            print(
                f"{mode:>20s} | {stats['total']:>6d} | "
                f"{stats['detected']:>8d} | {stats['missed']:>6d} | "
                f"{stats['detection_rate']:.2%}"
            )

    # Latency
    lat_stats = compute_latency_stats(data)
    print(f"\nLatency: p50={lat_stats['p50_ms']}ms, "
          f"p95={lat_stats['p95_ms']}ms, max={lat_stats['max_ms']}ms")

    # Misclassifications
    misclassified = identify_misclassifications(data)
    print(f"\nMisclassified images: {len(misclassified)}")
    for m in misclassified[:10]:
        print(f"  {m['filename']}: {m['ground_truth']} → {m['predicted']} "
              f"({m['error_type']}) [{m['reason'][:60]}]")

    # Write full report
    report = {
        "total_images": len(data),
        "confusion_matrix": matrix,
        "error_rates": error_rates,
        "per_failure_mode": mode_stats,
        "latency": lat_stats,
        "misclassified": misclassified,
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {args.output}")


if __name__ == "__main__":
    main()
