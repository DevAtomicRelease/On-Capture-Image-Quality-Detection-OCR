"""
Plot signal distribution histograms from CSV files.

Reads per-signal CSVs (value, class) and produces overlapping histograms
showing USABLE vs RETAKE distributions with threshold lines.

Usage:
    python plot_histograms.py --input histograms/ --output plots/
"""

import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt


# Threshold lines to overlay on each signal's histogram.
# Format: signal_name -> [(value, label, style), ...]
THRESHOLDS = {
    "laplacian_var": [
        (300, "retake < 300", "--"),
        (800, "borderline < 800", ":"),
    ],
    "laplacian_p95": [
        (100, "retake < 100", "--"),
        (2000, "veto ≥ 2000", ":"),
    ],
    "tenengrad": [
        (25, "retake < 25", "--"),
        (40, "borderline < 40", ":"),
    ],
    "fft_hf_ratio": [
        (0.004, "retake < 0.004", "--"),
        (0.006, "borderline < 0.006", ":"),
    ],
    "glare_ratio": [
        (0.50, "retake > 0.50", "--"),
        (0.45, "borderline > 0.45", ":"),
    ],
    "contrast": [
        (12, "retake < 12", "--"),
        (20, "borderline < 20", ":"),
    ],
    "mean_brightness": [
        (40, "retake < 40", "--"),
        (245, "retake > 245", "--"),
    ],
    "border_content_ratio": [
        (0.40, "retake > 0.40", "--"),
    ],
}


def load_csv(filepath):
    """Load a histogram CSV and return (usable_values, retake_values)."""
    usable = []
    retake = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = float(row["value"])
            cls = row["class"].strip().lower()
            if cls == "usable":
                usable.append(val)
            else:
                retake.append(val)
    return np.array(usable), np.array(retake)


def plot_histogram(usable, retake, signal_name, output_path):
    """Generate an overlapping histogram with threshold lines."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Determine common bin range
    all_vals = np.concatenate([usable, retake])
    lo, hi = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
    margin = (hi - lo) * 0.05
    bins = np.linspace(lo - margin, hi + margin, 50)

    # Plot histograms
    ax.hist(usable, bins=bins, alpha=0.55, color="#2ecc71", label=f"USABLE (n={len(usable)})",
            edgecolor="white", linewidth=0.5)
    ax.hist(retake, bins=bins, alpha=0.55, color="#e74c3c", label=f"RETAKE (n={len(retake)})",
            edgecolor="white", linewidth=0.5)

    # Threshold lines
    thresholds = THRESHOLDS.get(signal_name, [])
    for val, label, style in thresholds:
        if lo - margin <= val <= hi + margin:
            ax.axvline(x=val, color="#2c3e50", linestyle=style, linewidth=1.5,
                       label=label)

    # Annotation: class statistics
    u_med = np.median(usable) if len(usable) > 0 else 0
    r_med = np.median(retake) if len(retake) > 0 else 0
    stats_text = f"USABLE median: {u_med:.2f}\nRETAKE median: {r_med:.2f}"
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    ax.set_xlabel(signal_name, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Signal Distribution: {signal_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot signal distribution histograms.")
    parser.add_argument("--input", required=True, help="Directory containing dist_*.csv files")
    parser.add_argument("--output", required=True, help="Directory to save plot images")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    csv_files = sorted([f for f in os.listdir(args.input) if f.startswith("dist_") and f.endswith(".csv")])

    if not csv_files:
        print(f"No dist_*.csv files found in {args.input}")
        return

    for csv_file in csv_files:
        signal_name = csv_file.replace("dist_", "").replace(".csv", "")
        csv_path = os.path.join(args.input, csv_file)
        png_path = os.path.join(args.output, f"{signal_name}.png")

        usable, retake = load_csv(csv_path)
        plot_histogram(usable, retake, signal_name, png_path)
        print(f"  Saved: {png_path}")

    print(f"\nDone. {len(csv_files)} plots saved to {args.output}/")


if __name__ == "__main__":
    main()
