# On-Capture Image Quality Detector

A CPU-only, inference-free image quality detection system that evaluates photographed answer sheets at the moment of capture and returns a verdict: **usable**, **retake**, or **borderline**.

Built using classical image processing techniques only. No machine learning models, no neural networks, no external API calls.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic test set (for development)
python generate_test_images.py --output test_images/ --count 120

# 3. Run detector on a single image
python detector.py path/to/image.jpg

# 4. Run full evaluation harness
python harness.py --images test_images/ --labels labels.csv --histograms

# 5. Generate histogram plots for threshold derivation
python plot_histograms.py --input histograms/ --output plots/
```

## Output Contract

Every call to `detect(image_path)` returns:

```json
{
  "verdict": "usable",
  "reason": "image is acceptable",
  "signals": {
    "laplacian_var": 45.23,
    "laplacian_p95": 112.56,
    "tenengrad": 12.34,
    "fft_hf_ratio": 0.012345,
    "glare_ratio": 0.0234,
    "mean_brightness": 178.45,
    "contrast": 42.12,
    "border_content_ratio": 0.0567
  },
  "latency_ms": 28.45
}
```

## Signals

| Signal | What it measures | Failure mode |
|--------|-----------------|-------------|
| `laplacian_var` | Variance of Laplacian operator (overall sharpness) | Motion blur, defocus |
| `laplacian_p95` | 95th percentile of patch-level Laplacian variance | Blur (content-aware; handles faint pencil) |
| `tenengrad` | Mean Sobel gradient magnitude | Motion blur, defocus |
| `fft_hf_ratio` | Ratio of high-frequency energy in FFT spectrum | Motion blur, defocus |
| `glare_ratio` | Fraction of overexposed/desaturated pixels | Glare |
| `mean_brightness` | Average pixel intensity | Insufficient/excessive lighting |
| `contrast` | Standard deviation of pixel intensities | Low contrast, poor lighting |
| `border_content_ratio` | Fraction of content pixels in border strips | Partial framing |

## Architecture

```
detector.py             Core detection function + all signal computations
harness.py              Batch runner, metrics, confusion matrix, failure analysis
generate_test_images.py Synthetic dataset generator for development
plot_histograms.py      Signal distribution plotter for threshold derivation
labels.csv              Ground-truth labels (filename, ground_truth, failure_mode)
test_images/            Test image directory
```

## Key Design Decisions

### Content-Aware Blur Detection
A naive Laplacian variance threshold fails on faint pencil writing: low high-frequency content gets misclassified as blurry. The `laplacian_p95` signal solves this by computing Laplacian variance in 64x64 patches and taking the 95th percentile. Blank-paper patches are ignored; only the sharpest content regions determine the blur verdict.

### Multi-Signal Blur Voting
The detector uses four independent blur signals. The verdict logic takes the *best* (most lenient) score unless 3+ signals agree the image should be rejected. This minimizes false rejects, which the problem statement identifies as the primary UX risk.

### Glare vs. White Paper
White paper is naturally bright. The glare detector requires both high brightness (V > 240) AND low saturation (S < 30) to count a pixel as glare, preventing false positives on clean white paper.

## Labels CSV Format

```
filename,ground_truth,failure_mode
img_001.jpg,usable,none
img_002.jpg,retake,motion_blur
img_003.jpg,retake,defocus
img_004.jpg,retake,glare
img_005.jpg,retake,framing
img_006.jpg,retake,lighting
img_007.jpg,borderline,defocus
```

Valid `failure_mode` values: `none`, `motion_blur`, `defocus`, `glare`, `framing`, `lighting`

## Reproducing Results

All reported metrics can be reproduced with:

```bash
python harness.py --images test_images/ --labels labels.csv --histograms --output results.csv
```

This outputs:
- `results.csv` — per-image verdicts and signals
- `results_summary.json` — confusion matrix, rates, latency, misclassified
- `histograms/` — signal distribution CSVs for histogram plotting

## Threshold Tuning

To adjust thresholds, modify the `Thresholds` dataclass in `detector.py`. Run the harness with `--histograms` to generate signal distributions, then use `plot_histograms.py` to visualize class separation and pick thresholds from the data.

## System Requirements

- Python 3.8+
- CPU only (no GPU required)
- Tested on: [YOUR CPU HERE — e.g., "Intel i5-12400, 16GB RAM"]
- Target latency: <100ms per 8-12 MP image (sub-50ms typical after downsampling)
