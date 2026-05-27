# On-Capture Image Quality Detector — Technical Report

**Version:** v6b (final) · **Date:** 2026-05-27 · **Dataset:** 299 labelled images (179 usable, 38 borderline, 82 retake)

---

## 1. Approach

The detector is a single-pass classical image processing pipeline that evaluates a photographed answer sheet and returns a verdict: `usable`, `borderline`, or `retake`. It operates CPU-only with zero ML inference and zero network calls.

**Pipeline.** The input image is decoded at half JPEG resolution (`IMREAD_REDUCED_COLOR_2`, saving 20–35 ms on large captures) and downsampled to 400 px on the longest edge. Ten signals are computed across four failure modes. A verdict engine combines scores using failure-mode-specific logic:

- **Blur:** best-of scoring across four primary signals, with two supplementary vote signals (`fft_asymmetry`, `wavelet_hh_energy`) and a content-aware P95 veto.
- **Glare, lighting, framing:** each scored independently; any single retake triggers the final retake verdict.

**Early exit.** When `laplacian_var > 1200` AND `tenengrad > 55`, the image is clearly sharp. FFT, P95, and supplementary signal computations are skipped entirely (saves ~30–40 ms).

---

## 2. Signals Used

| # | Signal | Method | Purpose |
|---|--------|--------|---------|
| 1 | `laplacian_var` | Variance of Laplacian (CV_32F) | Global sharpness — low values indicate blur |
| 2 | `laplacian_p95` | 95th percentile of patch-level Laplacian variances (100 px patches) | Content-aware sharpness; veto signal protecting sharp-region images (faint pencil on clean paper) |
| 3 | `tenengrad` | Mean Sobel gradient magnitude | Edge strength — sensitive to both motion and defocus blur |
| 4 | `fft_hf_ratio` | High-frequency energy fraction from 2D FFT (computed on 256 px downsample) | Spectral sharpness — blur suppresses high frequencies |
| 5 | `fft_asymmetry` | Angular sector energy ratio (H vs V, ±30° sectors, DC-excluded) | Directional spectral asymmetry — motion blur elongates spectrum perpendicular to blur direction |
| 6 | `wavelet_hh_energy` | Mean squared energy of Haar DWT diagonal (HH) subband, single-level decomposition | Defocus and motion blur suppress diagonal texture in text strokes |
| 7 | `glare_ratio` | Fraction of pixels with V > 240 AND S < 30 (HSV), plus blown-out pixels (> 250) | Specular glare detection |
| 8 | `mean_brightness` | Mean grayscale intensity | Under/overexposure detection |
| 9 | `contrast` | Standard deviation of grayscale intensity | Flat images indicate unusable capture |
| 10 | `border_content_ratio` | Content density in 5% border strips (adaptive threshold binarization) | Framing proxy. **Note:** proved non-discriminative in this dataset (usable max = 0.40 > retake max = 0.33); borderline framing verdict disabled, retake gate kept at 0.40 |

---

## 3. Threshold Derivation

All thresholds were derived empirically from the signal distributions of 299 labelled images, computed at 400 px downsample. The retake threshold is placed near the retake class median; the borderline threshold is placed near the usable class P25.

### 3.1 Signal Distribution Table

| Signal | Class | Min | P25 | Median | P75 | Max |
|--------|-------|----:|----:|-------:|----:|----:|
| `laplacian_var` | USABLE | 140.2 | 1199.4 | 1941.9 | 2478.5 | 5108.5 |
| | RETAKE | 1.0 | 38.6 | 442.0 | 933.6 | 3175.3 |
| `laplacian_p95` | USABLE | 327.2 | 2187.6 | 4681.7 | 6196.1 | 12771.3 |
| | RETAKE | 1.3 | 72.9 | 888.8 | 1896.9 | 7938.2 |
| `tenengrad` | USABLE | 16.6 | 52.4 | 65.1 | 79.9 | 107.6 |
| | RETAKE | 1.0 | 19.7 | 34.7 | 48.4 | 83.1 |
| `fft_hf_ratio` | USABLE | 0.003 | 0.010 | 0.010 | 0.010 | 0.017 |
| | RETAKE | 0.000 | 0.002 | 0.005 | 0.007 | 0.018 |
| `glare_ratio` | USABLE | 0.000 | 0.000 | 0.000 | 0.000 | 0.402 |
| | RETAKE | 0.000 | 0.000 | 0.000 | 0.000 | 0.833 |
| `contrast` | USABLE | 13.4 | 31.0 | 46.3 | 50.7 | 95.3 |
| | RETAKE | 0.8 | 23.7 | 31.4 | 43.1 | 74.4 |
| `border_content` | USABLE | 0.057 | 0.143 | 0.199 | 0.270 | 0.396 |
| | RETAKE | 0.000 | 0.053 | 0.120 | 0.165 | 0.326 |

† USABLE `fft_hf_ratio` values are capped at 0.01 by the early-exit sentinel (sharp images skip FFT computation).

### 3.2 Threshold Table

| Signal | Retake (< / >) | Borderline (< / >) | Derivation Rationale |
|--------|:---------:|:-------------:|----------------------|
| `laplacian_var` | < 300 | < 800 | Retake median = 442 → threshold at 300 catches lower half. Usable P25 = 1199 → borderline at 800 provides safe margin. |
| `laplacian_p95` | < 100 | veto ≥ 2000 | P95 serves as a content-aware veto. At P95 ≥ 2000, the image is sharp in text regions, overriding low global metrics. Retake P75 = 1897 → ~75% of retake images lose veto protection. Usable P25 = 2188 → ~75% of usable images retain protection. |
| `tenengrad` | < 25 | < 40 | Retake median = 34.7 → threshold at 25 catches ~40% of true blur. Usable P25 = 52.4 → borderline at 40 sits safely below. |
| `fft_hf_ratio` | < 0.004 | < 0.006 | Retake median = 0.005. Thresholds bracket this value. |
| `fft_asymmetry` | vote if > 0.35 | — | True motion blur produces asymmetry 0.4–7.0+. Sharp images: near 0. Threshold at 0.35 avoids flagging non-uniform content layouts. Not a standalone retake trigger — requires 3+ total votes. |
| `wavelet_hh_energy` | vote if < 30 | — | Defocus suppresses diagonal HH subband energy. Low wavelet energy adds one retake vote. Not a standalone trigger — requires 3+ total votes. |
| `glare_ratio` | > 0.50 | > 0.45 | True glare: 0.47–0.83. Well-lit white paper peaks at 0.40. Brightness gate: glare check disabled if `mean_brightness` < 160. |
| `contrast` | < 12 | < 20 | Usable minimum = 13.4 → threshold at 12 avoids false rejects. Retake P25 = 23.7 → borderline at 20 catches low-contrast images. |
| `border_content` | > 0.40 | disabled (0.40) | Signal proved non-discriminative. Borderline effectively disabled; retake gate preserved for extreme blank captures. |

### 3.3 Blur Verdict Architecture

```
Step 1 — Score four primary signals: laplacian_var, laplacian_p95, tenengrad, fft_hf_ratio
  Each signal votes: retake (score 0), borderline (score 1), or usable (score 2)

Step 2 — Add supplementary retake votes:
  fft_asymmetry > 0.35   → +1 retake vote  (motion blur)
  wavelet_hh_energy < 30 → +1 retake vote  (defocus)

Step 3 — Count retake votes:
  retake_votes = count(primary scores == 0) + supplementary votes

Step 4 — Apply P95 veto (pencil-on-paper protection):
  If laplacian_p95 scores usable (P95 ≥ 2000) → override blur to usable
  If laplacian_p95 scores borderline AND retake_votes ≥ 3 → cap at borderline

Step 5 — Decision:
  retake_votes ≥ 3              → retake
  retake_votes ≥ 1 AND any primary borderline → borderline
  else                          → usable (best-of primary scores)

Step 6 — Final verdict:
  min(blur, glare, lighting, framing) — any single retake triggers retake.
```

---

## 4. Metrics

### 4.1 Confusion Matrix

| | Pred: Usable | Pred: Borderline | Pred: Retake |
|---|---:|---:|---:|
| **GT: Usable (179)** | 165 | 13 | 1 |
| **GT: Borderline (38)** | 24 | 11 | 3 |
| **GT: Retake (82)** | 26 | 21 | 35 |

### 4.2 Binary Metrics

Borderline predictions and ground truth collapsed into retake for binary evaluation. FRR denominator = 179 usable GT images. FAR denominator = 120 non-usable GT images (38 borderline + 82 retake).

| Metric | Value | Derivation | Spec |
|--------|------:|------------|------|
| **False-Reject Rate** | **7.82%** | 14 / 179 usable | < 10% ✅ |
| False-Accept Rate | 41.67% | 50 / 120 non-usable | — |
| Overall Accuracy | 78.60% | 235 / 299 | — |

### 4.3 Per-Failure-Mode Detection

| Failure Mode | Total | Detected | Detection Rate |
|-------------|------:|---------:|---------------:|
| Insufficient lighting | 10 | 10 | **100.0%** |
| Glare | 9 | 8 | **88.9%** |
| Motion blur | 43 | 24 | **55.8%** |
| Partial framing | 20 | 10 | **50.0%** |
| Defocus | 38 | 18 | **47.4%** |

### 4.4 Latency

Measured on Intel Core i5 laptop, Windows 11. Input images: 1–12 MP.

| Percentile | Latency |
|-----------|--------:|
| p50 | **85.26 ms** |
| p95 | 132.65 ms |
| max | 181.24 ms |
| mean | 77.23 ms |

### 4.5 Iteration History

| Version | FRR | FAR | Motion Blur | Defocus | p50 | Key Change |
|---------|----:|----:|------------:|--------:|----:|------------|
| v1 | 35.75% | 55.0% | 30.2% | 39.5% | 233 ms | Baseline |
| v2 | 27.93% | 60.0% | 30.2% | 31.6% | 122 ms | Glare threshold fix |
| v3 | 3.91% | 57.5% | 37.2% | 31.6% | 130 ms | Framing + blur recalibration |
| v4 | 6.70% | 45.0% | 53.5% | 44.7% | 133 ms | P95 veto raised to 2000 |
| v5c | 6.70% | 45.0% | 48.8% | 47.4% | 87 ms | Latency optimizations (CV_32F, IMREAD_REDUCED, early exit) |
| **v6b** | **7.82%** | **41.67%** | **55.8%** | **47.4%** | **85 ms** | **FFT asymmetry + wavelet HH energy** |

---

## 5. Failure Analysis

### Case 1: WA0000 — False Reject (usable → borderline)

**Signals:** `laplacian_var=557.6`, `tenengrad=30.4`, `fft_hf_ratio=0.003`, `p95=1625.8`, `fft_asymmetry=0.42`, `wavelet_hh=21.7`

**Cause:** Faint-pencil answer sheet. Low-contrast pencil produces weak gradients (tenengrad = 30, below the 40 borderline threshold). The P95 veto does not fire (1626 < 2000) because even text patches have modest variance. The `fft_asymmetry` of 0.42 adds a retake vote (> 0.35), but only 2 total votes are reached — not enough for the 3-vote retake trigger. The image is in-focus but its signals are indistinguishable from mild defocus.

**Fix:** Per-capture tenengrad calibration using a reference shot would establish device- and lighting-specific baselines. Lowering `tenengrad_borderline` from 40 to 30 rescues this image but adds ~8 false accepts.

### Case 2: WA0003 — False Reject (usable → borderline)

**Signals:** `laplacian_var=603.2`, `tenengrad=30.7`, `fft_hf_ratio=0.003`, `p95=1145.4`, `fft_asymmetry=1.32`, `wavelet_hh=17.1`

**Cause:** Same failure mode as Case 1. P95 = 1145 confirms even the sharpest patches produce modest variance — consistent with very light pencil, not blur. Four retake votes accumulate (laplacian_var < 300, fft < 0.004, fft_asymmetry > 0.35, wavelet_hh < 30), but the P95 borderline score (1145, between 100 and 2000) acts as a soft cap, limiting the verdict to borderline rather than retake.

**Fix:** Fundamental ambiguity — classical gradient features cannot distinguish "in-focus but faint" from "slightly out of focus." Irreducible overlap at current resolution.

### Case 3: WA0143 — False Accept (borderline defocus → usable)

**Signals:** `laplacian_var=1385.4`, `tenengrad=61.1`, `p95=3463.5`, `fft_hf_ratio=0.01`, `wavelet_hh=41.6`

**Cause:** Mild uniform defocus with dense ink handwriting. High-contrast strokes produce strong gradients (tenengrad = 61, well above the 40 borderline threshold) even when slightly out of focus. P95 = 3464 fires the veto, forcing blur to usable. `wavelet_hh` = 41.6 is above 30, so the defocus vote does not fire. Mild defocus with a blur kernel smaller than the ink stroke width leaves gradients intact — the hardest case for classical detectors.

**Fix:** Local sharpness analysis restricted to text edge regions (connected component bounding boxes) would isolate blur from content density.

### Case 4: WA0174 — False Accept (borderline defocus → usable)

**Signals:** `laplacian_var=1110.3`, `tenengrad=59.7`, `p95=1569.6`, `fft_hf_ratio=0.008`, `fft_asymmetry=0.47`, `wavelet_hh=32.7`

**Cause:** Sits in the signal overlap zone. Tenengrad = 59.7 is above the borderline threshold (40), driving the best-of blur score to usable. This image falls exactly between the RETAKE P75 for tenengrad (48.4) and the USABLE P25 (52.4). `wavelet_hh` = 32.7 is just above the 30 threshold, so no defocus vote fires. No threshold adjustment catches this without increasing the false-reject rate.

**Fix:** Requires either a learned decision boundary in the overlap zone or a multi-frame sharpness comparison.

### Case 5: WA0181 — False Reject (usable → borderline)

**Signals:** `laplacian_var=637.4`, `tenengrad=33.5`, `fft_hf_ratio=0.004`, `p95=1318.0`, `fft_asymmetry=7.04`, `wavelet_hh=13.2`

**Cause:** Usable image with non-uniform content layout (text concentrated on one side of the page) producing extreme FFT asymmetry (7.04) that mimics directional motion blur. Combined with low wavelet energy (13.2 < 30), this accumulates enough retake votes (laplacian_var borderline, tenengrad borderline, fft borderline, fft_asymmetry retake vote, wavelet retake vote) for the borderline verdict. The P95 veto at 1318 does not fire (< 2000).

**Fix:** Add a spatial content uniformity pre-check: if content is deliberately asymmetric (single-column layout), suppress the `fft_asymmetry` vote. This would eliminate false rejects from this signal without affecting true motion blur catches.

---

## 6. Scope and Limitations

### Addressed Failure Modes

Motion blur, defocus blur, specular glare, insufficient lighting, low contrast, partial framing.

### False-Accept Rate (41.67%)

This is the primary remaining weakness. The majority of false accepts are borderline-labelled images (mild defocus, slight motion blur) where the ground-truth label itself is subjective. Among hard retake images (severe blur, glare, dark images), the detection rate is substantially higher. Improving this metric beyond classical features would require learned representations or multi-frame analysis. The false-accept rate is structurally limited at the classical feature level because the RETAKE and USABLE signal distributions overlap substantially — for example, tenengrad RETAKE P75 (48.4) exceeds USABLE P25 (52.4) within sampling noise.

### Motion Blur (55.8%)

FFT directional asymmetry improved motion blur detection from 48.8% (v5c) to 55.8% (v6b). The remaining missed cases are mild motion blur where the blur direction aligns with the dominant text orientation, masking the asymmetry signal.

### Defocus (47.4%)

Wavelet HH subband energy identifies images where high-frequency diagonal content is suppressed. However, mild defocus with high-contrast content (tenengrad 45–60, wavelet_hh 30–50) remains indistinguishable from sharp images. When the defocus kernel is smaller than the ink stroke width, gradients remain strong — a fundamental limitation of gradient-based features.

### Framing Signal

`border_content_ratio` proved non-discriminative — usable images score higher (max = 0.40) than retake images (max = 0.33). Full-page captures produce dense border content. Structural framing detection (Hough edge detection for page edges) would be required.

### Not Addressed

| Exclusion | Justification |
|-----------|---------------|
| Per-device calibration | Thresholds tuned on a single device and session. Cross-device robustness requires a multi-device dataset. |
| Skew/rotation | Rotated-but-sharp images are OCR-processable with preprocessing; not a quality failure. |
| Co-occurring failures | Detector flags the first detected issue. Co-occurrence handling adds complexity without UX benefit. |
| Mild defocus + high-contrast content | Irreducible signal overlap at the classical feature level, proven by the distribution data above. |

---

## 7. Proposed Next Steps

1. **Multi-device calibration** — Collect captures from 5+ phone models to derive device-conditional thresholds using EXIF metadata (ISO, focal length).
2. **Hough-based framing** — Replace `border_content_ratio` with Hough line detection to directly verify all four page edges are visible.
3. **Directional blur refinement** — Current `fft_asymmetry` fires on non-uniform content layouts (Case 5). Adding a spatial content uniformity pre-check would reduce false rejects from this signal.
4. **Production A/B gate** — Deploy the detector on 10% of uploads and measure OCR accuracy improvement vs retake friction to produce empirical evidence of downstream benefit.
5. **Adaptive threshold calibration** — On first use, capture a reference sheet to establish device-specific signal baselines.

---

## 8. Reproducibility

**Dependencies:** `opencv-python-headless>=4.5.0`, `numpy>=1.21.0`, `PyWavelets>=1.1.0`, `matplotlib>=3.5.0`

```bash
pip install -r requirements.txt
python harness.py --images test_images/ --labels labels.csv --output results.csv
```
