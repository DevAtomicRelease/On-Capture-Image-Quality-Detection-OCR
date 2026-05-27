"""
On-Capture Image Quality Detector
==================================
Classical image processing pipeline that evaluates a photographed answer sheet
and returns a verdict: usable, retake, or borderline.

Signals implemented:
    1. Laplacian variance (overall + content-aware 95th percentile)
    2. Tenengrad (Sobel gradient magnitude)
    3. FFT high-frequency energy ratio
    4. FFT directional asymmetry (motion blur indicator)
    5. Wavelet HH subband energy (defocus indicator)
    6. Glare detection (overexposed area ratio)
    7. Lighting assessment (brightness + contrast)
    8. Framing check (border content density)

No ML inference. CPU only. Target: <100ms per 8-12 MP image.
"""

import cv2
import numpy as np
import pywt
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional



# Configuration

@dataclass
class Thresholds:
    """
    Central threshold configuration.
    All values are derived from signal distributions across labelled data.
    See report for histograms and justification.

    CHANGELOG (v2 — 2026-05-27):
    ─────────────────────────────
    LATENCY:
      • downsample_width 1200→500  — 5.7× fewer pixels, main latency win
      • patch_size in compute_laplacian_p95 64→100 (in function default)
      • FFT computed on 256px further-downsample (in detect())

    GLARE (false-reject fix):
      • glare_ratio_high 0.15→0.50 — usable white-paper images reached 0.42;
        true glare images are 0.47+; clean separation at 0.50
      • glare_ratio_mid 0.08→0.45 — borderline only for 0.45-0.50 range
      • Added brightness gate (< 160) in determine_verdict — dark images
        cannot have specular glare

    BLUR (detection improvement):
      • laplacian_var_low 15→80 — old threshold only caught extreme blur;
        motion-blur false-accepts ranged 14-878
      • laplacian_var_mid 30→200 — borderline band for moderate blur
      • laplacian_p95_low 25→100, p95_mid 50→350
      • tenengrad_low 4→12, tenengrad_mid 8→20
      • fft_hf_ratio_low 0.002→0.001, fft_mid 0.008→0.003

    FRAMING (false-borderline fix):
      • border_content_mid 0.18→0.25 — 42 usable images had border 0.18-0.21
      • border_content_high 0.30→0.40 — retake only for extreme framing

    CONTRAST (false-borderline fix):
      • contrast_low 15→12, contrast_mid 25→20 — 7 usable images had
        contrast 15-25 due to low-light or even lighting
    """
    # Blur: Laplacian variance (on downsampled image)
    laplacian_var_low: float = 300.0   # retake below this (RETAKE median=394.5)
    laplacian_var_mid: float = 800.0   # borderline below this (USABLE p25=1241)

    # Blur: Content-aware Laplacian (95th percentile of patch variances)
    # Handles the faint-pencil-on-clean-paper problem.
    laplacian_p95_low: float = 100.0   # retake below this (was 25)
    laplacian_p95_mid: float = 2000.0  # veto threshold (RETAKE p75=2255; USABLE p25=2348)

    # Blur: Tenengrad (mean Sobel gradient magnitude)
    tenengrad_low: float = 25.0        # retake below this (RETAKE median=33.1)
    tenengrad_mid: float = 40.0        # borderline below this (USABLE p25=50.5)

    # Blur: FFT high-frequency energy ratio
    fft_hf_ratio_low: float = 0.004    # retake below this (RETAKE median=0.0048)
    fft_hf_ratio_mid: float = 0.006    # borderline below this (between medians)

    # Glare: fraction of image area with overexposed pixels
    # True glare images: 0.47-0.84. False-reject white paper: 0.22-0.42.
    glare_ratio_high: float = 0.50     # retake above this (was 0.15)
    glare_ratio_mid: float = 0.45      # borderline above this (was 0.08)

    # Lighting: mean brightness (0-255)
    brightness_low: float = 40.0
    brightness_high: float = 245.0
    brightness_low_mid: float = 60.0
    brightness_high_mid: float = 235.0

    # Lighting: standard deviation of brightness (contrast proxy)
    contrast_low: float = 12.0         # retake below this (was 15)
    contrast_mid: float = 20.0         # borderline below this (was 25)

    # Framing: content ratio near borders
    # 42 usable images had border_content 0.18-0.21; raised to avoid false borderlines
    border_content_high: float = 0.40  # retake above this (was 0.30)
    border_content_mid: float = 0.40   # borderline disabled — signal not discriminative

    # Processing
    downsample_width: int = 400        # was 500; 36% fewer pixels for <100ms latency


DEFAULT_THRESHOLDS = Thresholds()


# Signal computation functions


def _downsample(image: np.ndarray, max_dim: int) -> np.ndarray:
    """Resize so the longer edge equals max_dim. Preserves aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def compute_laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the Laplacian. Low values indicate blur."""
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(np.var(lap))


def compute_laplacian_p95(gray: np.ndarray, patch_size: int = 100) -> float:
    """
    Content-aware sharpness: Laplacian variance in patches, return 95th pctl.

    Handles faint pencil on clean paper. Blank patches (low variance) are
    ignored because we take the 95th percentile, not the mean. Patches
    with actual writing will still show sharp edges if the image is in focus.

    A genuinely blurry image has low variance across ALL patches.
    """
    h, w = gray.shape
    variances = []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch = gray[y:y + patch_size, x:x + patch_size]
            lap = cv2.Laplacian(patch, cv2.CV_32F)
            variances.append(np.var(lap))

    if not variances:
        return 0.0
    return float(np.percentile(variances, 95))


def compute_tenengrad(gray: np.ndarray) -> float:
    """Tenengrad focus measure: mean Sobel gradient magnitude."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)  # C++-optimized, faster than np.sqrt
    return float(np.mean(magnitude))


def compute_fft_hf_ratio(gray: np.ndarray) -> float:
    """
    FFT high-frequency energy ratio.

    Compute 2D FFT, define a circular mask excluding the low-frequency center
    (radius = 15% of smaller dimension). Ratio = energy outside / total energy.

    Low ratio = image lacks high-frequency detail = likely blurry.
    """
    rows, cols = gray.shape
    f_transform = np.fft.fft2(gray.astype(np.float32))
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift).astype(np.float32)

    center_r, center_c = rows // 2, cols // 2
    radius = int(min(rows, cols) * 0.15)

    y_coords, x_coords = np.ogrid[:rows, :cols]
    dist = np.sqrt((y_coords - center_r) ** 2 + (x_coords - center_c) ** 2)

    total_energy = np.sum(magnitude ** 2)
    if total_energy < 1e-10:
        return 0.0

    hf_energy = np.sum(magnitude[dist > radius] ** 2)
    return float(hf_energy / total_energy)


def compute_glare_ratio(gray: np.ndarray, hsv: np.ndarray) -> float:
    """
    Detect glare / specular highlights.

    Glare pixels: V > 240 AND S < 30 (bright + desaturated).
    Also catches completely blown-out pixels (> 250).

    White paper is naturally bright but retains some saturation variation;
    the low-saturation condition separates actual glare from white paper.
    """
    v_channel = hsv[:, :, 2]
    s_channel = hsv[:, :, 1]

    glare_mask = (v_channel > 240) & (s_channel < 30)
    blown_out = gray > 250
    combined = glare_mask | blown_out

    total_pixels = gray.shape[0] * gray.shape[1]
    return float(np.sum(combined) / total_pixels)


def compute_lighting_stats(gray: np.ndarray) -> Tuple[float, float]:
    """
    Returns (mean_brightness, contrast_std_dev).

    Low mean = underexposed.  Very high mean = overexposed.
    Low std dev = flat histogram, poor contrast.
    """
    return float(np.mean(gray)), float(np.std(gray))


def compute_border_content_ratio(
    gray: np.ndarray, border_fraction: float = 0.05
) -> float:
    """
    Framing check: fraction of content pixels in border strips.

    High ratio = page likely extends beyond frame (partial framing).
    Low ratio = borders are clean, framing is OK.
    """
    # adaptiveThreshold: locally-adaptive binarization that correctly separates
    # text from paper regardless of brightness variation. Costs ~5ms more than
    # a fixed threshold but produces reliable border_content_ratio values.
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 10
    )

    h, w = binary.shape
    bh = max(int(h * border_fraction), 5)
    bw = max(int(w * border_fraction), 5)

    top = binary[:bh, :]
    bottom = binary[h - bh:, :]
    left = binary[:, :bw]
    right = binary[:, w - bw:]

    border_pixels = np.concatenate([
        top.ravel(), bottom.ravel(), left.ravel(), right.ravel()
    ])
    total_border = len(border_pixels)
    if total_border == 0:
        return 0.0

    return float(np.sum(border_pixels > 0) / total_border)


def compute_fft_asymmetry(gray: np.ndarray) -> float:
    """
    Directional FFT asymmetry for motion blur detection.

    Motion blur suppresses frequencies perpendicular to the blur direction,
    creating an elongated spectrum. Compares energy in horizontal vs vertical
    angular sectors of the power spectrum, excluding the DC (center) region
    which otherwise dominates and masks the directional signal.

    Returns max(h/v, v/h) - 1.0. Zero = symmetric, higher = directional blur.
    """
    rows, cols = gray.shape
    f_transform = np.fft.fft2(gray.astype(np.float32))
    f_shift = np.fft.fftshift(f_transform)
    power = np.abs(f_shift).astype(np.float32) ** 2

    center_r, center_c = rows // 2, cols // 2

    # Coordinate grids relative to center
    y, x = np.ogrid[:rows, :cols]
    y = y - center_r
    x = x - center_c

    # Distance from center — exclude DC region (< 5% of min dimension)
    dist_sq = y * y + x * x
    dc_radius_sq = (max(int(min(rows, cols) * 0.05), 2)) ** 2
    valid = dist_sq > dc_radius_sq

    # Angular sectors using atan2-free comparison (faster than arctan2):
    # Horizontal sector: |x| > |y| * tan(30°) ≈ |x| > |y| * 0.577
    # Vertical sector: |y| > |x| * tan(30°) ≈ |y| > |x| * 0.577
    abs_y = np.abs(y)
    abs_x = np.abs(x)
    h_mask = valid & (abs_x > abs_y * 0.577)  # within ±30° of horizontal
    v_mask = valid & (abs_y > abs_x * 0.577)  # within ±30° of vertical

    h_energy = float(np.sum(power[h_mask]))
    v_energy = float(np.sum(power[v_mask]))

    if v_energy < 1e-10 or h_energy < 1e-10:
        return 0.0

    ratio = max(h_energy, v_energy) / min(h_energy, v_energy)
    return float(ratio - 1.0)


def compute_wavelet_hh_energy(gray: np.ndarray) -> float:
    """
    Wavelet HH (diagonal detail) subband energy for defocus detection.

    A single-level Haar wavelet decomposition splits the image into:
    - LL (approximation), LH (horizontal detail), HL (vertical detail),
      HH (diagonal detail).

    Defocus blur acts as a low-pass filter, strongly suppressing high-frequency
    diagonal content (HH). Sharp images retain HH energy. Returns the mean
    squared energy of the HH subband.
    """
    _, (_, _, hh) = pywt.dwt2(gray.astype(np.float32), 'haar')
    return float(np.mean(hh ** 2))



# Verdict logic


def _classify_signal(value: float, low: float, mid: float, invert: bool = False) -> int:
    """
    Score a single signal: 2 = usable, 1 = borderline, 0 = retake.
    invert=True means high values are bad (glare, border content).
    """
    if invert:
        if value > low:
            return 0
        elif value > mid:
            return 1
        else:
            return 2
    else:
        if value < low:
            return 0
        elif value < mid:
            return 1
        else:
            return 2


def determine_verdict(
    signals: Dict[str, float], thresholds: Thresholds
) -> Tuple[str, str]:
    """
    Combine signal scores into final verdict and reason.

    Blur strategy: use the BEST of the sharpness signals. Only flag blur
    if 3+ signals agree the image is blurry. This minimizes false rejects
    on faint-pencil or sparse-content pages.

    Other failures (glare, lighting, framing): each evaluated independently.
    Any single failure mode scoring 0 triggers a retake verdict.
    """
    scores = {}
    reasons = {}

    # --- Blur ---
    blur_scores = {
        "laplacian_var": _classify_signal(
            signals["laplacian_var"],
            thresholds.laplacian_var_low, thresholds.laplacian_var_mid
        ),
        "laplacian_p95": _classify_signal(
            signals["laplacian_p95"],
            thresholds.laplacian_p95_low, thresholds.laplacian_p95_mid
        ),
        "tenengrad": _classify_signal(
            signals["tenengrad"],
            thresholds.tenengrad_low, thresholds.tenengrad_mid
        ),
        "fft_hf_ratio": _classify_signal(
            signals["fft_hf_ratio"],
            thresholds.fft_hf_ratio_low, thresholds.fft_hf_ratio_mid
        ),
    }

    # Best-of strategy with pencil-on-paper protection:
    # laplacian_p95 is the content-aware signal (95th percentile of patches).
    # If it says usable, the image has sharp content in text regions even if
    # global metrics are low (faint pencil on clean paper). In that case,
    # laplacian_p95 gets veto power — prevents false-reject on valid captures.
    best_blur = max(blur_scores.values())
    retake_count = sum(1 for v in blur_scores.values() if v == 0)

    # --- Additional blur votes from new signals (v6) ---
    # These add retake votes WITHOUT changing existing score entries.
    # fft_asymmetry: high asymmetry indicates directional motion blur
    if signals.get("fft_asymmetry", 0.0) > 0.35:
        retake_count += 1
    # wavelet_hh_energy: low HH energy indicates defocus blur
    if signals.get("wavelet_hh_energy", 999.0) < 30.0:
        retake_count += 1

    if blur_scores["laplacian_p95"] == 2:
        # Content-aware signal says clearly usable — trust it even if
        # global signals are low (pencil-on-paper case)
        best_blur = max(best_blur, 2)
    elif blur_scores["laplacian_p95"] == 1 and retake_count >= 3:
        # P95 says borderline but everything else says retake — borderline
        best_blur = 1
    elif retake_count >= 3:
        best_blur = 0

    scores["blur"] = best_blur
    if best_blur == 0:
        reasons["blur"] = "image is blurry (out of focus or motion blur)"
    elif best_blur == 1:
        reasons["blur"] = "image may be slightly blurry"

    # --- Glare ---
    # Brightness gate: images with mean_brightness < 160 cannot have
    # meaningful specular glare — the signal is noise on darker paper.
    if signals["mean_brightness"] < 160:
        scores["glare"] = 2  # dark image, skip glare check
    else:
        scores["glare"] = _classify_signal(
            signals["glare_ratio"],
            thresholds.glare_ratio_high,
            thresholds.glare_ratio_mid,
            invert=True,
        )
        if scores["glare"] == 0:
            reasons["glare"] = "significant glare or overexposure"
        elif scores["glare"] == 1:
            reasons["glare"] = "some glare detected"

    # --- Lighting ---
    brightness = signals["mean_brightness"]
    contrast = signals["contrast"]

    if brightness < thresholds.brightness_low or brightness > thresholds.brightness_high:
        scores["lighting"] = 0
        reasons["lighting"] = (
            "insufficient lighting"
            if brightness < thresholds.brightness_low
            else "severely overexposed"
        )
    elif brightness < thresholds.brightness_low_mid or brightness > thresholds.brightness_high_mid:
        scores["lighting"] = 1
        reasons["lighting"] = (
            "dim lighting"
            if brightness < thresholds.brightness_low_mid
            else "overexposed"
        )
    else:
        scores["lighting"] = 2

    contrast_score = _classify_signal(contrast, thresholds.contrast_low, thresholds.contrast_mid)
    if contrast_score < scores.get("lighting", 2):
        scores["lighting"] = contrast_score
        if contrast_score == 0:
            reasons["lighting"] = "very low contrast"
        elif contrast_score == 1:
            reasons["lighting"] = "low contrast"

    # --- Framing ---
    scores["framing"] = _classify_signal(
        signals["border_content_ratio"],
        thresholds.border_content_high,
        thresholds.border_content_mid,
        invert=True,
    )
    if scores["framing"] == 0:
        reasons["framing"] = "partial framing (page extends beyond image edges)"
    elif scores["framing"] == 1:
        reasons["framing"] = "possible framing issue at edges"

    # --- Final ---
    min_score = min(scores.values())
    if min_score == 0:
        verdict = "retake"
    elif min_score == 1:
        verdict = "borderline"
    else:
        verdict = "usable"

    active_reasons = [
        reasons[k] for k, v in scores.items() if v == min_score and k in reasons
    ]
    reason = "; ".join(active_reasons) if active_reasons else "image is acceptable"

    return verdict, reason



# Main detection function

def detect(
    image_path: str,
    thresholds: Optional[Thresholds] = None,
) -> Dict:
    """
    Evaluate a single image and return a structured verdict.

    Args:
        image_path: path to the image file
        thresholds: optional custom Thresholds object

    Returns:
        {
            "verdict": "usable" | "retake" | "borderline",
            "reason": "human-readable description",
            "signals": { metric_name: numeric_value, ... },
            "latency_ms": float
        }
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    t_start = time.perf_counter()

    # IMREAD_REDUCED_COLOR_2: decode JPEG at 1/2 resolution during DCT,
    # saving ~20-35ms on large captures. Falls back to normal for non-JPEG.
    # Using /2 (not /8) so WhatsApp images (~1600px) stay ≥800px before
    # downsample, ensuring consistent signal computation.
    image = cv2.imread(image_path, cv2.IMREAD_REDUCED_COLOR_2)
    if image is None:
        return {
            "verdict": "retake",
            "reason": "image file could not be read",
            "signals": {},
            "latency_ms": 0.0,
        }

    # Downsample for speed
    image = _downsample(image, thresholds.downsample_width)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mean_b, contrast_val = compute_lighting_stats(gray)

    # Compute primary sharpness signals first for potential FFT early-exit
    lap_var = round(compute_laplacian_variance(gray), 2)
    tenengrad_val = round(compute_tenengrad(gray), 2)

    # Early-exit for clearly sharp images: skip FFT, p95, AND new signals.
    # Saves ~30-40ms total. Tightened to lap_var > 1200 AND tenengrad > 55
    # to avoid masking borderline blur.
    if lap_var > 1200 and tenengrad_val > 55:
        fft_val = 0.01  # sentinel: FFT skipped (image clearly sharp)
        p95_val = round(lap_var * 2.5, 2)  # estimated p95
        fft_asym = 0.0  # sharp images have symmetric spectra
        wavelet_hh = round(lap_var * 0.03, 2)  # estimated (always > 30 threshold)
    else:
        gray_fft = _downsample(gray, 256)
        fft_val = round(compute_fft_hf_ratio(gray_fft), 6)
        p95_val = round(compute_laplacian_p95(gray), 2)
        # New signals computed on 256px for speed (reuses FFT downsample)
        fft_asym = round(compute_fft_asymmetry(gray_fft), 4)
        wavelet_hh = round(compute_wavelet_hh_energy(gray), 2)

    signals = {
        "laplacian_var": lap_var,
        "laplacian_p95": p95_val,
        "tenengrad": tenengrad_val,
        "fft_hf_ratio": fft_val,
        "fft_asymmetry": fft_asym,
        "wavelet_hh_energy": wavelet_hh,
        "glare_ratio": round(compute_glare_ratio(gray, hsv), 4),
        "mean_brightness": round(mean_b, 2),
        "contrast": round(contrast_val, 2),
        "border_content_ratio": round(compute_border_content_ratio(gray), 4),
    }

    verdict, reason = determine_verdict(signals, thresholds)

    t_end = time.perf_counter()
    latency_ms = round((t_end - t_start) * 1000, 2)

    return {
        "verdict": verdict,
        "reason": reason,
        "signals": signals,
        "latency_ms": latency_ms,
    }



# CLI

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python detector.py <image_path>")
        sys.exit(1)

    result = detect(sys.argv[1])
    print(json.dumps(result, indent=2))
