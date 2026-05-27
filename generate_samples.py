"""
Generate Sample Test Images
============================
Creates synthetic test images covering each failure mode so you can
verify the detector pipeline works end-to-end before collecting
real photographs.

These are NOT a substitute for real handwritten captures — use them
only to validate code and then replace with your actual dataset.

Usage:
    python generate_samples.py --output_dir test_images/
"""

import os
import argparse
import csv
import numpy as np
import cv2


def create_base_document(width=1200, height=1600):
    """Create a synthetic handwritten document on white paper."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 245  # off-white paper

    # Add some paper texture
    noise = np.random.normal(0, 3, (height, width)).astype(np.int16)
    for c in range(3):
        img[:, :, c] = np.clip(img[:, :, c].astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Draw horizontal ruled lines
    for y in range(100, height - 100, 40):
        jitter = np.random.randint(-1, 2)
        cv2.line(img, (80, y + jitter), (width - 80, y + jitter),
                 (200, 200, 220), 1, cv2.LINE_AA)

    # Simulate handwritten text with random strokes
    np.random.seed(42)
    for y_base in range(120, height - 150, 40):
        x = 100
        while x < width - 150:
            # Simulate a "word" with connected strokes
            word_len = np.random.randint(3, 8)
            pts = []
            for j in range(word_len):
                px = x + j * np.random.randint(8, 16)
                py = y_base + np.random.randint(-8, 8)
                pts.append((px, py))
            if len(pts) >= 2:
                pts = np.array(pts, dtype=np.int32)
                thickness = np.random.choice([1, 2])
                # Dark blue/black ink color with variation
                color = (
                    np.random.randint(10, 50),
                    np.random.randint(10, 50),
                    np.random.randint(10, 50),
                )
                cv2.polylines(img, [pts], False, color, thickness, cv2.LINE_AA)
            x += word_len * 14 + np.random.randint(15, 40)

    return img


def make_usable(base_img):
    """Good quality capture — sharp, well-lit, properly framed."""
    return base_img.copy()


def make_motion_blur(base_img, kernel_size=25):
    """Apply directional motion blur."""
    img = base_img.copy()
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = np.ones(kernel_size)
    kernel /= kernel_size
    return cv2.filter2D(img, -1, kernel)


def make_defocus_blur(base_img, ksize=15):
    """Apply circular defocus blur."""
    img = base_img.copy()
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def make_heavy_defocus(base_img, ksize=31):
    """Heavy defocus — clearly unusable."""
    img = base_img.copy()
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def make_glare(base_img):
    """Add a glare spot to the image."""
    img = base_img.copy()
    h, w = img.shape[:2]
    # Create an elliptical glare spot
    cx, cy = w // 3, h // 3
    for c in range(3):
        Y, X = np.ogrid[:h, :w]
        dist = ((X - cx) / 200) ** 2 + ((Y - cy) / 150) ** 2
        glare_mask = np.exp(-dist * 0.5) * 200
        img[:, :, c] = np.clip(
            img[:, :, c].astype(np.float32) + glare_mask, 0, 255
        ).astype(np.uint8)
    return img


def make_dark(base_img):
    """Simulate insufficient lighting."""
    img = base_img.copy()
    return np.clip(img.astype(np.float32) * 0.2, 0, 255).astype(np.uint8)


def make_low_contrast(base_img):
    """Low contrast — dim but not completely dark."""
    img = base_img.copy()
    return np.clip(img.astype(np.float32) * 0.4 + 30, 0, 255).astype(np.uint8)


def make_partial_frame(base_img):
    """Simulate partial framing — page cut off."""
    img = base_img.copy()
    h, w = img.shape[:2]
    # Shift image so right side is cut off, fill with dark background
    shift = w // 3
    result = np.ones_like(img) * 40  # dark background (desk/table)
    result[:, :w - shift] = img[:, shift:]
    return result


def make_pencil_light(width=1200, height=1600):
    """
    Light pencil on white paper — should be classified as USABLE.
    This tests the pencil-on-paper edge case mentioned in the spec.
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 250

    # Very faint pencil strokes
    np.random.seed(99)
    for y_base in range(120, height - 150, 45):
        x = 100
        while x < width - 150:
            word_len = np.random.randint(3, 7)
            pts = []
            for j in range(word_len):
                px = x + j * np.random.randint(8, 14)
                py = y_base + np.random.randint(-5, 5)
                pts.append((px, py))
            if len(pts) >= 2:
                pts = np.array(pts, dtype=np.int32)
                # Very light gray — pencil
                color = (180, 180, 180)
                cv2.polylines(img, [pts], False, color, 1, cv2.LINE_AA)
            x += word_len * 13 + np.random.randint(20, 45)

    return img


def make_mild_blur(base_img, ksize=7):
    """Mild blur — borderline case."""
    return cv2.GaussianBlur(base_img.copy(), (ksize, ksize), 0)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic test images for detector development."
    )
    parser.add_argument("--output_dir", "-o", default="test_images",
                        help="Output directory for generated images.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    base = create_base_document()

    # Define samples: (filename, generator_func, ground_truth, failure_mode)
    samples = [
        # Usable
        ("usable_001.jpg", lambda: make_usable(base), "usable", ""),
        ("usable_002.jpg", lambda: make_usable(create_base_document()), "usable", ""),
        ("usable_pencil_003.jpg", lambda: make_pencil_light(), "usable", ""),
        ("usable_mild_blur_004.jpg", lambda: make_mild_blur(base, 3), "usable", ""),

        # Motion blur
        ("retake_motion_001.jpg", lambda: make_motion_blur(base, 25), "retake", "motion_blur"),
        ("retake_motion_002.jpg", lambda: make_motion_blur(base, 35), "retake", "motion_blur"),

        # Defocus
        ("retake_defocus_001.jpg", lambda: make_defocus_blur(base, 15), "retake", "defocus"),
        ("retake_defocus_002.jpg", lambda: make_heavy_defocus(base, 31), "retake", "defocus"),

        # Glare
        ("retake_glare_001.jpg", lambda: make_glare(base), "retake", "glare"),

        # Dark / lighting
        ("retake_dark_001.jpg", lambda: make_dark(base), "retake", "insufficient_lighting"),
        ("retake_lowcontrast_001.jpg", lambda: make_low_contrast(base), "retake", "insufficient_lighting"),

        # Framing
        ("retake_framing_001.jpg", lambda: make_partial_frame(base), "retake", "partial_framing"),

        # Borderline
        ("borderline_blur_001.jpg", lambda: make_mild_blur(base, 9), "borderline", "defocus"),
    ]

    labels = []
    for filename, gen_func, gt, mode in samples:
        img = gen_func()
        path = os.path.join(args.output_dir, filename)
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        labels.append({"filename": filename, "ground_truth": gt, "failure_mode": mode})
        print(f"  Created {filename} (gt={gt}, mode={mode or 'n/a'})")

    # Write labels CSV
    labels_path = os.path.join(args.output_dir, "..", "labels.csv")
    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "ground_truth", "failure_mode"])
        writer.writeheader()
        writer.writerows(labels)

    print(f"\n{len(samples)} images created in {args.output_dir}/")
    print(f"Labels written to {labels_path}")
    print("\nNOTE: These synthetic images are for pipeline validation only.")
    print("Replace with real phone captures for your final submission.")


if __name__ == "__main__":
    main()
