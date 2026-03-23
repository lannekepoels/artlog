"""
Extract artwork image regions from scanned catalogue pages using contour detection.

Input:  data/dataset_B/*.jpg
Output: data/dataset_B_images/{stem}_crop_1.jpg, {stem}_crop_2.jpg, ...
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

# Area bounds as a fraction of the total page area
MIN_AREA_FRAC = 0.01   # ignore contours smaller than 1% of the page
MAX_AREA_FRAC = 0.90   # ignore contours larger than 90% of the page (full-page layouts)

# Aspect ratio (width / height) — artwork is rarely extremely tall or wide
MIN_ASPECT = 0.3
MAX_ASPECT = 3.5

# Pixel variance threshold: regions below this are considered blank/white
MIN_VARIANCE = 200.0

# Canny edge detection thresholds
CANNY_LOW  = 50
CANNY_HIGH = 150

# Gaussian blur kernel size (must be odd)
BLUR_KERNEL = 5

# Dilation kernel size — closes small gaps in region borders so contours are complete
DILATE_KERNEL = 3
DILATE_ITERS  = 2

# Overlap suppression: if a smaller box overlaps a larger one by more than this
# fraction of the smaller box's area, discard the smaller box
OVERLAP_THRESHOLD = 0.6

# Minimum pixel dimension for a saved crop (safety guard)
MIN_CROP_PX = 50

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _intersection_area(a, b):
    """Area of intersection between two (x, y, w, h) rectangles."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(ax, bx)
    iy = max(ay, by)
    iw = min(ax + aw, bx + bw) - ix
    ih = min(ay + ah, by + bh) - iy
    return max(0, iw) * max(0, ih)


def suppress_overlapping(rects):
    """
    Remove smaller rectangles that are substantially contained within larger ones.
    Returns a filtered list sorted by area descending.
    """
    rects = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)
    kept = []
    for candidate in rects:
        ca = candidate[2] * candidate[3]
        discard = False
        for keeper in kept:
            inter = _intersection_area(candidate, keeper)
            if ca > 0 and inter / ca >= OVERLAP_THRESHOLD:
                discard = True
                break
        if not discard:
            kept.append(candidate)
    return kept


def detect_regions(img_bgr):
    """
    Return a list of (x, y, w, h) bounding rectangles for candidate artwork regions.
    """
    h, w = img_bgr.shape[:2]
    page_area = h * w

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (BLUR_KERNEL, BLUR_KERNEL), 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (DILATE_KERNEL, DILATE_KERNEL)
    )
    dilated = cv2.dilate(edges, kernel, iterations=DILATE_ITERS)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        x, y, rw, rh = cv2.boundingRect(cnt)
        area = rw * rh
        frac = area / page_area

        if frac < MIN_AREA_FRAC or frac > MAX_AREA_FRAC:
            continue

        aspect = rw / rh if rh > 0 else 0
        if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
            continue

        if rw < MIN_CROP_PX or rh < MIN_CROP_PX:
            continue

        candidates.append((x, y, rw, rh))

    return suppress_overlapping(candidates)


def is_blank(crop_bgr):
    """Return True if the crop has very low pixel variance (likely white/empty)."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.var(gray)) < MIN_VARIANCE


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    input_dir  = Path("data/dataset_B")
    output_dir = Path("data/dataset_B_images")
    output_dir.mkdir(exist_ok=True)

    jpg_files = sorted(input_dir.glob("*.jpg"))
    if not jpg_files:
        print(f"No JPG files found in {input_dir}")
        return

    files_processed = 0
    files_skipped   = 0
    total_regions   = 0

    for jpg_path in jpg_files:
        stem = jpg_path.stem
        img = cv2.imread(str(jpg_path))
        if img is None:
            print(f"  [WARN] Could not read {jpg_path.name} — skipping")
            files_skipped += 1
            continue

        files_processed += 1
        rects = detect_regions(img)

        valid_crops = []
        for rect in rects:
            x, y, rw, rh = rect
            crop = img[y:y+rh, x:x+rw]
            if crop.size == 0 or is_blank(crop):
                continue
            valid_crops.append((rect, crop))

        if not valid_crops:
            print(f"  [SKIP] {jpg_path.name} — no valid regions found")
            files_skipped += 1
            continue

        for idx, (rect, crop_bgr) in enumerate(valid_crops, start=1):
            out_name = f"{stem}_crop_{idx}.jpg"
            out_path = output_dir / out_name

            # Use Pillow to save (handles colour conversion cleanly)
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            Image.fromarray(crop_rgb).save(str(out_path), quality=95)

        total_regions += len(valid_crops)
        print(f"  [OK]   {jpg_path.name} — {len(valid_crops)} region(s) saved")

    print()
    print("=" * 50)
    print(f"Files processed : {files_processed}")
    print(f"Regions saved   : {total_regions}")
    print(f"Files skipped   : {files_skipped}")
    print(f"Output folder   : {output_dir.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    main()
