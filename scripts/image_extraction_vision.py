"""
Extract artwork image regions from scanned catalogue pages using Google Cloud Vision API.

Strategy:
  1. Object Localization — if any DIRECT_ART_TERMS label is found above MIN_OBJ_SCORE,
     crop that bounding box and save to output/.
  2. Everything else (no detections, subject-only labels, unrelated labels, API errors,
     crops that fail the quality filter) goes to output/fallback/ as a full-page image
     for human review.
Input:  data/dataset_B/*.jpg
Output: data/dataset_B_images/{stem}_crop_1.jpg, _crop_2.jpg, ...
        data/dataset_B_images/fallback/{stem}_full.jpg

Install dependencies:
  pip install google-cloud-vision pillow opencv-python

Set your API key:
  export GOOGLE_API_KEY="your_key_here"

Usage:
  python extract_dataset_B_images.py               # normal run
  python extract_dataset_B_images.py --survey       # collect all Object Localization labels, no files written
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
from google.cloud import vision

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_DIR    = Path("data/dataset_B")
OUTPUT_DIR   = Path("data/dataset_B_images")
FALLBACK_DIR = OUTPUT_DIR / "fallback"

MIN_CROP_PX   = 100    # minimum width/height of a saved crop in pixels
MIN_VARIANCE  = 500    # minimum pixel variance for a crop to be kept
MIN_OBJ_SCORE = 0.4    # minimum Object Localization confidence to accept a detection
REQUEST_DELAY  = 0.1    # seconds between API calls

# Labels that directly identify an artwork — crop when found above MIN_OBJ_SCORE
DIRECT_ART_TERMS = {
    "painting", "artwork", "illustration", "picture",
    "photograph", "poster", "print", "drawing", "sculpture",
    "picture frame",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_direct_art_label(label: str) -> bool:
    label_lower = label.lower()
    return any(term in label_lower for term in DIRECT_ART_TERMS)


def is_valid_crop(crop_bgr: np.ndarray) -> bool:
    h, w = crop_bgr.shape[:2]
    if w < MIN_CROP_PX or h < MIN_CROP_PX:
        return False
    return float(np.var(crop_bgr)) >= MIN_VARIANCE


def save_fallback(image_bgr: np.ndarray, stem: str) -> None:
    out_path = FALLBACK_DIR / f"{stem}_full.jpg"
    cv2.imwrite(str(out_path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])


def crop_and_save(image_bgr: np.ndarray, boxes: list, stem: str) -> int:
    """Crop each bounding box and save to OUTPUT_DIR. Returns number of crops saved."""
    h, w = image_bgr.shape[:2]
    saved = 0
    for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        cx1 = max(0, int(x1))
        cy1 = max(0, int(y1))
        cx2 = min(w, int(x2))
        cy2 = min(h, int(y2))
        crop = image_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue
        if not is_valid_crop(crop):
            continue
        out_path = OUTPUT_DIR / f"{stem}_crop_{idx}.jpg"
        cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1
    return saved


def localized_boxes_to_pixels(objects, img_w: int, img_h: int) -> list:
    """Return pixel (x1,y1,x2,y2) boxes for DIRECT_ART_TERMS detections above MIN_OBJ_SCORE."""
    boxes = []
    for obj in objects:
        if not is_direct_art_label(obj.name):
            continue
        if obj.score < MIN_OBJ_SCORE:
            continue
        verts = obj.bounding_poly.normalized_vertices
        xs = [v.x * img_w for v in verts]
        ys = [v.y * img_h for v in verts]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes


# ---------------------------------------------------------------------------
# Survey mode
# ---------------------------------------------------------------------------

def run_survey(client, image_files: list) -> None:
    from collections import defaultdict

    label_counts = defaultdict(int)
    label_scores = defaultdict(float)

    total = len(image_files)
    for i, img_path in enumerate(image_files, start=1):
        print(f"  [{i}/{total}] {img_path.name}", end="\r", flush=True)
        with open(img_path, "rb") as f:
            content = f.read()
        vision_image = vision.Image(content=content)
        try:
            response = client.object_localization(image=vision_image)
            time.sleep(REQUEST_DELAY)
        except Exception as exc:
            print(f"\n  [WARN] {img_path.name}: API error — {exc}")
            continue

        for obj in response.localized_object_annotations:
            label_counts[obj.name] += 1
            label_scores[obj.name] += obj.score

    print()  # newline after progress line

    if not label_counts:
        print("No labels returned across all images.")
        return

    col_label = max(len(lbl) for lbl in label_counts) + 2
    col_label = max(col_label, 20)

    header = f"{'Label':<{col_label}} {'Count':>7}  {'Avg confidence':>14}"
    print()
    print(f"Survey results — {total} images from {INPUT_DIR}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        avg_score = label_scores[label] / count
        print(f"{label:<{col_label}} {count:>7}  {avg_score:>14.3f}")

    print("=" * len(header))
    print(f"Unique labels: {len(label_counts)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract artwork crops from catalogue pages.")
    parser.add_argument(
        "--survey",
        action="store_true",
        help="Collect all Object Localization labels across the full dataset; print a frequency table. No files written.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY", "").strip().strip('\'""\u201c\u201d\u2018\u2019')
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")

    client = vision.ImageAnnotatorClient(
        client_options={"api_key": api_key},
        transport="rest",
    )

    image_files = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )

    if not image_files:
        print(f"No image files found in {INPUT_DIR}")
        return

    if args.survey:
        run_survey(client, image_files)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FALLBACK_DIR.mkdir(parents=True, exist_ok=True)

    pages_processed = 0
    pages_skipped   = 0
    crops_saved     = 0
    fall_no_detect  = 0   # no detections
    fall_other      = 0   # detections but no direct art label above threshold
    fall_filtered   = 0   # direct art found but all crops failed quality filter
    fall_api_error  = 0   # API error

    for img_path in image_files:
        stem = img_path.stem

        try:
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                raise ValueError("cv2.imread returned None")
        except Exception as exc:
            print(f"  [WARN] Cannot open {img_path.name}: {exc}")
            pages_skipped += 1
            continue

        pages_processed += 1
        img_h, img_w = image_bgr.shape[:2]

        with open(img_path, "rb") as f:
            content = f.read()
        vision_image = vision.Image(content=content)

        try:
            obj_response = client.object_localization(image=vision_image)
            time.sleep(REQUEST_DELAY)

            detections   = obj_response.localized_object_annotations
            direct_boxes = localized_boxes_to_pixels(detections, img_w, img_h)

            if not direct_boxes:
                save_fallback(image_bgr, stem)
                if not detections:
                    fall_no_detect += 1
                    print(f"  [FALL] {img_path.name} — no detections")
                else:
                    fall_other += 1
                    top_label = detections[0].name
                    print(f"  [FALL] {img_path.name} — no direct art label above threshold (top: {top_label})")
                continue

            saved = crop_and_save(image_bgr, direct_boxes, stem)

            if saved == 0:
                save_fallback(image_bgr, stem)
                fall_filtered += 1
                print(f"  [FALL] {img_path.name} — direct art detected but crops failed quality filter")
            else:
                crops_saved += saved
                print(f"  [OK]   {img_path.name} — {saved} crop(s) saved")

        except Exception as exc:
            print(f"  [ERR]  {img_path.name} — API error: {exc}; saving to fallback")
            save_fallback(image_bgr, stem)
            fall_api_error += 1

    total_fallbacks = fall_no_detect + fall_other + fall_filtered + fall_api_error

    print()
    print("=" * 50)
    print(f"Pages processed      : {pages_processed}")
    print(f"Pages skipped        : {pages_skipped}")
    print(f"Crops saved          : {crops_saved}")
    print(f"Fallback (total)     : {total_fallbacks}")
    print(f"  no detections        : {fall_no_detect}")
    print(f"  no direct art label  : {fall_other}")
    print(f"  crops filtered out   : {fall_filtered}")
    print(f"  API error            : {fall_api_error}")
    print(f"Output folder        : {OUTPUT_DIR.resolve()}")
    print(f"Fallback folder      : {FALLBACK_DIR.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    main()
