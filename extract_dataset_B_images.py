"""
Extract artwork image regions from scanned catalogue pages using Google Cloud Vision API.

Strategy:
  1. Object Localization — keep boxes whose label matches art-related terms
  2. Crop Hints fallback — used when no art objects are detected
  3. Full-page fallback — saved as {stem}_full.jpg if neither produces results or on API error

Input:  ./input/*.jpg
Output: ./output/{stem}_crop_1.jpg, {stem}_crop_2.jpg, ...
        ./output/fallbacks/{stem}_full.jpg  (fallback pages)

Install dependencies:
  pip install google-cloud-vision pillow opencv-python

Set your API key:
  export GOOGLE_API_KEY="your_key_here"

Usage:
  python extract_dataset_B_images.py               # normal run
  python extract_dataset_B_images.py --debug        # inspect 20 sample images, no files written
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
FALLBACK_DIR = OUTPUT_DIR / "fallbacks"

MIN_CROP_PX       = 100    # minimum width/height of a crop in pixels
MIN_VARIANCE      = 500    # minimum pixel variance (rejects blank/near-solid regions)
MIN_OBJ_SCORE     = 0.5    # minimum Object Localization confidence to accept a detection
REQUEST_DELAY     = 0.1    # seconds between API calls

DEBUG_SAMPLE_SIZE = 20     # number of images to inspect in --debug mode

ART_TERMS = {
    "painting", "artwork", "illustration", "picture",
    "photograph", "poster", "print", "drawing", "sculpture",
    "picture frame",
}

# If the top detection is one of these, skip crop hints and save the whole page
PERSON_CLOTHING_TERMS = {"person", "clothing"}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_art_label(label: str) -> bool:
    label_lower = label.lower()
    return any(term in label_lower for term in ART_TERMS)


def top_is_person_or_clothing(detections) -> bool:
    """Return True if the highest-confidence detection is Person or Clothing."""
    if not detections:
        return False
    return detections[0].name.lower() in PERSON_CLOTHING_TERMS


def is_valid_crop(crop_bgr: np.ndarray) -> bool:
    h, w = crop_bgr.shape[:2]
    if w < MIN_CROP_PX or h < MIN_CROP_PX:
        return False
    variance = float(np.var(crop_bgr))
    if variance < MIN_VARIANCE:
        return False
    return True


def save_full_page(image_bgr: np.ndarray, stem: str) -> None:
    out_path = FALLBACK_DIR / f"{stem}_full.jpg"
    cv2.imwrite(str(out_path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])


def crop_and_save(image_bgr: np.ndarray, boxes: list, stem: str) -> int:
    """Crop each bounding box and save. Returns number of crops saved."""
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
    """Convert Vision API normalized vertices to pixel (x1,y1,x2,y2) tuples."""
    boxes = []
    for obj in objects:
        if not is_art_label(obj.name):
            continue
        if obj.score < MIN_OBJ_SCORE:
            continue
        verts = obj.bounding_poly.normalized_vertices
        xs = [v.x * img_w for v in verts]
        ys = [v.y * img_h for v in verts]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes


def crop_hint_boxes_to_pixels(hints) -> list:
    """Convert crop hint bounding polys to pixel (x1,y1,x2,y2) tuples."""
    boxes = []
    for hint in hints:
        verts = hint.bounding_poly.vertices
        xs = [v.x for v in verts]
        ys = [v.y for v in verts]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes


# ---------------------------------------------------------------------------
# Debug mode
# ---------------------------------------------------------------------------

def run_debug(client, image_files: list) -> None:
    sample = image_files[:DEBUG_SAMPLE_SIZE]

    col_file  = 22
    col_label = 16
    col_score = 7
    col_art   = 7
    col_fb    = 12
    col_full  = 10

    header = (
        f"{'File':<{col_file}} "
        f"{'ObjLoc label':<{col_label}} "
        f"{'Score':>{col_score}} "
        f"{'Art?':<{col_art}} "
        f"{'Fallback':<{col_fb}} "
        f"{'Full page':<{col_full}}"
    )
    divider = "-" * len(header)

    print(f"\nDebug mode — sampling {len(sample)} images from {INPUT_DIR}\n")
    print(header)
    print(divider)

    for img_path in sample:
        with open(img_path, "rb") as f:
            content = f.read()
        vision_image = vision.Image(content=content)

        try:
            obj_response = client.object_localization(image=vision_image)
            time.sleep(REQUEST_DELAY)
        except Exception as exc:
            print(f"{img_path.name:<{col_file}} API ERROR: {exc}")
            continue

        detections = obj_response.localized_object_annotations
        all_labels = [(obj.name, obj.score) for obj in detections]
        art_above  = [(n, s) for n, s in all_labels if is_art_label(n) and s >= MIN_OBJ_SCORE]
        any_art    = any(is_art_label(n) for n, _ in all_labels)

        # Mirror the main-loop decision logic
        if art_above:
            fallback_col = ""
            full_col     = ""
        elif not detections:
            fallback_col = ""
            full_col     = "YES (no detections)"
        elif top_is_person_or_clothing(detections):
            fallback_col = ""
            full_col     = "YES (person/clothing)"
        elif any_art:
            fallback_col = "crop hints"
            full_col     = ""
        else:
            fallback_col = ""
            full_col     = "YES (no art labels)"

        if all_labels:
            first_name, first_score = all_labels[0]
            art_flag = "YES" if (is_art_label(first_name) and first_score >= MIN_OBJ_SCORE) else "no"
            print(
                f"{img_path.name:<{col_file}} "
                f"{first_name:<{col_label}} "
                f"{first_score:>{col_score}.3f} "
                f"{art_flag:<{col_art}} "
                f"{fallback_col:<{col_fb}} "
                f"{full_col:<{col_full}}"
            )
            for name, score in all_labels[1:]:
                art_flag = "YES" if (is_art_label(name) and score >= MIN_OBJ_SCORE) else "no"
                print(
                    f"{'':>{col_file}} "
                    f"{name:<{col_label}} "
                    f"{score:>{col_score}.3f} "
                    f"{art_flag:<{col_art}}"
                )
        else:
            print(
                f"{img_path.name:<{col_file}} "
                f"{'(none)':<{col_label}} "
                f"{'':>{col_score}} "
                f"{'':>{col_art}} "
                f"{fallback_col:<{col_fb}} "
                f"{full_col:<{col_full}}"
            )

        print(divider)

    print(f"\nART_TERMS filter: {sorted(ART_TERMS)}")
    print(f"MIN_OBJ_SCORE   : {MIN_OBJ_SCORE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract artwork crops from catalogue pages.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help=f"Print detection table for {DEBUG_SAMPLE_SIZE} sample images; no files written.",
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

    if args.debug:
        run_debug(client, image_files)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FALLBACK_DIR.mkdir(parents=True, exist_ok=True)

    pages_processed   = 0
    pages_skipped     = 0
    crops_obj_loc     = 0   # crops from Object Localization
    crops_crop_hints  = 0   # crops from Crop Hints fallback
    full_no_detection = 0   # full-page: no regions detected
    full_filtered     = 0   # full-page: all crops filtered out
    full_api_error    = 0   # full-page: API error

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

        # Read raw bytes for the Vision API
        with open(img_path, "rb") as f:
            content = f.read()
        vision_image = vision.Image(content=content)

        try:
            # Object Localization
            obj_response = client.object_localization(image=vision_image)
            time.sleep(REQUEST_DELAY)

            detections = obj_response.localized_object_annotations
            art_boxes  = localized_boxes_to_pixels(detections, img_w, img_h)
            any_art    = any(is_art_label(obj.name) for obj in detections)

            if art_boxes:
                # Art detected above confidence threshold — crop directly
                saved  = crop_and_save(image_bgr, art_boxes, stem)
                source = "object localization"
            elif not detections:
                # Nothing detected at all — full page, no crop hints
                save_full_page(image_bgr, stem)
                full_no_detection += 1
                print(f"  [FULL] {img_path.name} — no detections at all")
                continue
            elif top_is_person_or_clothing(detections):
                # Page is dominated by people/clothing — full page, no crop hints
                save_full_page(image_bgr, stem)
                full_no_detection += 1
                print(f"  [FULL] {img_path.name} — top detection is person/clothing ({detections[0].name})")
                continue
            elif any_art:
                # Art labels exist but below confidence threshold — use crop hints
                crop_response = client.crop_hints(image=vision_image)
                time.sleep(REQUEST_DELAY)
                hint_boxes = crop_hint_boxes_to_pixels(
                    crop_response.crop_hints_annotation.crop_hints
                )
                if hint_boxes:
                    saved  = crop_and_save(image_bgr, hint_boxes, stem)
                    source = "crop hints"
                else:
                    save_full_page(image_bgr, stem)
                    full_no_detection += 1
                    print(f"  [FULL] {img_path.name} — art below threshold, no crop hints returned")
                    continue
            else:
                # Detections exist but none are art-related — full page
                save_full_page(image_bgr, stem)
                full_no_detection += 1
                print(f"  [FULL] {img_path.name} — no art-related detections")
                continue

            if saved == 0:
                save_full_page(image_bgr, stem)
                full_filtered += 1
                print(f"  [FULL] {img_path.name} — all crops filtered out ({source})")
            else:
                if source == "object localization":
                    crops_obj_loc += saved
                else:
                    crops_crop_hints += saved
                print(f"  [OK]   {img_path.name} — {saved} crop(s) saved ({source})")

        except Exception as exc:
            print(f"  [ERR]  {img_path.name} — API error: {exc}; saving full page")
            save_full_page(image_bgr, stem)
            full_api_error += 1

    total_crops     = crops_obj_loc + crops_crop_hints
    total_fallbacks = full_no_detection + full_filtered + full_api_error

    print()
    print("=" * 50)
    print(f"Pages processed        : {pages_processed}")
    print(f"Pages skipped          : {pages_skipped}")
    print(f"Crops saved (total)    : {total_crops}")
    print(f"  from object localization : {crops_obj_loc}")
    print(f"  from crop hints          : {crops_crop_hints}")
    print(f"Full-page fallbacks    : {total_fallbacks}")
    print(f"  no detection             : {full_no_detection}")
    print(f"  all crops filtered       : {full_filtered}")
    print(f"  API error                : {full_api_error}")
    print(f"Output folder          : {OUTPUT_DIR.resolve()}")
    print(f"Fallbacks folder       : {FALLBACK_DIR.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    main()
