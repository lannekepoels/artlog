"""
Extract figure/image regions from scanned catalogue pages using YOLOv8 DocLayNet.

Model:  yolov8x-doclaynet.pt  (pretrained on DocLayNet — includes "Picture" class)
Input:  data/dataset_B/*.jpg
Output: data/dataset_B_images/{stem}_crop_1.jpg, ...

Download the model weights before first run:
  https://huggingface.co/nickmuchi/yolov8-medium-finetuned-doclaynet/resolve/main/yolov8x-doclaynet.pt
  or:  wget https://huggingface.co/nickmuchi/yolov8-medium-finetuned-doclaynet/resolve/main/yolov8x-doclaynet.pt

Requires: pip install ultralytics Pillow
"""

from pathlib import Path

import numpy as np
from PIL import Image
from doclayout_yolo import YOLOv10 as YOLO

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

MODEL_PATH = "doclayout_yolo_docstructbench_imgsz1024.pt"

# Minimum YOLO confidence to accept a detection
CONFIDENCE_THRESHOLD = 0.3

# DocLayNet class names to treat as artwork regions (case-insensitive substring match)
# DocLayNet uses "Picture" — keep this broad in case a variant model uses "Figure"
TARGET_CLASSES = {"picture", "figure", "illustration", "image"}

# Pixel variance threshold — crops below this are blank/white and skipped
MIN_VARIANCE = 150.0

# Minimum pixel dimension on either side for a saved crop
MIN_CROP_PX = 80

INPUT_DIR  = Path("data/dataset_B")
OUTPUT_DIR = Path("data/dataset_B_images")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_target_class_ids(model) -> set:
    """
    Return the set of class IDs whose names overlap with TARGET_CLASSES.
    Prints all available classes on first run so you can verify / adjust TARGET_CLASSES.
    """
    names = model.names  # {0: 'Caption', 1: 'Footnote', ...}
    print("Available classes in model:")
    for idx, name in names.items():
        print(f"  {idx}: {name}")
    print()

    matched = {idx for idx, name in names.items()
               if any(t in name.lower() for t in TARGET_CLASSES)}

    if not matched:
        print("[WARN] No classes matched TARGET_CLASSES. Check the class names above "
              "and update TARGET_CLASSES in the script.")
    else:
        matched_names = [names[i] for i in sorted(matched)]
        print(f"Targeting class(es): {matched_names}\n")

    return matched


def is_blank(crop: Image.Image) -> bool:
    gray = np.array(crop.convert("L"), dtype=np.float32)
    return float(np.var(gray)) < MIN_VARIANCE


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    jpg_files = sorted(INPUT_DIR.glob("*.jpg"))
    if not jpg_files:
        print(f"No JPG files found in {INPUT_DIR}")
        return

    # Detect available device — Ultralytics handles this internally,
    # but we pass device explicitly so CPU-only machines work without warnings
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    print(f"Loading model: {MODEL_PATH}  (device: {device})")
    model = YOLO(MODEL_PATH)
    target_ids = get_target_class_ids(model)

    pages_processed = 0
    pages_skipped   = 0
    total_figures   = 0

    for jpg_path in jpg_files:
        stem = jpg_path.stem
        try:
            image = Image.open(jpg_path).convert("RGB")
        except Exception as exc:
            print(f"  [WARN] Cannot open {jpg_path.name}: {exc}")
            pages_skipped += 1
            continue

        pages_processed += 1

        results = model(
            str(jpg_path),
            conf=CONFIDENCE_THRESHOLD,
            device=device,
            verbose=False,
        )

        boxes = results[0].boxes
        valid_crops = []

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            if target_ids and cls_id not in target_ids:
                continue

            x1, y1, x2, y2 = (int(round(c)) for c in boxes.xyxy[i].tolist())

            # Clamp to image bounds
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(image.width,  x2)
            y2 = min(image.height, y2)

            w, h = x2 - x1, y2 - y1
            if w < MIN_CROP_PX or h < MIN_CROP_PX:
                continue

            crop = image.crop((x1, y1, x2, y2))
            if is_blank(crop):
                continue

            valid_crops.append(crop)

        if not valid_crops:
            print(f"  [SKIP] {jpg_path.name} — no figures detected")
            pages_skipped += 1
            continue

        for idx, crop in enumerate(valid_crops, start=1):
            out_path = OUTPUT_DIR / f"{stem}_crop_{idx}.jpg"
            crop.save(str(out_path), quality=95)

        total_figures += len(valid_crops)
        print(f"  [OK]   {jpg_path.name} — {len(valid_crops)} figure(s) saved")

    print()
    print("=" * 50)
    print(f"Pages processed : {pages_processed}")
    print(f"Figures saved   : {total_figures}")
    print(f"Pages skipped   : {pages_skipped}")
    print(f"Output folder   : {OUTPUT_DIR.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    main()
