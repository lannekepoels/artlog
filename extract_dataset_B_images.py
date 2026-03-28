"""
Extract figure/image regions from scanned catalogue pages using PPStructureV3 (PaddlePaddle).

Model:  PPStructureV3 layout analysis — table, seal, formula, chart recognition disabled
Input:  data/dataset_B/*.jpg  (or any folder passed as INPUT_DIR)
Output: data/dataset_B_images/{stem}_figure_1.jpg, ...

Install dependencies before first run:
  pip install paddlepaddle        # CPU build
  # or: pip install paddlepaddle-gpu   # GPU build
  pip install "paddlex[ocr]" opencv-python pillow
"""

from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

# Minimum detection score (0–1) to accept a box
SCORE_THRESHOLD = 0.5

# Minimum bounding-box area in pixels to keep a detection
MIN_AREA_PX = 5_000

# Margin (pixels) added on every side of a detected box before cropping
BOX_MARGIN = 10

# Maximum side length (px) of the image fed to the model.
# Smaller = less memory, faster, slightly less accurate.
# The original image is still used for cropping, so output quality is preserved.
MAX_MODEL_SIDE = 1500

INPUT_DIR  = Path("data/dataset_B")
OUTPUT_DIR = Path("data/dataset_B_images")

# Recognised image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Label used by PaddleX layout model for image/figure regions
FIGURE_LABEL = "image"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from paddleocr import PPStructureV3  # imported here so module loads without it

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )

    if not image_files:
        print(f"No image files found in {INPUT_DIR}")
        return

    print("Loading PPStructureV3 layout model …")
    engine = PPStructureV3(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_seal_recognition=False,
        use_table_recognition=False,
        use_formula_recognition=False,
        use_chart_recognition=False,
    )
    print("Model loaded.\n")

    pages_processed = 0
    pages_skipped   = 0
    total_figures   = 0

    for img_path in image_files:
        try:
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                raise ValueError("cv2.imread returned None")
        except Exception as exc:
            print(f"  [WARN] Cannot open {img_path.name}: {exc}")
            pages_skipped += 1
            continue

        pages_processed += 1
        h, w = image_bgr.shape[:2]

        # Downscale for the model to reduce memory use; crop from the original
        scale = min(1.0, MAX_MODEL_SIDE / max(h, w))
        if scale < 1.0:
            model_img = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
        else:
            model_img = image_bgr

        results = list(engine.predict(model_img))

        figure_blocks = []
        for res in results:
            layout = res.get("layout_det_res", {})
            for box in layout.get("boxes", []):
                label = str(box.get("label", "")).lower()
                if label != FIGURE_LABEL:
                    continue

                score = box.get("score", 1.0)
                if score < SCORE_THRESHOLD:
                    continue

                coord = box.get("coordinate", [])  # [x1, y1, x2, y2]
                if len(coord) != 4:
                    continue

                # Scale coordinates back to original image size
                x1, y1, x2, y2 = (c / scale for c in coord)

                area = max(0, x2 - x1) * max(0, y2 - y1)
                if area < MIN_AREA_PX:
                    continue

                figure_blocks.append((x1, y1, x2, y2))

        if not figure_blocks:
            print(f"  [SKIP] {img_path.name} — no figures detected")
            pages_skipped += 1
            continue

        stem = img_path.stem
        saved = 0

        for idx, (x1, y1, x2, y2) in enumerate(figure_blocks, start=1):
            cx1 = max(0, int(x1) - BOX_MARGIN)
            cy1 = max(0, int(y1) - BOX_MARGIN)
            cx2 = min(w, int(x2) + BOX_MARGIN)
            cy2 = min(h, int(y2) + BOX_MARGIN)

            crop = image_bgr[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            out_path = OUTPUT_DIR / f"{stem}_figure_{idx}.jpg"
            cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1

        total_figures += saved
        print(f"  [OK]   {img_path.name} — {saved} figure(s) saved")

    print()
    print("=" * 50)
    print(f"Pages processed : {pages_processed}")
    print(f"Figures saved   : {total_figures}")
    print(f"Pages skipped   : {pages_skipped}")
    print(f"Output folder   : {OUTPUT_DIR.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    main()
