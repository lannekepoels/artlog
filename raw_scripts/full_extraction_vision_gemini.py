"""
================================================================================
RKD EXTRACTION PIPELINE — GEMINI VERSION
================================================================================
What this script does:
  1. Takes a folder of scanned auction catalog images
  2. Uses Google Vision AI to:
       A) Detect blank pages and skip them
       B) Find and crop just the artwork portion of each image
  3. Sends each full page image to Google Gemini (vision LLM) to extract
     structured RKD metadata fields, including:
       - Artist, Title, Date, Dimensions, Shape, Object Name
       - Medium (verbatim from catalog text)
       - Genre (mapped to RKD controlled vocabulary)
       - Provenance
       - Signature/inscription and signature location
  4. Validates genre output against the official RKD genre list
  5. Exports a CSV ready for human review in Label Studio

--------------------------------------------------------------------------------
SETUP
--------------------------------------------------------------------------------
  pip install pillow pandas requests opencv-python pydantic google-genai
              google-cloud-vision

  Set your API keys in the terminal before running:
    export GOOGLE_API_KEY=your_key_here
    export GEMINI_API_KEY=your_key_here

  Then run:
    python rkd_vision_gemini_pipeline.py
================================================================================
"""

import os
import json
import mimetypes
import time
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from google.cloud import vision
from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FOLDER   = "/Users/lannekepoels/extraction-pipeline/data/dataset_B"
OUTPUT_FOLDER  = "/Users/lannekepoels/extraction-pipeline/data/dataset_B_output"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MAX_IMAGES     = 30   # set to None to process all images
DELAY_BETWEEN_REQUESTS = 5   # seconds between Gemini API calls
MIN_CROP_PX    = 100   # minimum width/height of a saved crop in pixels
MIN_VARIANCE   = 500   # minimum pixel variance for a crop to be kept
MIN_OBJ_SCORE  = 0.4   # minimum Object Localization confidence
REQUEST_DELAY  = 0.1   # seconds between API calls

DIRECT_ART_TERMS = {
    "painting", "artwork", "illustration", "picture",
    "photograph", "poster", "print", "drawing", "sculpture",
    "picture frame",
}


# ============================================================
# RKD CONTROLLED VOCABULARIES
# ============================================================

# Strict RKD-approved genre list.
# Note: 'still life' is NOT an RKD genre — those works map to 'genre' here.
GENRE_KEYWORDS = {
    'portrait':                  ['portrait', 'self-portrait', 'portret', 'bust', 'likeness',
                                  'zelfportret'],
    'landscape (genre)':         ['landscape', 'landschap', 'forest', 'meadow', 'polder',
                                  'dunes', 'countryside', 'trees', 'village', 'farm',
                                  'shepherd', 'pastoral', 'winter', 'autumn', 'wooded',
                                  'tree', 'field', 'mountain', 'hills', 'heath'],
    'marine (genre)':            ['harbour', 'harbor', 'boats', 'ship', 'sea', 'beach',
                                  'sailing', 'fishing', 'river', 'canal', 'coastal',
                                  'haven', 'schepen', 'zee', 'strand'],
    'cityscape':                 ['cityscape', 'city', 'street', 'town', 'amsterdam',
                                  'rotterdam', 'urban', 'buildings', 'stad', 'straat'],
    'figure':                    ['figure', 'nude', 'woman', 'man', 'child', 'people',
                                  'figures', 'bathing', 'seated', 'standing', 'reclining',
                                  'girl', 'boy', 'mother', 'family', 'person', 'persons',
                                  'exhibition', 'museum visit', 'naakt', 'figuur'],
    'animal painting (genre)':   ['cat', 'dog', 'horse', 'cow', 'animal', 'bird',
                                  'kittens', 'peacock', 'chickens', 'sheep', 'cows',
                                  'dieren', 'paard', 'hond', 'kat'],
    'interior view':             ['interior', 'inn', 'tavern', 'room', 'cafe', 'kitchen',
                                  'bar', 'restaurant', 'terras', 'interieur', 'kamer'],
    'church interior':           ['church interior', 'cathedral interior', 'kerkinterieur'],
    'history (genre)':           ['biblical', 'mythology', 'allegory', 'angel', 'saint',
                                  'mythological', 'historic scene', 'battle', 'bijbels'],
    'genre':                     ['market', 'fair', 'courting', 'dancing', 'reading',
                                  'gathering', 'retirement', 'bench', 'park',
                                  # still life subjects map here (not a separate RKD genre)
                                  'still life', 'stilleven', 'flowers', 'fruit', 'vase',
                                  'roses', 'bouquet', 'jug', 'pitcher', 'peaches', 'grapes',
                                  'bloemen', 'vaas', 'vruchten'],
    'abstraction':               ['abstract', 'compositie', 'untitled', 'geometric',
                                  'composition', 'non-figurative', 'abstracte'],
    'architecture (genre)':      ['architecture', 'ruins', 'facade', 'tower', 'gate',
                                  'building exterior', 'architectuur', 'ruïne'],
    'design (artistic concept)': ['design', 'decorative art', 'applied art', 'furniture',
                                  'ceramics', 'glassware', 'textile', 'pattern', 'ornament',
                                  'kunstnijverheid', 'decoratief'],
}

ALLOWED_GENRES = set(GENRE_KEYWORDS.keys())


def validate_and_map_genre(raw_genre: str) -> str:
    """Map a Gemini-returned genre string to one or more allowed RKD genre terms.

    First tries an exact match, then checks if any allowed genre name appears
    in the raw string, and finally falls back to keyword scoring against the
    entry text so that non-standard labels (e.g. 'still life', 'flowers') are
    correctly mapped to their RKD equivalents."""
    if not raw_genre:
        return ""
    raw_lower = raw_genre.lower()

    # Direct match
    if raw_lower in ALLOWED_GENRES:
        return raw_lower

    # Partial match — allowed genre name appears inside the raw string
    matched = [g for g in ALLOWED_GENRES if g in raw_lower]
    if matched:
        return " | ".join(matched)

    # Keyword fallback: score each allowed genre against the raw text
    scored = []
    for genre, keywords in GENRE_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in raw_lower)
        if hits:
            scored.append((genre, hits))
    scored.sort(key=lambda x: -x[1])
    if scored:
        return " | ".join(g for g, _ in scored[:3])

    return ""


# ============================================================
# RKD METADATA SCHEMA
# ============================================================

class LM_entry(BaseModel):
    Artwork_Number:       Optional[str] = None
    Artist:               str
    Date_Dutch:           Optional[str] = None
    Date_English:         Optional[str] = None
    Search_Margin_Begin:  Optional[str] = None
    Search_Margin_End:    Optional[str] = None
    Genre:                Optional[str] = None
    Object_Name:          Optional[str] = None
    Medium:               Optional[str] = None
    Shape:                Optional[str] = None
    Height:               Optional[str] = None
    Width:                Optional[str] = None
    Unit:                 Optional[str] = None
    Image_Number:         Optional[str] = None
    Title:                str
    Provenance:           Optional[str] = None
    Signature_Inscription: Optional[str] = None
    Signature_Location:   Optional[str] = None
    FullEntryText:        str


# ============================================================
# PART 1 — GOOGLE VISION API
# ============================================================

def is_direct_art_label(label: str) -> bool:
    return any(term in label.lower() for term in DIRECT_ART_TERMS)


def is_valid_crop(crop_bgr: np.ndarray) -> bool:
    h, w = crop_bgr.shape[:2]
    if w < MIN_CROP_PX or h < MIN_CROP_PX:
        return False
    return float(np.var(crop_bgr)) >= MIN_VARIANCE


def save_blank_page(img_path, blank_dir: str) -> str:
    """Save a copy of the page to blank_pages/ and return the saved path."""
    blank_path = os.path.join(blank_dir, img_path.stem + "_blank.jpg")
    image_bgr = cv2.imread(str(img_path))
    cv2.imwrite(blank_path, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return blank_path


def call_vision_api(client, image_path: str):
    """Send one image to Google Cloud Vision API (OCR + object detection)."""
    with open(image_path, "rb") as f:
        content = f.read()
    vision_image = vision.Image(content=content)
    request = vision.AnnotateImageRequest(
        image=vision_image,
        features=[
            vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION),
            vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION),
        ]
    )
    try:
        response = client.annotate_image(request)
        time.sleep(REQUEST_DELAY)
        return response
    except Exception as exc:
        print(f"   Warning: Vision API error — {exc}")
        return None


def extract_ocr_text(vision_response) -> str:
    """Return the full OCR text from a Vision API response."""
    if vision_response and vision_response.full_text_annotation:
        return vision_response.full_text_annotation.text
    return ""



# ============================================================
# PART 2 — ARTWORK REGION DETECTION & CROPPING
# ============================================================

def localized_boxes_to_pixels(vision_response, img_w: int, img_h: int) -> list:
    """Return pixel (x1,y1,x2,y2) boxes for direct art label detections above MIN_OBJ_SCORE."""
    if not vision_response:
        return []
    boxes = []
    for obj in vision_response.localized_object_annotations:
        if not is_direct_art_label(obj.name):
            continue
        if obj.score < MIN_OBJ_SCORE:
            continue
        verts = obj.bounding_poly.normalized_vertices
        xs = [v.x * img_w for v in verts]
        ys = [v.y * img_h for v in verts]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes


def crop_and_save(image_bgr: np.ndarray, boxes: list, stem: str, crops_dir: str) -> list:
    """Crop each bounding box and save. Returns list of saved crop paths."""
    h, w = image_bgr.shape[:2]
    saved_paths = []
    multi = len(boxes) > 1
    for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        cx1, cy1 = max(0, int(x1)), max(0, int(y1))
        cx2, cy2 = min(w, int(x2)), min(h, int(y2))
        crop = image_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0 or not is_valid_crop(crop):
            continue
        suffix = f"_crop_{idx}" if multi else "_crop"
        out_path = os.path.join(crops_dir, f"{stem}{suffix}.jpg")
        cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_paths.append(out_path)
    return saved_paths


# ============================================================
# PART 3 — GEMINI METADATA EXTRACTION
# ============================================================

GEMINI_PROMPT = """Extract all complete, numbered artwork entries from this art historical catalog image.
An entry must include the artist's attribution and all directly associated text lines until the next entry appears.

For each entry, extract these fields following RKD metadata standards:

Artwork_Number: The catalog number of the artwork (e.g. '1', '2').
Artist: Full name of the artist from header or inline text (e.g. 'door', 'van', 'dezelfden').
Date_Dutch: Creation date in Dutch format (e.g. '1887 gedateerd', 'ca. 1880').
Date_English: Creation date in English (e.g. 'dated 1887', 'c. 1880').
Search_Margin_Begin: Earliest possible year as a number only (e.g. '1880').
Search_Margin_End: Latest possible year as a number only (e.g. '1890').
Genre: The RKD genre of the artwork. You MUST choose only from the following allowed values (use ' | ' to combine multiple):
  portrait, landscape (genre), marine (genre), cityscape, figure, animal painting (genre),
  interior view, church interior, history (genre), genre, abstraction,
  architecture (genre), design (artistic concept).
  Important mapping rules:
  - 'still life', 'stilleven', flowers, fruit, vases, bouquets → use 'genre'
  - market scenes, fair, dancing, reading, courting, gathering → use 'genre'
  - Do NOT invent genre values outside this list.
Object_Name: Physical category of the work (e.g. 'painting', 'drawing', 'watercolour').
Medium: The exact medium and support as stated in the text (e.g. 'oil on canvas', 'olieverf op doek', 'watercolour on paper', 'aquarel op papier'). Copy the text verbatim.
Shape: Physical shape (e.g. 'liggende rechthoek', 'staande rechthoek', 'rond').
Height: Height dimension as a number only (e.g. '40').
Width: Width dimension as a number only (e.g. '50').
Unit: Unit of measurement (e.g. 'cm', 'mm').
Image_Number: Image reference number if mentioned.
Title: Main descriptive title. Replace 'dito', '"', 'als voren', 'idem', 'id.' by the antecedent title.
Provenance: Ownership history if present (e.g. text after 'Provenance:', 'Herkomst:', or named collectors/auction houses). Copy verbatim.
Signature_Inscription: How the work is signed or inscribed (e.g. 'signed', 'gesigneerd', 'monogram', 'signed and dated').
Signature_Location: Where the signature appears on the work. Use standardized English terms: 'lower right', 'lower left', 'upper right', 'upper left', 'lower center', 'upper center'. Map Dutch abbreviations: l.o./l.b. → 'lower left', r.o./r.b. → 'lower right'.
FullEntryText: The complete unedited raw text of the entry including all line breaks.

Leave fields empty if the information is not present in the image. DO NOT fabricate any information. Only extract what is clearly visible in the image."""


def extract_metadata_with_gemini(image_path: str, retries=3) -> list:
    """Send image to Gemini and return a list of extracted RKD field dicts."""
    import base64
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"

    client = genai.Client(api_key=GEMINI_API_KEY)

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=[
                    {"role": "user", "parts": [
                        {"inline_data": {"mime_type": mime_type, "data": image_data}},
                        {"text": GEMINI_PROMPT + "\n\nRespond with a JSON array only."},
                    ]}
                ],
                config={"response_mime_type": "application/json", "max_output_tokens": 65536},
            )
            return json.loads(response.text)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "503" in error_str:
                wait_time = 60 * (attempt + 1)
                print(f"   Rate limit hit, waiting {wait_time}s before retry ({attempt+1}/{retries})...")
                time.sleep(wait_time)
            else:
                print(f"   Gemini API error: {e}")
                return []

    print(f"   Failed after {retries} retries, skipping")
    return []


# ============================================================
# PART 5 — MAIN PIPELINE
# ============================================================

def run_pipeline():
    """Process all images and produce RKD-mapped CSV + crops."""

    missing = [name for name, val in [("GOOGLE_API_KEY", GOOGLE_API_KEY), ("GEMINI_API_KEY", GEMINI_API_KEY)] if not val]
    if missing:
        print(f"\nERROR: Missing environment variable(s): {', '.join(missing)}")
        print("Set them in your terminal before running, for example:")
        for name in missing:
            print(f"  export {name}=your_key_here")
        return

    if not os.path.isdir(INPUT_FOLDER):
        print(f"\nERROR: Input folder not found: {INPUT_FOLDER}")
        return

    vision_client = vision.ImageAnnotatorClient(
        client_options={"api_key": GOOGLE_API_KEY},
        transport="rest",
    )
    crops_dir     = os.path.join(OUTPUT_FOLDER, "cropped_artworks")
    uncropped_dir = os.path.join(OUTPUT_FOLDER, "uncropped_artworks")
    blank_dir     = os.path.join(OUTPUT_FOLDER, "blank_pages")
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(uncropped_dir, exist_ok=True)
    os.makedirs(blank_dir, exist_ok=True)

    supported_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    image_files = sorted([
        f for f in Path(INPUT_FOLDER).iterdir()
        if f.suffix.lower() in supported_extensions
    ])

    if not image_files:
        print(f"No image files found in: {INPUT_FOLDER}")
        return

    if MAX_IMAGES is not None:
        image_files = image_files[:MAX_IMAGES]

    print(f"\n{'=' * 60}")
    print("RKD AUCTION IMAGE PIPELINE")
    print(f"{'=' * 60}")
    print(f"Input folder:  {INPUT_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Images found:  {len(image_files)}")
    print(f"{'=' * 60}\n")

    records = []
    errors  = []

    for i, img_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] {img_path.name}")

        try:
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                raise ValueError("cv2.imread returned None")
            img_h, img_w = image_bgr.shape[:2]

            # --- Step 1: Vision API for OCR + cropping ---
            print("   -> Calling Google Vision API (OCR + cropping)...")
            vision_data = call_vision_api(vision_client, str(img_path))

            if vision_data is None:
                raise ValueError("Empty response from Vision API")

            ocr_text = extract_ocr_text(vision_data)
            boxes    = localized_boxes_to_pixels(vision_data, img_w, img_h)

            # --- Blank page detection ---
            # Skip pages with < 10 chars of text AND no detected artwork region.
            if len(ocr_text.strip()) < 10 and not boxes:
                print("   -> Blank/empty page detected (no text, no artwork); saving to blank_pages/ and skipping")
                save_blank_page(img_path, blank_dir)
                if i < len(image_files):
                    print()
                continue

            if not boxes:
                fallback_path = os.path.join(uncropped_dir, img_path.stem + "_full.jpg")
                cv2.imwrite(fallback_path, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                crop_paths = [fallback_path]
                print("   -> Could not detect artwork region; saved full image to uncropped_artworks/")
            else:
                crop_paths = crop_and_save(image_bgr, boxes, img_path.stem, crops_dir)
                if not crop_paths:
                    fallback_path = os.path.join(uncropped_dir, img_path.stem + "_full.jpg")
                    cv2.imwrite(fallback_path, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    crop_paths = [fallback_path]
                    print("   -> Crops failed quality filter; saved full image to uncropped_artworks/")
                else:
                    print(f"   -> {len(crop_paths)} crop(s) saved")

            # --- Step 2: Gemini for metadata extraction ---
            print("   -> Calling Gemini API (metadata extraction)...")
            entries = extract_metadata_with_gemini(str(img_path))

            if not entries:
                print("   -> Warning: no metadata returned by Gemini")
                entries = [{"Artist": "Anoniem", "Title": "", "FullEntryText": ""}]

            print(f"   -> Extracted {len(entries)} entry/entries")
            for entry in entries:
                rec = entry if isinstance(entry, dict) else entry.model_dump()
                # Validate and map genre to allowed RKD vocabulary
                rec["Genre"] = validate_and_map_genre(rec.get("Genre", "") or "")
                rec["Image_Filename"]       = img_path.name
                rec["Image_path_original"]  = str(img_path)
                rec["Image_path_crop"]      = crop_paths[0] if len(crop_paths) == 1 else "; ".join(crop_paths)
                records.append(rec)
                print(f"      Artist: {rec.get('Artist', '(not found)')}")
                print(f"      Title:  {rec.get('Title', '(not found)')}")
                print(f"      Genre:  {rec.get('Genre', '(not found)')}")

        except Exception as e:
            print(f"   -> ERROR: {e}")
            errors.append({"file": img_path.name, "error": str(e)})

        if i < len(image_files):
            print(f"   -> Waiting {DELAY_BETWEEN_REQUESTS}s...")
            time.sleep(DELAY_BETWEEN_REQUESTS)
        print()

    # --- Save CSV ---
    if not records:
        print("No records produced.")
        return

    output_csv = os.path.join(OUTPUT_FOLDER, "rkd_metadata.csv")

    col_order = [
        "Image_Filename", 
        "Artwork_Number", "Artist", "Title",
        "Date_Dutch", "Date_English", "Search_Margin_Begin", "Search_Margin_End",
        "Genre", "Object_Name", "Medium", "Shape",
        "Height", "Width", "Unit",
        "Provenance", "Signature_Inscription", "Signature_Location",
        "FullEntryText", "Image_path_original", "Image_path_crop",
    ]

    df = pd.DataFrame(records)
    existing_priority = [c for c in col_order if c in df.columns]
    other_cols        = [c for c in df.columns if c not in col_order]
    df = df[existing_priority + other_cols]
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    # --- Summary ---
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total records:        {len(records)}")
    print(f"Errors:               {len(errors)}")
    print(f"\nFiles saved:")
    print(f"  CSV:              {output_csv}")
    print(f"  Artwork crops:    {crops_dir}/")
    print(f"  Uncropped images: {uncropped_dir}/")
    print("\nNext step: import rkd_metadata.csv into Label Studio for review.")

    if errors:
        print(f"\nFiles with errors ({len(errors)}):")
        for e in errors:
            print(f"  {e['file']}: {e['error']}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_pipeline()
