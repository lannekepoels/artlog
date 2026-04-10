"""
================================================================================
RKD AUCTION IMAGE PIPELINE
================================================================================
What this script does:
  1. Takes a folder of scanned auction images
  2. Uses Google Vision AI to:
       A) Find and crop just the artwork portion of each image
       B) Read all the text on the page (OCR)
  3. Maps the extracted text onto RKD database fields
  4. Exports a CSV ready for human review in Label Studio

--------------------------------------------------------------------------------
SETUP
--------------------------------------------------------------------------------
  pip install pillow pandas requests

  Then run:
    python rkd_pipeline.py
================================================================================
"""

import os
import re

import time
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from google.cloud import vision


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FOLDER   = "/Users/macbookpro/Downloads/ERP-sample_assets"
OUTPUT_FOLDER  = "/Users/macbookpro/Downloads/ERP-output-v2"
GOOGLE_API_KEY = "AIzaSyC7IP5ICSg6cDvilNiwnkoxQp9tOKHF0zo"
MAX_IMAGES     = 100   # set to None to process all images
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
# Vision labels (detected objects) are appended to the search text at runtime.
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

OBJECT_NAME_MAP = [
    (['oil on canvas', 'oil on panel', 'oil on board', 'oil on paper',
      'oil on copper', 'oil stick', 'olieverf'],                         'painting'),
    (['watercolour', 'watercolor', 'aquarel', 'gouache', 'pencil on paper',
      'chalk on paper', 'charcoal', 'ink on paper', 'pastel on paper',
      'coloured chalk', 'black chalk', 'pen and ink', 'pen en inkt',
      'pencil', 'drypoint', 'indian ink', 'sepia', 'oost-indische inkt',
      'krijt', 'gemengde techniek'],                                      'drawing'),
    (['etching', 'engraving', 'lithograph', 'colour lithograph'],         'print'),
    (['bronze', 'paper-mache', 'papier-mache'],                           'sculpture'),
    (['mixed media'],                                    'needs human review'),
    (['acrylic on canvas', 'acrylic'],                                    'painting'),
]

# Ordered list of (regex_pattern, normalized_RKD_term).
# First match wins. If nothing matches, falls back to "unknown" per RKD manual.
MEDIUM_NORMALIZATION = [
    # Pencil + watercolour combos (must come before single-medium rules)
    (r'pencil.{0,25}watercolou?r.{0,25}paper',        'pencil and watercolour on paper'),
    (r'watercolou?r.{0,25}pencil.{0,25}paper',        'pencil and watercolour on paper'),
    (r'potlood.{0,25}aquarel.{0,25}papier',           'potlood en aquarel op papier'),
    (r'aquarel.{0,25}potlood.{0,25}papier',           'potlood en aquarel op papier'),
    # Pen & ink
    (r'pen\b.{0,15}\binkt\b.{0,25}\bpapier\b',        'pen en inkt op papier'),
    (r'pen\b.{0,15}\bink\b.{0,25}\bpaper\b',          'pen and ink on paper'),
    # Dutch oil
    (r'olieverf\b.{0,30}\bdoek\b',                    'olieverf op doek'),
    (r'olieverf\b.{0,30}\bpaneel\b',                  'olieverf op paneel'),
    (r'olieverf\b.{0,30}\bkarton\b',                  'olieverf op karton'),
    (r'olieverf\b.{0,30}\bpapier\b',                  'olieverf op papier'),
    # English oil
    (r'oil\b.{0,20}\bcanvas\b',                        'oil paint on canvas'),
    (r'oil\b.{0,20}\bpanel\b',                         'oil paint on panel'),
    (r'oil\b.{0,20}\bboard\b',                         'oil paint on board'),
    (r'oil\b.{0,20}\bcopper\b',                        'oil paint on copper'),
    (r'oil\b.{0,20}\bpaper\b',                         'oil paint on paper'),
    # Watercolour
    (r'aquarel\b.{0,30}\bpapier\b',                   'aquarel op papier'),
    (r'watercolou?r\b.{0,30}\bpaper\b',               'watercolour on paper'),
    (r'watercolou?r\b.{0,30}\bcanvas\b',              'watercolour on canvas'),
    # Gouache
    (r'gouache\b.{0,20}\bpapier\b',                   'gouache op papier'),
    (r'gouache\b.{0,20}\bpaper\b',                    'gouache on paper'),
    # Pastel
    (r'pastel\b.{0,20}\bpapier\b',                    'pastel op papier'),
    (r'pastel\b.{0,20}\bpaper\b',                     'pastel on paper'),
    # Chalk / krijt
    (r'krijt\b.{0,20}\bpapier\b',                     'krijt op papier'),
    (r'black chalk\b.{0,20}\bpaper\b',                'black chalk on paper'),
    (r'colou?red chalk\b.{0,20}\bpaper\b',            'coloured chalk on paper'),
    (r'charcoal\b.{0,20}\bpaper\b',                   'charcoal on paper'),
    # Pencil alone
    (r'potlood\b.{0,20}\bpapier\b',                   'potlood op papier'),
    (r'pencil\b.{0,20}\bpaper\b',                     'pencil on paper'),
    # Ink
    (r'inkt\b.{0,20}\bpapier\b',                      'inkt op papier'),
    (r'ink\b.{0,20}\bpaper\b',                        'ink on paper'),
    # Mixed / gemengd
    (r'gemengde techniek\b.{0,20}\bpapier\b',         'gemengde techniek op papier'),
    (r'gemengde techniek\b.{0,20}\bdoek\b',           'gemengde techniek op doek'),
    (r'mixed media\b.{0,20}\bpaper\b',                'mixed media on paper'),
    (r'mixed media\b.{0,20}\bcanvas\b',               'mixed media on canvas'),
    # Acrylic
    (r'acryl\b.{0,20}\bdoek\b',                       'acrylverf op doek'),
    (r'acrylic\b.{0,20}\bcanvas\b',                   'acrylic on canvas'),
    # Prints
    (r'\bets\b',                                       'ets'),
    (r'\betching\b',                                   'etching'),
    (r'\blithograph\b',                                'lithograph'),
    (r'\bengraving\b',                                 'engraving'),
    (r'\bdrypoint\b',                                  'drypoint'),
]


def normalize_medium(raw_medium: str) -> str:
    """Map raw OCR medium text to a standardized RKD thesaurus term.
    Returns 'unknown' if no match found, per RKD manual guidance."""
    if not raw_medium:
        return 'unknown'
    text_lower = raw_medium.lower()
    for pattern, normalized in MEDIUM_NORMALIZATION:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return normalized
    return 'unknown'


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


def extract_full_text(vision_response) -> str:
    """Pull all OCR text from the Vision API SDK response."""
    if vision_response and vision_response.full_text_annotation:
        return vision_response.full_text_annotation.text
    return ""


def extract_vision_labels(vision_response) -> list:
    """Return detected object label names from Vision API (used to assist genre detection)."""
    if not vision_response:
        return []
    return [obj.name.lower() for obj in vision_response.localized_object_annotations
            if obj.score >= MIN_OBJ_SCORE]


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
# PART 3 — TEXT TO RKD FIELD MAPPING
# ============================================================

def map_text_to_rkd_fields(raw_text: str, image_filename: str,
                            vision_labels: list = None) -> dict:
    text  = raw_text.strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    fields = {
        "Artist":                  "",
        "Title":                   "",
        "Date Dutch":              "",
        "Date English":            "",
        "Search margin: begin":    "",
        "Search margin: end":      "",
        "Object name":             "",
        "Medium (raw)":            "",
        "Medium (normalized)":     "",
        "Genre":                   "",
        "Height":                  "",
        "Width":                   "",
        "Unit":                    "cm",
        "Shape":                   "",
        "Provenance":              "",
        "Signature/inscription":   "",
        "Signature location":      "",
        "Lot No":                  "",
        "Inventory number":        "",
        "Source image":            image_filename,
        "Raw OCR text":            text,
        "Needs review":            "",
    }

    inv = re.search(r'\b(\d{4,5}/[\w\.]+)', text)
    if inv:
        fields["Inventory number"] = inv.group(1)

    lot = re.search(r'\blot\s*[:#]?\s*(\d+)\b', text, re.IGNORECASE)
    if not lot:
        lot = re.search(r'^\s*(\d{2,4})\s*$', text, re.MULTILINE)
    if lot:
        fields["Lot No"] = lot.group(1)

    chr_artist = re.search(
        r'([A-Z][A-Z\s\-\.]+?)\s*\((?:DUTCH|FLEMISH|FRENCH|GERMAN|BELGIAN'
        r'|BRITISH|AMERICAN),?\s*(\d{4})\s*[-–]\s*(\d{4})\)',
        text
    )
    if chr_artist:
        fields["Artist"] = chr_artist.group(1).strip().title()

    if not fields["Artist"]:
        dut = re.search(
            r'((?:[A-Z][a-z]+\s+){1,4}[A-Z][a-z]+)\n\([A-Za-z]+\s+\d{4}',
            text
        )
        if dut:
            fields["Artist"] = dut.group(1).strip()

    if not fields["Artist"] and fields["Inventory number"]:
        idx = text.find(fields["Inventory number"])
        if idx != -1:
            after = text[idx + len(fields["Inventory number"]):].strip()
            for candidate in after.splitlines()[:6]:
                candidate = candidate.strip()
                if (len(candidate) >= 5
                        and len(candidate.split()) >= 2
                        and not any(c.isdigit() for c in candidate)
                        and '@' not in candidate
                        and 'www.' not in candidate.lower()
                        and candidate.lower() != candidate):
                    fields["Artist"] = candidate
                    break

    if fields["Artist"]:
        artist_last = fields["Artist"].split()[-1]
        for i, line in enumerate(lines):
            if artist_last in line:
                for j in range(i + 1, min(i + 6, len(lines))):
                    cand = lines[j].strip()
                    if re.match(r'^\(', cand) and re.search(r'\d{4}', cand):
                        continue
                    if re.match(r'^(met|with)\s+', cand, re.IGNORECASE):
                        continue
                    if re.search(
                            r'\d{2}|\bop\s+papier\b|cm\b|mm\b|olieverf|aquarel'
                            r'|pencil|oil\b|signed|gesigneerd|monogram',
                            cand, re.IGNORECASE):
                        continue
                    if 2 <= len(cand.split()) <= 8:
                        fields["Title"] = cand
                        break
                break

    if not fields["Title"]:
        ch_t = re.search(
            r'\d{4}[-–]\d{4}\)\n(.+?)(?:\n|signed|watercolour|oil|pencil)',
            text, re.IGNORECASE
        )
        if ch_t:
            fields["Title"] = ch_t.group(1).strip()

    medium_raw = extract_medium_raw(text)
    fields["Medium (raw)"]        = medium_raw
    fields["Medium (normalized)"] = normalize_medium(medium_raw)
    fields["Object name"]         = detect_object_name(medium_raw) if medium_raw else "needs human review"

    dims = extract_dimensions(text)
    if dims["height"]:
        fields["Height"] = dims["height"]
        fields["Width"]  = dims["width"]
        fields["Unit"]   = dims["unit"]
        fields["Shape"]  = dims["shape"]

    date_info = extract_date_from_ocr(text)
    fields.update({
        "Date Dutch":           date_info["date_dutch"],
        "Date English":         date_info["date_english"],
        "Search margin: begin": date_info["margin_begin"],
        "Search margin: end":   date_info["margin_end"],
    })

    sig = extract_signature_info(text)
    fields["Signature/inscription"] = sig["inscription"]
    fields["Signature location"]    = sig["location"]

    prov = extract_provenance(text)
    fields["Provenance"] = prov[0] if prov else ""

    # Combine title + OCR text + Vision object labels for richer genre detection
    vision_label_str = " ".join(vision_labels) if vision_labels else ""
    genre_text = (fields["Title"] + " " + text + " " + vision_label_str).lower()
    matched_genres = []
    for genre, keywords in GENRE_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in genre_text)
        if hits:
            matched_genres.append((genre, hits))
    matched_genres.sort(key=lambda x: -x[1])
    fields["Genre"] = " | ".join(g for g, _ in matched_genres[:3]) if matched_genres else "undetermined"

    if not fields["Artist"]:
        fields["Artist"] = "Anoniem"

    missing = []
    if fields["Artist"] == "Anoniem": missing.append("artist (Anoniem)")
    if not fields["Title"]:           missing.append("title")
    if not fields["Height"]:          missing.append("dimensions")
    if fields["Object name"] in ("needs human review", ""):
        missing.append("medium")
    fields["Needs review"] = ", ".join(missing) if missing else "ok"

    return fields


# ============================================================
# PART 4 — HELPER EXTRACTION FUNCTIONS
# ============================================================

def extract_medium_raw(text: str) -> str:
    m = re.search(
        r'((?:olieverf|aquarel|pen|inkt|krijt|potlood|pastel|houtskool|gouache)'
        r'(?:[\s,]+(?:en\s+)?[\w\-]+)*\s+op\s+(?:papier|doek|paneel|karton))',
        text, re.IGNORECASE
    )
    if m:
        return m.group(1)

    m = re.search(
        r'((?:oil|watercolour|watercolor|gouache|pencil|chalk|charcoal|'
        r'ink|pastel|acrylic|etching|engraving|lithograph)'
        r'(?:\s+and\s+[\w]+)*\s+on\s+(?:paper|canvas|board|panel|copper))',
        text, re.IGNORECASE
    )
    if m:
        return m.group(1)

    for keywords, _ in OBJECT_NAME_MAP:
        for kw in keywords:
            if kw in text.lower():
                idx = text.lower().find(kw)
                return text[max(0, idx - 3): idx + 50].strip()

    return ""


def detect_object_name(medium_text: str) -> str:
    desc_lower = medium_text.lower()
    for keywords, obj_name in OBJECT_NAME_MAP:
        if any(kw in desc_lower for kw in keywords):
            return obj_name
    return "needs human review"


def extract_dimensions(text: str) -> dict:
    num     = r'(\d{1,4}(?:[,\.]\d{1,2})?)'
    sep     = r'\s*[x×]\s*'
    pattern = re.compile(num + sep + num + r'(?:' + sep + num + r')?\s*(cm|mm)?',
                         re.IGNORECASE)
    for m in pattern.finditer(text):
        try:
            h = float(m.group(1).replace(',', '.'))
            w = float(m.group(2).replace(',', '.'))
            if h < 2 or w < 2 or h > 500 or w > 500:
                continue
            unit  = (m.group(4) or 'cm').lower()
            shape = ('liggende rechthoek' if w > h else
                     'staande rechthoek'  if h > w else 'vierkant')
            return {'height': h, 'width': w, 'unit': unit, 'shape': shape}
        except (ValueError, AttributeError):
            continue
    return {'height': None, 'width': None, 'unit': 'cm', 'shape': 'onbekend'}


def extract_date_from_ocr(text: str) -> dict:
    m = re.search(
        r"(?:gedateerd|dated)\s+['\"]?\s*(['\"]?\d{2,4})['\"]?",
        text, re.IGNORECASE
    )
    if m:
        raw = re.sub(r"['\"]", "", m.group(1)).strip()
        try:
            yr   = int(raw)
            year = (1900 + yr if yr > 25 else 2000 + yr) if yr < 100 else yr
            return {
                'date_dutch':   f'{year} gedateerd',
                'date_english': f'dated {year}',
                'margin_begin': year,
                'margin_end':   year,
            }
        except ValueError:
            pass

    months = r'jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|mrt|mei|okt'
    for m in re.finditer(r'\b(1[5-9]\d{2}|20[0-2]\d)\b', text):
        context = text[max(0, m.start() - 30): m.end() + 30].lower()
        if not re.search(months, context):
            return {
                'date_dutch':   f'{m.group(1)} gedateerd',
                'date_english': f'dated {m.group(1)}',
                'margin_begin': int(m.group(1)),
                'margin_end':   int(m.group(1)),
            }

    return {'date_dutch': '', 'date_english': '', 'margin_begin': '', 'margin_end': ''}


def extract_signature_info(text: str) -> dict:
    result = {'inscription': '', 'location': ''}

    # Dutch abbreviations: l.o. = lower left, r.o. = lower right
    loc_match = re.search(
        r'(?:gesigneerd|signed)\s+([lrLR]\.[obOB]\.)', text, re.IGNORECASE
    )
    if loc_match:
        loc_raw = loc_match.group(1).lower()
        loc_map = {'l.o.': 'lower left', 'r.o.': 'lower right',
                   'l.b.': 'lower left', 'r.b.': 'lower right'}
        result['location'] = loc_map.get(loc_raw, loc_raw)

    # English parenthetical or inline: "(lower right)", "lower left", etc.
    if not result['location']:
        eng_loc = re.search(
            r'\(?(lower|upper)\s+(right|left)\)?',
            text, re.IGNORECASE
        )
        if eng_loc:
            result['location'] = (
                f"{eng_loc.group(1).lower()} {eng_loc.group(2).lower()}"
            )

    if re.search(r'\b(monogram|monogrammed|met monogram)\b', text, re.IGNORECASE):
        result['inscription'] = 'monogram'
    elif re.search(r'\bsigned\b|\bgesigneerd\b', text, re.IGNORECASE):
        result['inscription'] = 'signed'

    return result


def extract_provenance(text: str) -> list:
    m = re.search(r'\bprovenance\s*:\s*', text, re.IGNORECASE)
    if m:
        prov_text = text[m.end():]
        stop = re.search(
            r'\b(exhibited|literature|bibliography)\s*:', prov_text, re.IGNORECASE
        )
        if stop:
            prov_text = prov_text[:stop.start()]
        entries = re.split(r'\s*-\s*(?=[A-Z])', prov_text.strip())
        return [e.strip() for e in entries if e.strip()] or ['']

    if 'simonis' in text.lower() and 'buunk' in text.lower():
        return ['Simonis & Buunk Kunsthandel, Ede']

    return ['']


# ============================================================
# PART 5 — MAIN PIPELINE
# ============================================================

def run_pipeline():
    if not os.path.isdir(INPUT_FOLDER):
        print(f"\nERROR: Input folder not found: {INPUT_FOLDER}")
        return

    client = vision.ImageAnnotatorClient(
        client_options={"api_key": GOOGLE_API_KEY},
        transport="rest",
    )

    crops_dir     = os.path.join(OUTPUT_FOLDER, "cropped_artworks")
    uncropped_dir = os.path.join(OUTPUT_FOLDER, "uncropped_artworks")
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(uncropped_dir, exist_ok=True)

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

            print("   -> Calling Google Vision API...")
            vision_data = call_vision_api(client, str(img_path))

            if vision_data is None:
                raise ValueError("Empty response from Vision API")

            ocr_text      = extract_full_text(vision_data)
            vision_labels = extract_vision_labels(vision_data)
            n_chars       = len(ocr_text)
            print(f"   -> OCR: {n_chars} characters extracted")

            boxes = localized_boxes_to_pixels(vision_data, img_w, img_h)

            # Skip blank pages — no usable text AND no artwork detected
            if n_chars < 50 and not boxes:
                print("   -> Blank page detected (no text, no artwork) — skipping")
                errors.append({"file": img_path.name, "error": "blank page — skipped"})
                print()
                continue

            if n_chars < 20:
                print("   -> Warning: very little text found")

            if not boxes:
                fallback_path = os.path.join(uncropped_dir, img_path.stem + "_full.jpg")
                cv2.imwrite(fallback_path, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                crop_paths = [fallback_path]
                print("   -> No art region detected; saved full image to uncropped_artworks/")
            else:
                crop_paths = crop_and_save(image_bgr, boxes, img_path.stem, crops_dir)
                if not crop_paths:
                    fallback_path = os.path.join(uncropped_dir, img_path.stem + "_full.jpg")
                    cv2.imwrite(fallback_path, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    crop_paths = [fallback_path]
                    print("   -> Crops failed quality filter; saved full image to uncropped_artworks/")

            multi = len(crop_paths) > 1
            if multi:
                print(f"   -> {len(crop_paths)} artworks detected on this page")

            for art_idx, crop_path in enumerate(crop_paths):
                rkd_fields = map_text_to_rkd_fields(ocr_text, img_path.name,
                                                     vision_labels=vision_labels)
                rkd_fields["Image path (original)"] = str(img_path)
                rkd_fields["Image path (crop)"]     = crop_path

                if multi:
                    rkd_fields["Source image"] = (
                        f"{img_path.name} [artwork {art_idx + 1} of {len(crop_paths)}]"
                    )
                    nr = rkd_fields["Needs review"]
                    rkd_fields["Needs review"] = (
                        nr + ", multi-artwork page" if nr != "ok" else "multi-artwork page"
                    )

                records.append(rkd_fields)

                status = "NEEDS REVIEW" if rkd_fields["Needs review"] != "ok" else "OK"
                label  = f" [artwork {art_idx + 1}]" if multi else ""
                print(f"   -> Status{label}:  {status}")
                if rkd_fields["Needs review"] != "ok":
                    print(f"      Missing: {rkd_fields['Needs review']}")
                print(f"      Artist:  {rkd_fields['Artist']}")
                print(f"      Title:   {rkd_fields['Title'] or '(not found)'}")
                print(f"      Date:    {rkd_fields['Date English'] or '(not found)'}")
                print(f"      Medium:  {rkd_fields['Medium (raw)'] or '(not found)'}")

        except Exception as e:
            print(f"   -> ERROR: {e}")
            errors.append({"file": img_path.name, "error": str(e)})

        print()

    # --- Post-processing: lot number cross-matching ---
    lot_metadata = {}
    for rec in records:
        lot = rec.get("Lot No", "")
        if lot and rec.get("Artist") != "Anoniem" and lot not in lot_metadata:
            lot_metadata[lot] = rec

    merge_fields = ["Artist", "Title", "Medium (raw)", "Object name", "Height", "Width",
                    "Unit", "Genre", "Signature/inscription", "Signature location",
                    "Date Dutch", "Date English", "Search margin: begin",
                    "Search margin: end", "Provenance"]
    linked = 0
    for rec in records:
        lot = rec.get("Lot No", "")
        if lot and rec.get("Artist") == "Anoniem" and lot in lot_metadata:
            donor = lot_metadata[lot]
            for field in merge_fields:
                if not rec.get(field) and donor.get(field):
                    rec[field] = donor[field]
            rec["Needs review"] = rec.get("Needs review", "") + " [linked via lot#]"
            linked += 1
    if linked:
        print(f"\nPost-processing: linked {linked} records via lot number.")

    # --- Post-processing: adjacent page linking ---
    for rec in records:
        n = len(rec.get("Raw OCR text", ""))
        rec["_page_type"] = ("image_only" if n < 60 else
                             "text_only"  if n > 300 and rec.get("Artist") == "Anoniem"
                             else "mixed")

    linked_adj = 0
    for i, rec in enumerate(records):
        if rec["_page_type"] == "image_only" and rec.get("Artist") == "Anoniem":
            for neighbor in filter(None, [
                records[i - 1] if i > 0 else None,
                records[i + 1] if i + 1 < len(records) else None,
            ]):
                if (neighbor.get("Artist") != "Anoniem"
                        and neighbor["_page_type"] != "image_only"):
                    for field in ["Artist", "Title", "Medium (raw)", "Object name"]:
                        if not rec.get(field) and neighbor.get(field):
                            rec[field] = neighbor[field]
                    rec["Needs review"] = (
                        rec.get("Needs review", "") + " [metadata from adjacent page]"
                    ).strip()
                    linked_adj += 1
                    break
    if linked_adj:
        print(f"Post-processing: linked {linked_adj} image-only pages to adjacent metadata.")

    for rec in records:
        rec.pop("_page_type", None)

    if not records:
        print("No records produced.")
        return

    output_csv = os.path.join(OUTPUT_FOLDER, "rkd_metadata.csv")

    col_order = [
        "Source image",
        "Artist", "Title",
        "Date Dutch", "Date English", "Search margin: begin", "Search margin: end",
        "Object name", "Medium (normalized)", "Medium (raw)", "Genre",
        "Height", "Width", "Unit", "Shape",
        "Signature/inscription", "Signature location",
        "Provenance",
        "Needs review",
        "Image path (original)", "Image path (crop)",
        "Raw OCR text",
    ]
    internal_fields = ["Lot No", "Inventory number"]

    df = pd.DataFrame(records)
    df = df.drop(columns=[c for c in internal_fields if c in df.columns])
    existing_priority = [c for c in col_order if c in df.columns]
    other_cols        = [c for c in df.columns if c not in col_order]
    df = df[existing_priority + other_cols]
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total processed:      {len(records)}")
    print(f"Errors:               {len(errors)}")
    ok_count     = (df["Needs review"] == "ok").sum()
    review_count = (df["Needs review"] != "ok").sum()
    print(f"Records OK:           {ok_count}")
    print(f"Needs human review:   {review_count}")

    if review_count > 0:
        print("\nMost common fields needing review:")
        for field in ["artist", "title", "dimensions", "medium"]:
            count = df["Needs review"].str.contains(field, na=False).sum()
            if count:
                print(f"  {field}: {count} records")

    print(f"\nFiles saved:")
    print(f"  CSV:              {output_csv}")
    print(f"  Artwork crops:    {crops_dir}/")
    print(f"  Uncropped images: {uncropped_dir}/")
    print("\nNext step: import rkd_metadata.csv into Label Studio for review.")

    if errors:
        print(f"\nFiles with errors ({len(errors)}):")
        for e in errors:
            print(f"  {e['file']}: {e['error']}")


if __name__ == "__main__":
    run_pipeline()
