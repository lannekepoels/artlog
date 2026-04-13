"""
Extraction Web App
======================
Run with:
    pip install flask pillow pandas requests opencv-python pydantic google-genai google-cloud-vision
    python app.py

Then open: http://localhost:5000
"""

import os
import json
import zipfile
import uuid
import threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB upload limit

BASE_DIR     = Path(__file__).parent
UPLOAD_DIR   = BASE_DIR / "uploads"
RESULTS_DIR  = BASE_DIR / "results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# In-memory job store  { job_id: { status, progress, log, records, images } }
JOBS: dict = {}
JOBS_LOCK = threading.Lock()

# ── Text-normalisation helpers ──────────────────────────────────────────────

_DUTCH_PARTICLES = {'van', 'de', 'den', 'der', 'ter', 'ten', 'op', 'in', 'het', "'t"}

def smart_title_case(text: str) -> str:
    """Convert ALL-CAPS artist names to proper title case with Dutch particles."""
    if not text or not text.strip().isupper():
        return text
    words = text.lower().split()
    result = []
    for i, word in enumerate(words):
        result.append(word if i > 0 and word in _DUTCH_PARTICLES else word.capitalize())
    return ' '.join(result)

def smart_sentence_case(text: str) -> str:
    """Convert ALL-CAPS titles to sentence case (first letter only)."""
    if not text or not text.strip().isupper():
        return text
    lower = text.lower()
    return lower[0].upper() + lower[1:]

def infer_shape(height, width, current_shape: str) -> str:
    """Fill in Dutch shape term from dimensions when shape is empty."""
    if current_shape and str(current_shape).strip():
        return current_shape
    try:
        h = float(str(height).replace(',', '.'))
        w = float(str(width).replace(',', '.'))
        if h == w:
            return 'vierkant'
        return 'liggende rechthoek' if h < w else 'staande rechthoek'
    except (ValueError, TypeError, AttributeError):
        return current_shape or ''

# ── CSV column order (matches annotation interface) ──────────────────────────
CSV_COLUMN_ORDER = [
    "Artist", "Title", "Date",
    "Search_Margin_Begin", "Search_Margin_End",
    "Genre", "Object_Name", "Medium", "Shape",
    "Height", "Width", "Unit",
    "Signature_Inscription", "Signature_Location", "Provenance",
    "Artwork_Number", "Image_Number", "_filename",
    "FullEntryText", "_notes",
    "_id", "_validated", "_flagged", "_original_url", "_crop_url",
]

def ordered_dataframe(records):
    import pandas as pd
    df = pd.DataFrame(records)
    ordered = [c for c in CSV_COLUMN_ORDER if c in df.columns]
    remaining = [c for c in df.columns if c not in CSV_COLUMN_ORDER]
    return df[ordered + remaining]


# ─────────────────────────────────────────────
# ROUTES — pages
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/annotate/<job_id>")
def annotate(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return "Job not found", 404
    return render_template("annotate.html", job_id=job_id)


# ─────────────────────────────────────────────
# ROUTES — API
# ─────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def upload():
    """Accept a ZIP or folder of images, start pipeline job."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded = request.files["file"]
    api_keys = {
        "google":  request.form.get("google_api_key", "").strip(),
        "gemini":  request.form.get("gemini_api_key", "").strip(),
    }

    if not api_keys["google"] or not api_keys["gemini"]:
        return jsonify({"error": "Both API keys are required"}), 400

    job_id   = str(uuid.uuid4())[:8]
    work_dir = UPLOAD_DIR / job_id
    work_dir.mkdir(parents=True)

    # Save and extract ZIP
    zip_path = work_dir / "upload.zip"
    uploaded.save(str(zip_path))

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(work_dir / "images"))
    except zipfile.BadZipFile:
        return jsonify({"error": "Uploaded file is not a valid ZIP archive"}), 400

    # Collect image files (skip __MACOSX and hidden files)
    supported = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    image_files = sorted([
        p for p in (work_dir / "images").rglob("*")
        if p.suffix.lower() in supported
        and not any(part.startswith("__") or part.startswith(".") for part in p.parts)
    ])

    if not image_files:
        return jsonify({"error": "No supported image files found in ZIP"}), 400

    with JOBS_LOCK:
        JOBS[job_id] = {
            "status":   "queued",
            "progress": 0,
            "total":    len(image_files),
            "log":      [],
            "records":  [],
            "images":   {},   # filename -> url-accessible path
        }

    # Run pipeline in background thread
    t = threading.Thread(
        target=run_pipeline_job,
        args=(job_id, image_files, work_dir, api_keys),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id, "total": len(image_files)})


@app.route("/api/job/<job_id>")
def job_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404
    return jsonify({
        "status":   job["status"],
        "progress": job["progress"],
        "total":    job["total"],
        "log":      job["log"][-20:],   # last 20 log lines
        "count":    len(job["records"]),
    })


@app.route("/api/job/<job_id>/records")
def job_records(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404
    return jsonify(job["records"])


@app.route("/api/job/<job_id>/save", methods=["POST"])
def save_records(job_id):
    """Save annotated/edited records back to the job."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404

    data = request.get_json()
    records = data.get("records", [])

    with JOBS_LOCK:
        JOBS[job_id]["records"] = records

    # Also write CSV to results folder (excluding deleted records)
    try:
        exportable = [r for r in records if not r.get("_deleted")]
        df = ordered_dataframe(exportable)
        csv_path = RESULTS_DIR / f"{job_id}_annotated.csv"
        df.to_csv(str(csv_path), index=False, encoding="utf-8-sig")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"saved": len(exportable), "csv": f"/results/{job_id}_annotated.csv"})


@app.route("/api/job/<job_id>/export")
def export_csv(job_id):
    """Trigger CSV export and return download link."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404

    try:
        exportable = [r for r in job["records"] if not r.get("_deleted")]
        df = ordered_dataframe(exportable)
        csv_path = RESULTS_DIR / f"{job_id}_annotated.csv"
        df.to_csv(str(csv_path), index=False, encoding="utf-8-sig")
        return jsonify({"url": f"/results/{job_id}_annotated.csv"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/results/<filename>")
def download_result(filename):
    return send_from_directory(str(RESULTS_DIR), filename, as_attachment=True)


@app.route("/image/<job_id>/<path:filename>")
def serve_image(job_id, filename):
    """Serve uploaded/cropped images for display in the UI."""
    work_dir = UPLOAD_DIR / job_id
    return send_from_directory(str(work_dir), filename)


@app.route("/api/job/<job_id>/export_images")
def export_images(job_id):
    """Bundle all cropped images into a ZIP and return it for download."""
    import io
    from collections import defaultdict

    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404

    work_dir = UPLOAD_DIR / job_id

    # Group crop URLs by original filename to handle numbering
    # Only skip records the user has explicitly deleted — blank pages are included
    groups = defaultdict(list)
    for rec in job["records"]:
        if rec.get("_deleted"):
            continue
        orig = rec.get("_filename", "")
        crop_url = rec.get("_crop_url", "") or rec.get("_original_url", "")
        if orig and crop_url:
            groups[orig].append(crop_url)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for orig_fname, crop_urls in groups.items():
            stem = Path(orig_fname).stem
            for i, url in enumerate(crop_urls):
                # Strip cache-bust query param and resolve to local path
                rel = url.split("?")[0].replace(f"/image/{job_id}/", "")
                img_file = work_dir / rel
                if not img_file.exists():
                    # Fall back to the original page scan
                    matches = list((work_dir / "images").rglob(orig_fname))
                    if not matches:
                        continue
                    img_file = matches[0]

                if len(crop_urls) == 1:
                    zip_name = f"{stem}_crop.jpg"
                else:
                    zip_name = f"{stem}_crop_{i + 1}.jpg"
                zf.write(str(img_file), zip_name)

    buf.seek(0)
    return send_file(
        buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"crops_{job_id}.zip",
    )


@app.route("/api/job/<job_id>/record/<record_id>/crop", methods=["POST"])
def crop_record(job_id, record_id):
    """Manually crop the artwork region for a record."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404

    data = request.get_json()
    filename = data.get("filename", "")
    x_norm = float(data.get("x", 0))
    y_norm = float(data.get("y", 0))
    w_norm = float(data.get("w", 1))
    h_norm = float(data.get("h", 1))

    work_dir = UPLOAD_DIR / job_id
    matching = list((work_dir / "images").rglob(filename))
    if not matching:
        return jsonify({"error": "Image not found"}), 404
    img_path = matching[0]

    try:
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            return jsonify({"error": "Could not read image"}), 500

        ih, iw = img.shape[:2]
        x1 = max(0, int(x_norm * iw))
        y1 = max(0, int(y_norm * ih))
        x2 = min(iw, int((x_norm + w_norm) * iw))
        y2 = min(ih, int((y_norm + h_norm) * ih))

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return jsonify({"error": "Invalid crop region"}), 400

        crops_dir = work_dir / "crops"
        crops_dir.mkdir(exist_ok=True)
        crop_fname = f"{Path(img_path).stem}_crop_manual.jpg"
        cv2.imwrite(str(crops_dir / crop_fname), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

        crop_url = f"/image/{job_id}/crops/{crop_fname}"

        with JOBS_LOCK:
            for rec in job["records"]:
                if rec.get("_id") == record_id:
                    rec["_crop_url"] = crop_url
                    break

        return jsonify({"crop_url": crop_url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# PIPELINE RUNNER (background thread)
# ─────────────────────────────────────────────

def log_job(job_id: str, msg: str):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id]["log"].append(msg)
    print(f"[{job_id}] {msg}")


def run_pipeline_job(job_id: str, image_files: list, work_dir: Path, api_keys: dict):
    """Run the full extraction pipeline for a job."""

    with JOBS_LOCK:
        JOBS[job_id]["status"] = "running"

    crops_dir = work_dir / "crops"
    crops_dir.mkdir(exist_ok=True)

    try:
        import cv2
        import numpy as np
        import time
        import base64
        import mimetypes
        from google.cloud import vision
        from google import genai

        # ── Build Vision client ──
        vision_client = vision.ImageAnnotatorClient(
            client_options={"api_key": api_keys["google"]},
            transport="rest",
        )

        DIRECT_ART_TERMS = {
            "painting", "artwork", "illustration", "picture",
            "photograph", "poster", "print", "drawing", "sculpture",
            "picture frame",
        }
        MIN_OBJ_SCORE = 0.4
        MIN_CROP_PX   = 100
        MIN_VARIANCE  = 500

        def is_direct_art_label(label):
            return any(t in label.lower() for t in DIRECT_ART_TERMS)

        def is_valid_crop(crop):
            h, w = crop.shape[:2]
            return w >= MIN_CROP_PX and h >= MIN_CROP_PX and float(np.var(crop)) >= MIN_VARIANCE

        GEMINI_PROMPT = """Extract all complete, numbered artwork entries from this art historical catalog image.
An entry must include the artist's attribution and all directly associated text lines until the next entry appears.

For each entry, extract these fields following RKD metadata standards:

Artwork_Number: The catalog number of the artwork (e.g. '1', '2').
Artist: Full name of the artist from header or inline text (e.g. 'door', 'van', 'dezelfden').
Date: Creation date as it appears in the source text (e.g. '1887 gedateerd', 'ca. 1880', 'dated 1887').
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

        def validate_genre(raw):
            if not raw:
                return ""
            raw_lower = raw.lower()
            allowed = set(GENRE_KEYWORDS.keys())
            if raw_lower in allowed:
                return raw_lower
            matched = [g for g in allowed if g in raw_lower]
            if matched:
                return " | ".join(matched)
            scored = [(g, sum(1 for kw in kws if kw in raw_lower)) for g, kws in GENRE_KEYWORDS.items()]
            scored = [(g, s) for g, s in scored if s > 0]
            scored.sort(key=lambda x: -x[1])
            return " | ".join(g for g, _ in scored[:3]) if scored else ""

        total = len(image_files)

        for i, img_path in enumerate(image_files):
            log_job(job_id, f"[{i+1}/{total}] Processing {img_path.name}")

            try:
                image_bgr = cv2.imread(str(img_path))
                if image_bgr is None:
                    log_job(job_id, f"  ⚠ Could not read image, skipping")
                    continue
                img_h, img_w = image_bgr.shape[:2]

                # ── Vision API ──
                with open(str(img_path), "rb") as f:
                    content = f.read()
                vision_image = vision.Image(content=content)
                vision_req = vision.AnnotateImageRequest(
                    image=vision_image,
                    features=[
                        vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION),
                        vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION),
                    ]
                )
                vision_resp = vision_client.annotate_image(vision_req)
                time.sleep(0.1)

                ocr_text = ""
                if vision_resp and vision_resp.full_text_annotation:
                    ocr_text = vision_resp.full_text_annotation.text

                # Blank page check
                boxes = []
                if vision_resp:
                    for obj in vision_resp.localized_object_annotations:
                        if is_direct_art_label(obj.name) and obj.score >= MIN_OBJ_SCORE:
                            verts = obj.bounding_poly.normalized_vertices
                            xs = [v.x * img_w for v in verts]
                            ys = [v.y * img_h for v in verts]
                            boxes.append((min(xs), min(ys), max(xs), max(ys)))

                if len(ocr_text.strip()) < 10 and not boxes:
                    log_job(job_id, f"  ↳ Blank page — adding BLANK record")
                    original_url = f"/image/{job_id}/images/{img_path.relative_to(work_dir / 'images')}"
                    blank_rec = {
                        "Artist": "BLANK",
                        "Title": "BLANK",
                        "_id": str(uuid.uuid4())[:8],
                        "_filename": img_path.name,
                        "_original_url": original_url,
                        "_crop_url": original_url,
                        "_validated": False,
                        "_flagged": False,
                        "_deleted": False,
                        "_notes": "",
                        "_blank": True,
                    }
                    with JOBS_LOCK:
                        JOBS[job_id]["records"].append(blank_rec)
                        JOBS[job_id]["progress"] = i + 1
                    continue

                # ── Crop artwork region ──
                crop_url = None
                if boxes:
                    x1, y1, x2, y2 = boxes[0]
                    cx1, cy1 = max(0, int(x1)), max(0, int(y1))
                    cx2, cy2 = min(img_w, int(x2)), min(img_h, int(y2))
                    crop = image_bgr[cy1:cy2, cx1:cx2]
                    if crop.size > 0 and is_valid_crop(crop):
                        crop_fname = f"{img_path.stem}_crop.jpg"
                        cv2.imwrite(str(crops_dir / crop_fname), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        crop_url = f"/image/{job_id}/crops/{crop_fname}"

                original_url = f"/image/{job_id}/images/{img_path.relative_to(work_dir / 'images')}"

                # ── Gemini extraction ──
                with open(str(img_path), "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")
                mime_type = mimetypes.guess_type(str(img_path))[0] or "image/jpeg"

                gemini_client = genai.Client(api_key=api_keys["gemini"])
                entries = []
                for attempt in range(3):
                    try:
                        response = gemini_client.models.generate_content(
                            model="gemini-3-flash-preview",
                            contents=[{"role": "user", "parts": [
                                {"inline_data": {"mime_type": mime_type, "data": image_data}},
                                {"text": GEMINI_PROMPT + "\n\nRespond with a JSON array only."},
                            ]}],
                            config={"response_mime_type": "application/json", "max_output_tokens": 65536},
                        )
                        raw = response.text.strip()
                        if raw.startswith("```"):
                            raw = raw.split("```")[1]
                            if raw.startswith("json"):
                                raw = raw[4:]
                        entries = json.loads(raw)
                        break
                    except Exception as e:
                        err = str(e)
                        if "429" in err or "RESOURCE_EXHAUSTED" in err:
                            wait = 60 * (attempt + 1)
                            log_job(job_id, f"  ⏳ Rate limit, waiting {wait}s...")
                            time.sleep(wait)
                        else:
                            log_job(job_id, f"  ⚠ Gemini error: {e}")
                            break

                if not entries:
                    entries = [{"Artist": "Anoniem", "Title": "", "FullEntryText": ""}]

                log_job(job_id, f"  ✓ {len(entries)} entr{'y' if len(entries)==1 else 'ies'} extracted")

                for entry in entries:
                    rec = entry if isinstance(entry, dict) else {}
                    rec["Artist"] = smart_title_case(rec.get("Artist", "") or "")
                    rec["Title"]  = smart_sentence_case(rec.get("Title", "") or "")
                    rec["Shape"]  = infer_shape(rec.get("Height"), rec.get("Width"), rec.get("Shape", ""))
                    rec["Genre"]  = validate_genre(rec.get("Genre", "") or "")
                    rec["_id"]           = str(uuid.uuid4())[:8]
                    rec["_filename"]     = img_path.name
                    rec["_original_url"] = original_url
                    rec["_crop_url"]     = crop_url or ""
                    rec["_validated"]    = False
                    rec["_notes"]        = ""

                    with JOBS_LOCK:
                        JOBS[job_id]["records"].append(rec)

            except Exception as e:
                log_job(job_id, f"  ✗ Error: {e}")

            with JOBS_LOCK:
                JOBS[job_id]["progress"] = i + 1
            time.sleep(3)  # polite delay between images

        with JOBS_LOCK:
            JOBS[job_id]["status"] = "done"
        log_job(job_id, f"✅ Pipeline complete — {len(JOBS[job_id]['records'])} records extracted")

    except ImportError as e:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "error"
        log_job(job_id, f"✗ Missing dependency: {e}. Run: pip install flask pillow pandas requests opencv-python pydantic google-genai google-cloud-vision")
    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "error"
        log_job(job_id, f"✗ Fatal error: {e}")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  RKD Extraction App — http://localhost:5001")
    print("="*50 + "\n")
    # use_reloader=False is critical — debug mode's file watcher would
    # restart the server when a ZIP is saved, wiping the in-memory job store.
    app.run(debug=True, host="0.0.0.0", port=5001, use_reloader=False)
