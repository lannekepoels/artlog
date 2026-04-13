# Extraction Pipeline — Web App

A local web application that runs your catalogue extraction pipeline and provides an annotation interface for reviewing and correcting extracted metadata.

---

## Quick Start

### 1. Install dependencies

```bash
cd interface
pip install -r requirements.txt
```

> **Note:** If you're using a system Python or get errors, try:
> ```bash
> pip install -r requirements.txt --break-system-packages
> ```
> Or use a virtual environment:
> ```bash
> python3 -m venv venv && source venv/bin/activate
> pip install -r requirements.txt
> ```

### 2. Run the app

```bash
python app.py
```

### 3. Open in browser

```
http://localhost:5000
```

---

## How it Works

### Upload page (`/`)
1. Enter your **Google Cloud Vision API key** and **Gemini API key**
2. Upload a **ZIP file** containing your catalogue scan images (`.jpg`, `.png`, `.tif`, etc.)
3. Click **Run Extraction Pipeline** — the pipeline runs in the background
4. Watch the live log as pages are processed
5. When done, click **Open Annotation Interface**

### Annotation interface (`/annotate/<job_id>`)

- **Left panel**: Scrollable record list with thumbnails, filterable by artist/title
- **Center panel**: Image viewer — toggle between cropped artwork and full page scan
- **Right panel**: All RKD metadata fields, fully editable inline

#### Actions per record:
| Action | How |
|--------|-----|
| Edit any field | Click and type directly |
| Mark as validated | Click **✓ Mark Validated** or press `V` |
| Flag for review | Click **⚑ Flag** |
| Navigate records | Click Prev/Next, or use `←` `→` arrow keys |
| Filter records | Type in the search box |

#### Global actions (top bar):
| Button | What it does |
|--------|-------------|
| **Save All** | Saves all edits back to the server (JSON) |
| **Export CSV** | Saves + downloads a UTF-8 CSV of all records |

---

## File Structure

```
interface/
├── app.py              ← Flask server + pipeline runner
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
├── templates/
│   ├── index.html      ← Upload & progress page
│   └── annotate.html   ← Annotation interface
├── uploads/            ← Uploaded ZIPs and extracted images (auto-created)
└── results/            ← Exported CSVs (auto-created)
```

### Standalone pipeline scripts (outside `interface/`)

| File                            | Description                                                                                                              |
|---------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| `rkd_vision_gemini_pipeline.py` | Full pipeline: Google Cloud Vision (blank page detection + artwork cropping) + Gemini metadata extraction. Outputs CSV. |
| `metadata_extraction_gemini`    | Gemini-only extraction (no Vision step). Outputs Excel.                                                                  |

---

## Notes

- **API keys are never stored** — they're passed in memory only for the duration of the pipeline run
- The web app pipeline uses `gemini-3-flash-preview`
- Rate limiting is handled automatically with exponential backoff
- Each job gets a unique 8-character ID; you can run multiple jobs without conflicts
- Keyboard shortcut: `V` = validate current record, `←`/`→` = navigate

---

## Troubleshooting

**"No module named 'cv2'"**
```bash
pip install opencv-python
```

**Gemini 429 rate limit errors**
The pipeline automatically waits and retries. For large batches, consider increasing `DELAY_BETWEEN_REQUESTS` in `app.py`.

**"No supported image files found in ZIP"**
Make sure your ZIP contains images at the top level or one folder deep. macOS `.DS_Store` and `__MACOSX` folders are automatically ignored.
