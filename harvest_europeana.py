"""
Europeana Multi-Artist Harvester — Fixed
Uses proxy_dc_creator field (correct EDM field for creator search).
Relaxed filters so we don't miss records.

BEFORE RUNNING:
  1. Paste your Europeana API key below (API_KEY = "...")
  2. Run: python harvest_europeana.py

Requires: pip install requests

Output:
    europeana_dataset/
    ├── images/          downloaded images
    ├── metadata_full/   one raw JSON per record
    └── metadata.csv     flat summary of all records
"""

import requests
import json
import csv
import time
import re
from pathlib import Path

# ── PASTE YOUR API KEY HERE ───────────────────────────────────────────────────
API_KEY = "ametobaciso"
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("europeana_dataset")
SEARCH_URL = "https://api.europeana.eu/record/v2/search.json"
DELAY      = 0.3
MAX_TOTAL  = 2000
ROWS       = 100

# Per-artist search queries.
# We try multiple strategies per artist:
#   1. proxy_dc_creator:"Name"  — exact EDM creator field
#   2. plain text search "Name" — catches records where name appears anywhere
# We do NOT restrict to reusability=open at search time — we handle that at download.
ARTISTS = [
    (
        "Theo van Hoytema",
        [
            'proxy_dc_creator:"Theo van Hoytema"',
            'proxy_dc_creator:"Theodorus van Hoytema"',
            'proxy_dc_creator:"T. van Hoytema"',
            '"Theo van Hoytema"',          # broad fallback
        ],
    ),
    (
        "Evert Moll",
        [
            'proxy_dc_creator:"Evert Moll"',
            'proxy_dc_creator:"Jan Evert Moll"',
            '"Evert Moll"',
        ],
    ),
    (
        "Harm Kamerlingh Onnes",
        [
            'proxy_dc_creator:"Harm Kamerlingh Onnes"',
            'proxy_dc_creator:"H.H. Kamerlingh Onnes"',
            '"Kamerlingh Onnes"',
        ],
    ),
    (
        "Willem Gerard Hofker",
        [
            'proxy_dc_creator:"Willem Gerard Hofker"',
            'proxy_dc_creator:"W.G. Hofker"',
            'proxy_dc_creator:"Willem Hofker"',
            '"Willem Hofker"',
        ],
    ),
    (
        "Jan van Vuuren",
        [
            'proxy_dc_creator:"Jan van Vuuren"',
            'proxy_dc_creator:"J. van Vuuren"',
            '"Jan van Vuuren"',
        ],
    ),
]

# ── Setup ──────────────────────────────────────────────────────────────────────

(OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "metadata_full").mkdir(parents=True, exist_ok=True)

# ── Helper: run one search query, return all paged results ────────────────────

def search_all(query: str) -> list:
    records  = []
    cursor   = "*"

    while True:
        params = {
            "wskey":   API_KEY,
            "query":   query,
            "qf":      "TYPE:IMAGE",   # images only — keep this to avoid books/audio
            "profile": "rich",
            "rows":    ROWS,
            "cursor":  cursor,
        }
        # Note: NOT adding reusability or media filters here —
        # those were cutting out too many valid records.
        # We check download-ability at image fetch time instead.

        try:
            r = requests.get(SEARCH_URL, params=params, timeout=20)

            if r.status_code == 401:
                print("\n  ERROR: API key rejected (401). Double-check your key.")
                exit(1)
            if r.status_code != 200:
                print(f" [HTTP {r.status_code}]")
                break

            data  = r.json()

            # Check for API-level errors
            if not data.get("success", True):
                print(f" [API error: {data.get('error', 'unknown')}]")
                break

            items = data.get("items", [])
            if not items:
                break

            records.extend(items)

            next_cursor = data.get("nextCursor")
            if not next_cursor or len(items) < ROWS:
                break

            cursor = next_cursor
            time.sleep(DELAY)

        except Exception as e:
            print(f" [error: {e}]")
            break

    return records


# ── Helper: safe filename ──────────────────────────────────────────────────────

def safe_filename(s: str, max_len: int = 80) -> str:
    s = re.sub(r'[^\w\-]', '_', s)
    return s.strip("_")[:max_len]


# ── Helper: extract flat metadata ─────────────────────────────────────────────

def extract_metadata(item: dict) -> dict:
    def first(field):
        v = item.get(field, [])
        return (v[0] if isinstance(v, list) and v else v) or ""

    def joined(field):
        v = item.get(field, [])
        if isinstance(v, list):
            return " | ".join(str(x) for x in v if x)
        return str(v) if v else ""

    return {
        "europeana_id":  item.get("id", ""),
        "title":         first("title"),
        "creator":       joined("dcCreator"),
        "date":          first("year"),
        "type":          first("type"),
        "description":   first("dcDescription"),
        "subject":       joined("dcSubject"),
        "format":        joined("dcFormat"),
        "rights":        first("rights"),
        "data_provider": first("dataProvider"),
        "provider":      first("provider"),
        "country":       first("country"),
        "landing_page":  first("guid"),
        "image_url":     first("edmIsShownBy"),
        "thumbnail_url": first("edmPreview"),
    }


# ── Helper: download image ─────────────────────────────────────────────────────

def download_image(url: str, dest_stem: Path) -> tuple:
    if not url:
        return "", "no_url"
    try:
        r = requests.get(url, timeout=25, allow_redirects=True)
        if r.status_code == 200:
            ct = r.headers.get("Content-Type", "image/jpeg")
            if "image" in ct or len(r.content) > 5000:
                ext = "jpg"
                if "png" in ct:  ext = "png"
                elif "tiff" in ct: ext = "tif"
                filename = dest_stem.name + f".{ext}"
                with open(dest_stem.parent / filename, "wb") as f:
                    f.write(r.content)
                return filename, "downloaded"
            return "", "not_image"
        if r.status_code in (401, 403):
            # Try thumbnail as fallback
            return "", "access_denied"
        return "", f"http_{r.status_code}"
    except Exception:
        return "", "error"


# ── Step 1: Collect records ────────────────────────────────────────────────────

print(f"\n{'='*65}")
print("  Europeana Multi-Artist Harvester")
print(f"{'='*65}\n")
print("[ Step 1 ] Searching Europeana...\n")

all_records: dict = {}   # europeana_id -> (item, artist_name)

for display_name, queries in ARTISTS:
    print(f"  {display_name}")
    artist_new = 0

    for query in queries:
        items = search_all(query)
        new   = [it for it in items if it.get("id") not in all_records]
        for it in new:
            all_records[it["id"]] = (it, display_name)
        artist_new += len(new)
        print(f"    {query[:55]:<55}  {len(items):>4} hits, {len(new):>4} new")

    print(f"    Subtotal: {artist_new} new records\n")

    if len(all_records) >= MAX_TOTAL:
        print(f"  Reached {MAX_TOTAL} limit.\n")
        break

flat_list = list(all_records.values())
print(f"  Total unique records: {len(flat_list)}")

if not flat_list:
    print("\n  Still no results. Possible causes:")
    print("  - API key might be invalid. Test it manually:")
    print(f"    https://api.europeana.eu/record/v2/search.json?wskey={API_KEY}&query=Rembrandt&rows=3")
    print("  - Paste that URL in your browser — if you get JSON back, the key works.")
    exit(0)

# ── Step 2: Save metadata + download images ────────────────────────────────────

print(f"\n[ Step 2 ] Saving metadata and downloading images...\n")

csv_rows = []
counts   = {"downloaded": 0, "access_denied": 0, "no_url": 0, "other": 0}

for i, (item, artist_name) in enumerate(flat_list, 1):
    euro_id  = item.get("id", f"unknown_{i}")
    id_slug  = safe_filename(euro_id.replace("/", "_"))

    print(f"  [{i:>4}/{len(flat_list)}] {id_slug[:35]:<35}", end="  ", flush=True)

    # Save raw JSON
    json_path = OUTPUT_DIR / "metadata_full" / f"{id_slug}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(item, f, indent=2, ensure_ascii=False)

    fields = extract_metadata(item)

    # Try full image first, then thumbnail as fallback
    img_url  = fields["image_url"]
    thumb    = fields["thumbnail_url"]
    dest     = OUTPUT_DIR / "images" / id_slug

    img_file, img_status = download_image(img_url, dest)

    # If full image failed, try thumbnail
    if img_status != "downloaded" and thumb:
        img_file, img_status = download_image(thumb, dest)
        if img_status == "downloaded":
            img_status = "downloaded_thumbnail"

    if "downloaded" in img_status:
        counts["downloaded"] += 1
        tag = "IMG"
    elif img_status == "access_denied":
        counts["access_denied"] += 1
        tag = " AD"
    elif img_status == "no_url":
        counts["no_url"] += 1
        tag = "   "
    else:
        counts["other"] += 1
        tag = "ERR"

    print(f"[{tag}]  {(fields['title'] or '(no title)')[:38]}")

    csv_rows.append({
        "artist_name":  artist_name,
        **fields,
        "image_status": img_status,
        "image_file":   img_file,
        "json_file":    json_path.name,
    })

    time.sleep(DELAY)

# ── Step 3: Save CSV ───────────────────────────────────────────────────────────

print(f"\n[ Step 3 ] Writing metadata.csv...")
if csv_rows:
    with open(OUTPUT_DIR / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

# ── Summary ────────────────────────────────────────────────────────────────────

artist_counts = {}
artist_images = {}
for row in csv_rows:
    a = row["artist_name"]
    artist_counts[a] = artist_counts.get(a, 0) + 1
    artist_images[a] = artist_images.get(a, 0) + (1 if "downloaded" in row["image_status"] else 0)

print(f"\n{'='*65}")
print("  Done!")
print(f"{'='*65}")
print(f"  Total records         : {len(csv_rows)}")
print(f"  Images downloaded     : {counts['downloaded']}")
print(f"  Access denied         : {counts['access_denied']}")
print(f"  No image URL          : {counts['no_url']}")
print(f"  Other errors          : {counts['other']}")
print()
print("  By artist:")
for a in artist_counts:
    print(f"    {a:<35} {artist_counts[a]:>4} records   {artist_images[a]:>4} images")
print()
print(f"  Output: {OUTPUT_DIR.resolve()}/")
print(f"    images/         {counts['downloaded']} files")
print(f"    metadata_full/  {len(csv_rows)} JSON files")
print(f"    metadata.csv    {len(csv_rows)} rows")
print(f"{'='*65}\n")
