import os
import pandas as pd
import re
import requests

# ============================================================
# 0. CONFIGURATION
# ============================================================

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
AUCTION_FILE = os.path.join(BASE_DIR, 'data', 'Auction Export - 685 - 090226.xlsx')
FIELDS_FILE  = os.path.join(BASE_DIR, 'data', 'Metadata fields.xlsx')
IMAGES_DIR   = os.path.join(BASE_DIR, 'data', 'auction_images')

# RKD API base URLs (no key required for public search)
RKD_ARTISTS_URL = 'https://api.rkd.nl/api/search/artists'
RKD_IMAGES_URL  = 'https://api.rkd.nl/api/search/images'

# ============================================================
# 1. CONTROLLED VOCABULARIES (from manual)
# ============================================================

# Genre list a–p (manual Section "Genre")
VALID_GENRES = {
    'abstraction', 'architecture (genre)', 'animal painting (genre)',
    'figure', 'genre', 'history (genre)', 'interior view',
    'church interior', 'landscape (genre)', 'marine (genre)',
    'undetermined', 'design (artistic concept)', 'portrait',
    'cityscape', 'still life', 'study (visual work)'
}

# Keyword → genre mapping
GENRE_KEYWORDS = {
    'portrait':                  ['portrait', 'self-portrait', 'bildnis', 'portret',
                                  'bust', 'likeness'],
    'still life':                ['still life', 'stilleven', 'flowers', 'fruit', 'vase',
                                  'trompe l\'oeil', 'roses', 'bouquet', 'jug', 'pitcher',
                                  'peaches', 'grapes', 'vegetables', 'pipe', 'book',
                                  'hanging birds', 'mistle toe', 'anemones'],
    'landscape (genre)':         ['landscape', 'landschap', 'forest', 'meadow', 'polder',
                                  'dunes', 'countryside', 'moorlands', 'trees', 'village',
                                  'farm', 'shepherd', 'pastoral', 'winter', 'autumn',
                                  'country road', 'wooded', 'lane', 'ploughing',
                                  'haystack', 'waterfall', 'sheepfold', 'flock'],
    'marine (genre)':            ['harbour', 'harbor', 'boats', 'ship', 'sea', 'beach',
                                  'sailing', 'fishing', 'maritime', 'river', 'canal',
                                  'haven', 'breakers', 'surf', 'bomschuiten', 'catch',
                                  'unloading', 'port', 'mast', 'pier', 'anchor',
                                  'coastal', 'flat-bottomed'],
    'cityscape':                 ['cityscape', 'city', 'street', 'town', 'amsterdam',
                                  'rotterdam', 'hague', 'delft', 'urban', 'buildings',
                                  'binnenhaven', 'geldersekade', 'montelbaanstoren',
                                  'singel', 'ouddorp', 'alkmaar', 'mechelen',
                                  'buitenhof', 'demolition'],
    'figure':                    ['figure', 'nude', 'woman', 'man', 'child', 'people',
                                  'figures', 'bathing', 'workers', 'peasant woman',
                                  'seated', 'standing', 'reclining', 'girl', 'boy',
                                  'young', 'mother', 'family', 'fisher', 'laundry'],
    'animal painting (genre)':   ['cat', 'dog', 'horse', 'cow', 'animal', 'bird',
                                  'kittens', 'peacock', 'chickens', 'sheep', 'goats',
                                  'duck', 'spaniel', 'cows', 'barnyard', 'hen'],
    'interior view':             ['interior', 'inn', 'tavern', 'room', 'monastery',
                                  'cafe', 'kitchen', 'town hall', 'ballroom',
                                  'forge', 'interment camp', 'laren interior'],
    'church interior':           ['church interior', 'cathedral interior', 'jacobikerk',
                                  'nieuwe kerk interior', 'gate of a monastery'],
    'history (genre)':           ['biblical', 'mythology', 'allegory', 'tobias', 'angel',
                                  'presentation in the temple', 'assumption of the virgin',
                                  'saint', 'religious', 'margaret the virgin', 'dragon',
                                  'peter the great', 'fete galante'],
    'genre':                     ['market', 'fair', 'laundry day', 'courting', 'dancing',
                                  'playing cards', 'reading', 'family gathering',
                                  'gathering wood', 'mending nets', 'feeding',
                                  'tavern', 'inn', 'watching', 'carousel',
                                  'village fair', 'street scene'],
    'abstraction':               ['abstract', 'compositie', 'untitled', 'geometric',
                                  'composition', 'mixed media on canvas'],
    'design (artistic concept)': ['design', 'fabric design', 'poster', 'illustration',
                                  'stoomweverij'],
    'study (visual work)':       ['study', 'sketch', 'studie', 'studies',
                                  'landscape studies'],
    'architecture (genre)':      ['architecture', 'capriccio', 'ruins', 'facade',
                                  'tower', 'gate', 'accijnstoren'],
}

# Object name mapping: medium keywords -> RKD object term
OBJECT_NAME_MAP = [
    (['oil on canvas', 'oil on panel', 'oil on board', 'oil on paper',
      'oil on copper', 'oil stick'],                     'painting'),
    (['watercolour', 'watercolor', 'gouache', 'pencil on paper',
      'chalk on paper', 'charcoal', 'ink on paper', 'pastel on paper',
      'coloured chalk', 'black chalk', 'pen and ink', 'pen and brown ink',
      'pencil', 'drypoint', 'washed ink', 'indian ink', 'sepia'],  'drawing'),
    (['etching', 'engraving', 'lithograph', 'drypoint etching',
      'colour lithograph', 'embossed tin'],              'print'),
    (['bronze', 'paper-mache', 'papier-mache',
      'cotton', 'silk', 'wool'],                         'sculpture'),
    (['mixed media'],                                    'painting'),
    (['acrylic on canvas', 'acrylic'],                   'painting'),
    (['album', 'portfolio'],                             'album'),
]

# ============================================================
# 2. CONFIDENCE SCORING FRAMEWORK
# ============================================================
# Each field has a max weight; all weights sum to 100.
# Image field removed from this version; weight redistributed.

FIELD_WEIGHTS = {
    'artwork_number': 17,
    'artist_name':    23,
    'date':           17,
    'genre':          17,
    'object_name':    11,
    'dimensions':     11,
    'subject_kw':      4,
}

def score_artwork_number(rkd_found: bool) -> tuple:
    if rkd_found:
        return FIELD_WEIGHTS['artwork_number'], 'RKD match confirmed'
    return 0, 'No RKD match found - human must assign'

def score_artist_name(rkd_found: bool, qualifier: str,
                      raw_name: str, is_school: bool) -> tuple:
    w = FIELD_WEIGHTS['artist_name']
    if is_school or raw_name == 'Anoniem':
        return round(w * 0.5), 'Anonymous/school entry - attribution inherently uncertain'
    if rkd_found and not qualifier:
        return w, 'Name confirmed in RKDartists'
    if rkd_found and qualifier:
        return round(w * 0.75), f'Name in RKDartists but has qualifier: {qualifier}'
    if not rkd_found and qualifier:
        return round(w * 0.3), f'Not in RKDartists; attribution qualifier present: {qualifier}'
    return round(w * 0.2), 'Not found in RKDartists - defaulted to Anoniem'

def score_date(date_dutch: str, sig_dated: bool,
               used_lifespan: bool, circa: bool) -> tuple:
    w = FIELD_WEIGHTS['date']
    if sig_dated and not circa:
        return w, 'Year from signed inscription - high confidence'
    if sig_dated and circa:
        return round(w * 0.75), 'Circa date from signature'
    if date_dutch and not used_lifespan:
        return round(w * 0.65), 'Year extracted from description text'
    if used_lifespan:
        return round(w * 0.35), 'No date found - margins estimated from artist lifespan'
    return 0, 'Date unknown - no source available'

def score_genre(genres: list, num_kw_matched: int) -> tuple:
    w = FIELD_WEIGHTS['genre']
    if 'undetermined' in genres:
        return 0, 'No genre keywords matched'
    if num_kw_matched >= 3:
        return w, 'Multiple strong keyword matches'
    if num_kw_matched == 2:
        return round(w * 0.75), 'Two keyword matches'
    if num_kw_matched == 1:
        return round(w * 0.55), 'Single keyword match - verify'
    return round(w * 0.4), 'Weak keyword match'

def score_object_name(obj_name: str, desc: str) -> tuple:
    w = FIELD_WEIGHTS['object_name']
    direct_terms = ['oil on canvas', 'oil on panel', 'oil on board',
                    'watercolour', 'gouache', 'etching', 'engraving',
                    'lithograph', 'bronze', 'pencil on paper', 'charcoal']
    if any(t in desc.lower() for t in direct_terms):
        return w, f'Medium explicitly stated -> {obj_name}'
    if obj_name != 'painting':
        return round(w * 0.7), f'Medium inferred -> {obj_name}'
    return round(w * 0.5), 'Defaulted to painting - verify medium'

def score_dimensions(height, width) -> tuple:
    w = FIELD_WEIGHTS['dimensions']
    if height and width:
        return w, 'Both dimensions extracted'
    if height or width:
        return round(w * 0.5), 'Only one dimension extracted'
    return 0, 'No dimensions found'

def score_subject_kw() -> tuple:
    return round(FIELD_WEIGHTS['subject_kw'] * 0.1), \
           'Left blank - requires expert AAT/RKD verification'

def compute_confidence(rkd_artwork_found, rkd_artist_found, qualifier,
                       raw_name, is_school, date_dutch, sig_dated,
                       used_lifespan, circa, genres, num_kw_matched,
                       obj_name, desc, height, width) -> dict:
    s_art_no, n_art_no = score_artwork_number(rkd_artwork_found)
    s_artist, n_artist = score_artist_name(rkd_artist_found, qualifier,
                                           raw_name, is_school)
    s_date,   n_date   = score_date(date_dutch, sig_dated,
                                    used_lifespan, circa)
    s_genre,  n_genre  = score_genre(genres, num_kw_matched)
    s_obj,    n_obj    = score_object_name(obj_name, desc)
    s_dims,   n_dims   = score_dimensions(height, width)
    s_subj,   n_subj   = score_subject_kw()

    total = s_art_no + s_artist + s_date + s_genre + s_obj + s_dims + s_subj

    breakdown = (
        f"artwork_no:{s_art_no}/{FIELD_WEIGHTS['artwork_number']} ({n_art_no}) | "
        f"artist:{s_artist}/{FIELD_WEIGHTS['artist_name']} ({n_artist}) | "
        f"date:{s_date}/{FIELD_WEIGHTS['date']} ({n_date}) | "
        f"genre:{s_genre}/{FIELD_WEIGHTS['genre']} ({n_genre}) | "
        f"object:{s_obj}/{FIELD_WEIGHTS['object_name']} ({n_obj}) | "
        f"dims:{s_dims}/{FIELD_WEIGHTS['dimensions']} ({n_dims}) | "
        f"subject_kw:{s_subj}/{FIELD_WEIGHTS['subject_kw']} ({n_subj})"
    )

    return {
        'confidence_pct':       total,
        'confidence_breakdown': breakdown,
        'conf_artwork_number':  s_art_no,
        'conf_artist_name':     s_artist,
        'conf_date':            s_date,
        'conf_genre':           s_genre,
        'conf_object_name':     s_obj,
        'conf_dimensions':      s_dims,
        'conf_subject_kw':      s_subj,
    }

# ============================================================
# 3. RKD API LOOKUPS (placeholder stubs)
# ============================================================

def lookup_rkd_artist(name: str) -> dict:
    """
    Query RKDartists for a name match.
    TO ACTIVATE: uncomment the block below and remove the stub return.

    resp = requests.get(RKD_ARTISTS_URL, params={
        'filters[name]': name, 'format': 'json', 'rows': 1
    })
    data = resp.json()
    if data['response']['numFound'] > 0:
        hit = data['response']['docs'][0]
        birth = int(hit.get('birthyear', 0) or 0)
        death = int(hit.get('deathyear', 0) or 0)
        return {'found': True, 'rkd_name': hit['name'],
                'rkd_id': hit['priref'], 'birth_year': birth,
                'death_year': death, 'born_after_1900': birth >= 1900}
    return {'found': False, 'rkd_name': None, 'rkd_id': None,
            'birth_year': None, 'death_year': None, 'born_after_1900': False}
    """
    return {'found': False, 'rkd_name': None, 'rkd_id': None,
            'birth_year': None, 'death_year': None, 'born_after_1900': False}


def lookup_rkd_artwork(artist_name: str, title: str) -> dict:
    """
    Query RKDimages for a matching artwork record.
    TO ACTIVATE: uncomment the block below and remove the stub return.

    resp = requests.get(RKD_IMAGES_URL, params={
        'filters[naam]': artist_name,
        'filters[title]': title,
        'format': 'json', 'rows': 1
    })
    data = resp.json()
    if data['response']['numFound'] > 0:
        hit = data['response']['docs'][0]
        return {'found': True, 'rkd_number': hit.get('priref', '')}
    return {'found': False, 'rkd_number': None}
    """
    return {'found': False, 'rkd_number': None}

# ============================================================
# 4. PARSE ARTIST NAME & BIRTH/DEATH FROM TITLE FIELD
# ============================================================

def parse_title_field(raw_title: str) -> dict:
    """
    Handles all auction Title formats, e.g.:
      "Evert Moll (1878-1955)"
      "Attributed to Jan van Cleve III (1646-1716)"
      "Circle of Abraham van Beijeren (1620/21-1690)"
      "Dutch School (19th Century)"
      "Theo van Hoytema (1863-1917), 'Het leelijke jonge eendje', 1893"
    """
    raw = str(raw_title).strip()

    # Detect attribution qualifiers
    qualifier = ''
    qual_map = {
        'Attributed to': 'toegeschreven aan',
        'Circle of':     'omgeving van',
        'Follower of':   'navolger van',
        'After':         'naar',
        'Manner of':     'wijze van',
        'Style of':      'wijze van',
    }
    for eng, dutch in qual_map.items():
        if raw.lower().startswith(eng.lower()):
            qualifier = dutch
            raw = raw[len(eng):].strip()
            break

    # Remove trailing ", 'Title', year" info
    raw = re.sub(r",\s*'[^']*'.*$", '', raw).strip()

    # Extract birth-death from parentheses
    birth, death = None, None
    paren = re.search(
        r'\((?:[^,)]+,\s*)?(?:circa\s*|ca\.\s*)?'
        r'(\d{3,4})(?:[/\-]\d{1,4})?'
        r'(?:\s*[-]\s*(?:circa\s*)?(\d{3,4}))?\)',
        raw
    )
    if paren:
        birth = int(paren.group(1)) if paren.group(1) else None
        death = int(paren.group(2)) if paren.group(2) else None

    # Clean name - everything before first parenthesis
    name = re.sub(r'\s*\(.*', '', raw).strip()

    # Detect school/anonymous entries
    is_school = bool(re.search(
        r'\b(Dutch|Flemish|French|German|Italian|European|Northern Italian'
        r'|Indonesian|Pita Maha|African)\s+School\b|\bSchool\b',
        name, re.IGNORECASE
    ))
    if is_school:
        name = 'Anoniem'

    return {
        'clean_name': name,
        'birth_year': birth,
        'death_year': death,
        'qualifier':  qualifier,
        'is_school':  is_school,
    }

# ============================================================
# 5. DIMENSIONS
# ============================================================

def extract_dimensions(text: str) -> dict:
    """
    Extracts the first (non-frame) HxW pair from description.
    Handles European comma decimals: '60,5 x 101 cm'.
    Ignores pairs followed by '(... incl. frame)'.
    """
    num     = r'(\d{1,4}(?:[,\.]\s?\d{1,2})?)'
    sep     = r'\s*[x]\s*'
    pattern = re.compile(
        num + sep + num + r'\s*(cm|mm)?(?!\s*cm\s*\()',
        re.IGNORECASE
    )
    for m in pattern.finditer(str(text)):
        try:
            h    = float(m.group(1).replace(',', '.').replace(' ', ''))
            w    = float(m.group(2).replace(',', '.').replace(' ', ''))
            unit = (m.group(3) or 'cm').lower()
            shape = ('liggende rechthoek' if w > h else
                     'staande rechthoek'  if h > w else 'vierkant')
            return {'height': h, 'width': w, 'unit': unit, 'shape': shape}
        except ValueError:
            continue
    return {'height': None, 'width': None, 'unit': 'cm', 'shape': 'onbekend'}

# ============================================================
# 6. DATE EXTRACTION & RKD FORMATTING
# ============================================================

def extract_and_format_date(description: str, title: str,
                             birth_year, death_year,
                             born_after_1900: bool) -> dict:
    """
    Outputs RKD-formatted date fields per the manual, plus signals
    used by the confidence scorer.
    """
    desc          = str(description)
    sig_dated     = False
    used_lifespan = False
    circa         = False

    # 1. Signed & dated (highest confidence)
    sig_match = re.search(
        r"signed.*?['\"].*?(?:/|\s)(\d{4})['\"]", desc, re.IGNORECASE)
    if not sig_match:
        sig_match = re.search(
            r"dated\s+['\"]?(?:[A-Za-z\s./]*)(\d{4})", desc, re.IGNORECASE)

    # 2. Year appended to title field: "..., 1893"
    title_year = re.search(r',\s*(\d{4})\s*$', str(title))

    # 3. General year anywhere in description
    gen_year = re.search(r'\b(1[4-9]\d{2}|20[0-2]\d)\b', desc)

    year = None
    if sig_match:
        year      = int(sig_match.group(1))
        sig_dated = True
    elif title_year:
        year = int(title_year.group(1))
    elif gen_year:
        year = int(gen_year.group(1))

    date_note = ''

    if year:
        circa_match = re.search(
            r'(circa|ca\.|c\.)\s*' + str(year), desc, re.IGNORECASE)
        circa = bool(circa_match)
        if circa:
            date_dutch   = f'ca. {year}'
            date_english = f'c. {year}'
            margin_begin = year - 5
            margin_end   = year + 5
        else:
            date_dutch   = f'{year} gedateerd'
            date_english = f'dated {year}'
            margin_begin = year
            margin_end   = year
    else:
        date_dutch   = ''
        date_english = ''
        if birth_year and death_year:
            offset        = 20 if born_after_1900 else 15
            margin_begin  = birth_year + offset
            margin_end    = death_year
            date_note     = 'active'
            used_lifespan = True
        elif birth_year:
            offset        = 20 if born_after_1900 else 15
            margin_begin  = birth_year + offset
            margin_end    = ''
            date_note     = 'active'
            used_lifespan = True
        else:
            margin_begin = ''
            margin_end   = ''
            date_note    = 'NEEDS REVIEW: no date and no artist lifespan found'

    return {
        'date_dutch':     date_dutch,
        'date_english':   date_english,
        'margin_begin':   margin_begin,
        'margin_end':     margin_end,
        'date_note':      date_note,
        'sig_dated':      sig_dated,
        'used_lifespan':  used_lifespan,
        'circa':          circa,
    }

# ============================================================
# 7. GENRE DETECTION
# ============================================================

def detect_genres(description: str, title_field: str) -> tuple:
    """Returns (genres list, total keyword hit count)."""
    text          = (str(description) + ' ' + str(title_field)).lower()
    matched       = []
    total_kw_hits = 0
    for genre, keywords in GENRE_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text)
        if hits > 0:
            matched.append((genre, hits))
            total_kw_hits += hits
    matched.sort(key=lambda x: -x[1])
    genres = [g for g, _ in matched] if matched else ['undetermined']
    return genres, total_kw_hits

# ============================================================
# 8. OBJECT NAME
# ============================================================

def detect_object_name(description: str) -> str:
    desc_lower = str(description).lower()
    for keywords, obj_name in OBJECT_NAME_MAP:
        if any(kw in desc_lower for kw in keywords):
            return obj_name
    return 'painting'

# ============================================================
# 9. CORE PIPELINE
# ============================================================

def build_image_lookup(images_dir: str) -> dict:
    """Returns {lot_no (int): full image path} for all *-1.* files."""
    lookup = {}
    if not os.path.isdir(images_dir):
        return lookup
    for fname in os.listdir(images_dir):
        name, _ = os.path.splitext(fname)
        parts = name.split('-')
        if len(parts) == 2 and parts[1] == '1' and parts[0].isdigit():
            lookup[int(parts[0])] = os.path.join(images_dir, fname)
    return lookup


def process_auction_metadata(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    image_lookup = build_image_lookup(IMAGES_DIR)

    for _, row in df.iterrows():
        desc      = str(row.get('Description', ''))
        title_raw = str(row.get('Title', ''))
        lot_no    = row.get('Lot No', '')

        # --- Parse artist from Title ---
        parsed       = parse_title_field(title_raw)
        clean_name   = parsed['clean_name']
        birth_year   = parsed['birth_year']
        death_year   = parsed['death_year']
        qualifier    = parsed['qualifier']
        is_school    = parsed['is_school']

        # --- RKD artist lookup ---
        rkd_artist       = lookup_rkd_artist(clean_name)
        rkd_artist_found = rkd_artist['found']
        if rkd_artist_found:
            final_name      = rkd_artist['rkd_name']
            birth_year      = rkd_artist['birth_year'] or birth_year
            death_year      = rkd_artist['death_year'] or death_year
            born_after_1900 = rkd_artist['born_after_1900']
        elif is_school or clean_name == 'Anoniem':
            final_name      = 'Anoniem'
            born_after_1900 = False
        else:
            final_name      = 'Anoniem'
            born_after_1900 = (birth_year or 0) >= 1900

        # --- RKD artwork lookup ---
        rkd_artwork       = lookup_rkd_artwork(clean_name, title_raw)
        rkd_artwork_found = rkd_artwork['found']
        artwork_number    = rkd_artwork['rkd_number'] or ''

        # --- Dates ---
        date_info = extract_and_format_date(
            desc, title_raw, birth_year, death_year, born_after_1900)

        # --- Dimensions ---
        dims = extract_dimensions(desc)

        # --- Genre ---
        genres, num_kw_matched = detect_genres(desc, title_raw)

        # --- Object name ---
        obj_name = detect_object_name(desc)

        # --- Confidence scores ---
        conf = compute_confidence(
            rkd_artwork_found = rkd_artwork_found,
            rkd_artist_found  = rkd_artist_found,
            qualifier         = qualifier,
            raw_name          = clean_name,
            is_school         = is_school,
            date_dutch        = date_info['date_dutch'],
            sig_dated         = date_info['sig_dated'],
            used_lifespan     = date_info['used_lifespan'],
            circa             = date_info['circa'],
            genres            = genres,
            num_kw_matched    = num_kw_matched,
            obj_name          = obj_name,
            desc              = desc,
            height            = dims['height'],
            width             = dims['width'],
        )

        record = {
            # --- Mandatory RKD fields ---
            'Artwork number (%0)':       artwork_number,
            'Status (sz)':               'huidig',
            'Name (na)':                 final_name,
            'Kwalificatie (nw)':         qualifier,
            'Date Dutch (od)':           date_info['date_dutch'],
            'Date English (oe)':         date_info['date_english'],
            'Search margin: begin (bv)': date_info['margin_begin'],
            'Search margin: end (ev)':   date_info['margin_end'],
            'Date remark':               date_info['date_note'],
            'Genre (gt)':                ' | '.join(genres),
            'Subject keyword (ot)':      '',   # blank - requires expert review
            'Object name (oj)':          obj_name,
            'Shape (vo)':                dims['shape'],
            'Height':                    dims['height'],
            'Width':                     dims['width'],
            'Unit (ee)':                 dims['unit'],
            # --- Confidence scores ---
            'Confidence (%)':            conf['confidence_pct'],
            'Conf: artwork number':      conf['conf_artwork_number'],
            'Conf: artist name':         conf['conf_artist_name'],
            'Conf: date':                conf['conf_date'],
            'Conf: genre':               conf['conf_genre'],
            'Conf: object name':         conf['conf_object_name'],
            'Conf: dimensions':          conf['conf_dimensions'],
            'Conf: subject keyword':     conf['conf_subject_kw'],
            'Confidence breakdown':      conf['confidence_breakdown'],
            # --- Reference columns ---
            'Image path':                os.path.relpath(image_lookup[int(float(lot_no))], BASE_DIR) if lot_no != '' and int(float(lot_no)) in image_lookup else '',
            'Lot No (source)':           lot_no,
            'Extracted_artist_raw':      clean_name,
            'Original_Title':            title_raw,
            'Original_Description':      desc,
        }
        records.append(record)

    return pd.DataFrame(records)

# ============================================================
# 10. EXECUTION
# ============================================================

print("Loading data...")
df_raw = pd.read_excel(AUCTION_FILE)

print(f"Processing {len(df_raw)} records...")
final_df = process_auction_metadata(df_raw)

output_file = os.path.join(BASE_DIR, 'data', 'RKD_Structured_Metadata.csv')
final_df.to_csv(output_file, index=False)
print(f"\nSaved: {output_file}")

# --- Summary statistics ---
total = len(final_df)
conf  = final_df['Confidence (%)']

print(f"\n--- Confidence Score Summary ---")
print(f"Total records:        {total}")
print(f"Mean confidence:      {conf.mean():.1f}%")
print(f"Median confidence:    {conf.median():.1f}%")
print(f"High  (>=80%):        {(conf >= 80).sum()} records")
print(f"Medium (50-79%):      {((conf >= 50) & (conf < 80)).sum()} records")
print(f"Low   (<50%):         {(conf < 50).sum()} records  <- prioritise for human review")

print(f"\n--- Lowest confidence records (review first) ---")
low = final_df.nsmallest(10, 'Confidence (%)')
print(low[['Lot No (source)', 'Name (na)', 'Confidence (%)',
           'Conf: artwork number', 'Conf: artist name',
           'Conf: date', 'Confidence breakdown']].to_string())