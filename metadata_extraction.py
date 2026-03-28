import os
import pandas as pd
import re
import requests

# ============================================================
# 0. CONFIGURATION
# ============================================================

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
AUCTION_FILE = os.path.join(BASE_DIR, 'data', 'dataset_A_metadata.xlsx')
IMAGES_DIR   = os.path.join(BASE_DIR, 'data', 'dataset_A_images')

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
      'colour lithograph'],              'print'),
    (['bronze', 'paper-mache', 'papier-mache'],                         'sculpture'),
    (['mixed media'],                                    'needs human review'),
    (['acrylic on canvas', 'acrylic'],                   'painting'),
    (['album', 'portfolio'],                             'album'),
]


# ============================================================
# 2. CONFIDENCE SCORING FRAMEWORK
# ============================================================

FIELD_WEIGHTS = {
    'artist_name': 20,
    'date':        20,
    'genre':       20,
    'dimensions':  20,
    'object_name': 20,
}

def score_artwork_number(rkd_found: bool) -> tuple:
    if rkd_found:
        return FIELD_WEIGHTS['artwork_number'], 'RKD match confirmed'
    return 0, 'No RKD match found - human must assign'

def score_artist_name(qualifier: str, raw_name: str, is_school: bool) -> tuple:
    w = FIELD_WEIGHTS['artist_name']
    if is_school or raw_name == 'Anoniem':
        return round(w * 0.5), 'Anonymous/school entry - attribution inherently uncertain'
    if qualifier:
        return round(w * 0.6), f'Attribution qualifier present: {qualifier}'
    return w, 'Artist name taken from auction record'

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
    if obj_name == 'needs human review':
        return 0, 'Object type undetermined - needs human review'
    direct_terms = ['oil on canvas', 'oil on panel', 'oil on board',
                    'watercolour', 'gouache', 'etching', 'engraving',
                    'lithograph', 'bronze', 'pencil on paper', 'charcoal']
    if any(t in desc.lower() for t in direct_terms):
        return w, f'Medium explicitly stated -> {obj_name}'
    return round(w * 0.7), f'Medium inferred -> {obj_name}'

def score_dimensions(height, width, is_3d: bool = False) -> tuple:
    w = FIELD_WEIGHTS['dimensions']
    if is_3d:
        return 0, '3D object - dimensions need human review'
    if height and width:
        return w, 'Both dimensions extracted'
    if height or width:
        return round(w * 0.5), 'Only one dimension extracted'
    return 0, 'No dimensions found'

def compute_confidence(qualifier, raw_name, is_school, date_dutch, sig_dated,
                       used_lifespan, circa, genres, num_kw_matched,
                       obj_name, desc, height, width, is_3d) -> dict:
    s_artist, n_artist = score_artist_name(qualifier, raw_name, is_school)
    s_date,   n_date   = score_date(date_dutch, sig_dated,
                                    used_lifespan, circa)
    s_genre,  n_genre  = score_genre(genres, num_kw_matched)
    s_obj,    n_obj    = score_object_name(obj_name, desc)
    s_dims,   n_dims   = score_dimensions(height, width, is_3d)

    total = s_artist + s_date + s_genre + s_obj + s_dims

    breakdown = (
        f"artist:{s_artist}/{FIELD_WEIGHTS['artist_name']} ({n_artist}) | "
        f"date:{s_date}/{FIELD_WEIGHTS['date']} ({n_date}) | "
        f"genre:{s_genre}/{FIELD_WEIGHTS['genre']} ({n_genre}) | "
        f"object:{s_obj}/{FIELD_WEIGHTS['object_name']} ({n_obj}) | "
        f"dims:{s_dims}/{FIELD_WEIGHTS['dimensions']} ({n_dims})"
    )

    return {
        'confidence_pct':       total,
        'confidence_breakdown': breakdown,
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
    If three dimensions are found (HxWxD), the object is 3D and flagged for
    human review rather than attempting to extract 2D measurements.
    """
    num     = r'(\d{1,4}(?:[,\.]\s?\d{1,2})?)'
    sep     = r'\s*[x]\s*'
    # Pattern for two dimensions followed by an optional third (3D check)
    pattern = re.compile(
        num + sep + num + r'(?:' + sep + num + r')?' + r'\s*(cm|mm)?(?!\s*cm\s*\()',
        re.IGNORECASE
    )
    for m in pattern.finditer(str(text)):
        try:
            # If a third dimension is captured, it's a 3D object
            if m.group(3) and m.group(3).replace(',', '.').replace(' ', '').replace('.', '').isdigit():
                return {'height': None, 'width': None, 'unit': 'cm',
                        'shape': 'needs human review', 'is_3d': True}
            h    = float(m.group(1).replace(',', '.').replace(' ', ''))
            w    = float(m.group(2).replace(',', '.').replace(' ', ''))
            unit = (m.group(4) or 'cm').lower()
            shape = ('liggende rechthoek' if w > h else
                     'staande rechthoek'  if h > w else 'vierkant')
            return {'height': h, 'width': w, 'unit': unit, 'shape': shape,
                    'is_3d': False}
        except ValueError:
            continue
    return {'height': None, 'width': None, 'unit': 'cm', 'shape': 'onbekend',
            'is_3d': False}

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

def _extract_dated_year(description: str):
    """Return the 4-digit year following an explicit 'dated' marker, or None."""
    m = re.search(r'\bdated\b[^0-9\n]{0,30}?(\d{4})\b', description, re.IGNORECASE)
    return int(m.group(1)) if m else None


# ============================================================
# 7. GENRE DETECTION
# ============================================================

def _strip_provenance(description: str) -> str:
    """Remove provenance/literature/exhibition sections from description text.
    These sections often contain city/gallery names that trigger false genre matches."""
    markers = re.compile(
        r'\b(provenance|literature|exhibited|exhibition|bibliography)\b',
        re.IGNORECASE
    )
    m = markers.search(description)
    return description[:m.start()] if m else description


def detect_genres(description: str, title_field: str) -> tuple:
    """Returns (genres list, total keyword hit count).
    Provenance is stripped from the description before matching to avoid
    false genre hits from location names in provenance text."""
    subject_desc = _strip_provenance(str(description))
    text         = (subject_desc + ' ' + str(title_field)).lower()
    matched       = []
    total_kw_hits = 0
    for genre, keywords in GENRE_KEYWORDS.items():
        hits = sum(1 for kw in keywords
                   if re.search(r'\b' + re.escape(kw) + r'\b', text))
        if hits > 0:
            matched.append((genre, hits))
            total_kw_hits += hits
    matched.sort(key=lambda x: -x[1])
    genres = [g for g, _ in matched] if matched else ['undetermined']
    return genres, total_kw_hits


# ============================================================
# 8. PROVENANCE, EXHIBITION & SIGNATURE EXTRACTION
# ============================================================

_SECTION_BOUNDARY = re.compile(
    r'\b(Provenance|Exhibited|Literature|Bibliography)\s*:',
    re.IGNORECASE
)


def extract_provenance(description: str) -> list:
    """Returns a list of provenance entries. Entries are split on ' - ' followed
    by a capital letter (the auction catalogue convention for new listings)."""
    match = re.search(r'\bProvenance\s*:\s*', description, re.IGNORECASE)
    if not match:
        return ['no provenance found']
    text = description[match.end():]
    stop = _SECTION_BOUNDARY.search(text)
    if stop:
        text = text[:stop.start()]
    text = text.strip()
    if not text:
        return ['no provenance found']
    entries = re.split(r' - (?=[A-Z])', text)
    return [e.strip() for e in entries if e.strip()]


def extract_exhibition(description: str) -> str:
    match = re.search(r'\bExhibited\s*:\s*', description, re.IGNORECASE)
    if not match:
        return 'no exhibition found'
    text = description[match.end():]
    stop = _SECTION_BOUNDARY.search(text)
    if stop:
        text = text[:stop.start()]
    return text.strip() or 'no exhibition found'


def extract_signature_info(description: str) -> dict:
    result = {'inscription': '', 'location': ''}

    signed_matches = list(re.finditer(r'\bsigned\b', description, re.IGNORECASE))
    if not signed_matches:
        return result

    # Inscription and location: segment after first 'signed' up to next comma
    first_end = signed_matches[0].end()
    comma_pos = description.find(',', first_end)
    segment = description[first_end: comma_pos if comma_pos != -1 else len(description)]

    quote_match = re.search(r"'[^']*'", segment)
    if quote_match:
        result['inscription'] = quote_match.group(0)

    bracket_match = re.search(r'\(([^)]*)\)', segment)
    if bracket_match:
        result['location'] = bracket_match.group(1)

    return result


# ============================================================
# 10. OBJECT NAME
# ============================================================


def detect_object_name(description: str) -> str:
    desc_lower = str(description).lower()
    for keywords, obj_name in OBJECT_NAME_MAP:
        if any(kw in desc_lower for kw in keywords):
            return obj_name
    return 'needs human review'

# ============================================================
# 10. CORE PIPELINE
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

        # --- Artist name (no RKD lookup) ---
        final_name      = clean_name
        born_after_1900 = (birth_year or 0) >= 1900

        # --- RKD artwork lookup ---
        rkd_artwork    = lookup_rkd_artwork(clean_name, title_raw)
        artwork_number = rkd_artwork['rkd_number'] or ''

        # --- Dates ---
        date_info = extract_and_format_date(
            desc, title_raw, birth_year, death_year, born_after_1900)

        # --- Dimensions ---
        dims = extract_dimensions(desc)

        # --- Genre ---
        genres, num_kw_matched = detect_genres(desc, title_raw)

        # --- Object name ---
        obj_name = detect_object_name(desc)

        # --- Provenance, exhibition & signature ---
        provenance = extract_provenance(desc)
        exhibition = extract_exhibition(desc)
        sig_info   = extract_signature_info(desc)

        # --- Validate / fill date fields from explicit 'dated' marker ---
        dated_year     = _extract_dated_year(desc)
        pre_sig_dated  = date_info['sig_dated']
        if dated_year:
            if not date_info['date_dutch']:
                # Date was unknown — fill from 'dated' inscription
                date_info.update({
                    'date_dutch':   f'{dated_year} gedateerd',
                    'date_english': f'dated {dated_year}',
                    'margin_begin': dated_year,
                    'margin_end':   dated_year,
                    'sig_dated':    True,
                })

        # --- Confidence scores ---
        conf = compute_confidence(
            qualifier      = qualifier,
            raw_name       = clean_name,
            is_school      = is_school,
            date_dutch     = date_info['date_dutch'],
            sig_dated      = date_info['sig_dated'],
            used_lifespan  = date_info['used_lifespan'],
            circa          = date_info['circa'],
            genres         = genres,
            num_kw_matched = num_kw_matched,
            obj_name       = obj_name,
            desc           = desc,
            height         = dims['height'],
            width          = dims['width'],
            is_3d          = dims['is_3d'],
        )

        # Boost confidence when 'dated' validates a year that wasn't from a signature
        if (dated_year and not pre_sig_dated
                and date_info.get('margin_begin') == dated_year):
            conf['confidence_pct'] = min(100, conf['confidence_pct'] + 5)
            conf['confidence_breakdown'] += ' | +5: "dated" inscription validates year'

        record = {
            # --- Mandatory RKD fields ---
            'Artwork number':        artwork_number,
            'Status':                'huidig',
            'Artist':                final_name,
            'Kwalificatie':          qualifier,
            'Date Dutch':            date_info['date_dutch'],
            'Date English':          date_info['date_english'],
            'Search margin: begin':  date_info['margin_begin'],
            'Search margin: end':    date_info['margin_end'],
            'Date remark':           date_info['date_note'],
            'Genre':                 ' | '.join(genres),
            'Subject keyword':       '',
            'Object name':           obj_name,
            'Shape':                 dims['shape'],
            'Height':                '' if dims['is_3d'] else dims['height'],
            'Width':                 '' if dims['is_3d'] else dims['width'],
            'Unit':                  dims['unit'],
            # --- Confidence scores ---
            'Confidence (%)':        conf['confidence_pct'],
            'Confidence breakdown':  conf['confidence_breakdown'],
            # --- Reference columns ---
            'Image path':            os.path.relpath(image_lookup[int(float(lot_no))], BASE_DIR) if lot_no != '' and int(float(lot_no)) in image_lookup else '',
        }
        # Provenance between Image path and Exhibition, extras in numbered columns
        record['Provenance'] = provenance[0]
        for i, entry in enumerate(provenance[1:], start=2):
            record[f'Provenance ({i})'] = entry
        record['Exhibition']           = exhibition
        record['Signature/inscription'] = sig_info['inscription']
        record['Signature location']   = sig_info['location']
        record['Lot No (source)']      = lot_no
        record['Original_Title']       = title_raw
        record['Original_Description'] = desc
        records.append(record)

    return pd.DataFrame(records)

# ============================================================
# 10. EXECUTION
# ============================================================

print("Loading data...")
df_raw = pd.read_excel(AUCTION_FILE)

print(f"Processing {len(df_raw)} records...")
final_df = process_auction_metadata(df_raw)

output_file = os.path.join(BASE_DIR, 'data', 'dataset_A_metadata_results.csv')
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
print(low[['Lot No (source)', 'Artist', 'Confidence (%)',
           'Confidence breakdown']].to_string())