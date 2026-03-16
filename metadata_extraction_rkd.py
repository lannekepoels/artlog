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

# Genre → subject keyword classification term (Dutch RKD)
GENRE_TO_SUBJECT_KW = {
    'portrait':                  'portret',
    'still life':                'stilleven',
    'landscape (genre)':         'landschap',
    'marine (genre)':            'marine',
    'cityscape':                 'stadsgezicht',
    'interior view':             'interieur',
    'church interior':           'kerkinterieur',
    'history (genre)':           'historie (genre)',
    'genre':                     'genre',
    'figure':                    'figuurstuk',
    'animal painting (genre)':   'dierenstuk',
    'abstraction':               'abstractie',
    'design (artistic concept)': 'ontwerp',
    'study (visual work)':       'studie',
    'architecture (genre)':      'architectuurstuk',
}

# Portrait: figure-type patterns (checked in order; first match wins)
PORTRAIT_TYPE_PATTERNS = [
    (['self-portrait', 'zelfportret'],               'zelfportret'),
    (['double portrait', 'pair of portraits'],        'dubbelportret'),
    (['group portrait'],                              'groepsportret'),
    (['child', 'boy', 'girl'],                        'kinderportret'),
    (['woman', 'female', 'lady', 'dame'],             'vrouwenportret'),
    (['man', 'male', 'gentleman'],                    'mansportret'),
]

# Portrait: attitude/position patterns (all matches collected, in manual order)
PORTRAIT_POSITION_PATTERNS = [
    (['bust', 'bust-length', 'head and shoulders',
      'head-and-shoulders'],                          'schouderstuk (figuurdeel)'),
    (['half-length', 'half length', 'to the waist',
      'waist-length'],                                'ten halven lijve'),
    (['three-quarter', 'three quarter'],              'driekwart figuur'),
    (['full-length', 'full length'],                  'ten voeten uit'),
    (['seated', 'sitting'],                           'zittend'),
    (['facing left', 'looking left', 'turned to the left',
      'to his left', 'to her left'],                  'hoofd naar links'),
    (['facing right', 'looking right', 'turned to the right',
      'to his right', 'to her right'],                'hoofd naar rechts'),
    (['facing viewer', 'facing forward', 'looking out',
      'full face', 'en face'],                        'aanziend'),
]

# Portrait: clothing/attribute patterns (all matches collected)
PORTRAIT_ATTRIBUTE_PATTERNS = [
    (['ruff', 'pleated ruff'],                        'plooikraag'),
    (['lace collar', 'falling collar'],               'kanten kraag'),
    (['collar'],                                      'kraag'),
    (['cuffs'],                                       'manchetten'),
    (['cap', 'bonnet', 'coif'],                       'muts'),
    (['hat'],                                         'hoed'),
    (['chain', 'necklace'],                           'ketting (schakels)'),
    (['brooch'],                                      'broche'),
    (['earring', 'earrings'],                         'oorbellen'),
    (['armor', 'armour', 'breastplate'],              'harnas'),
]

# General subject keywords for non-portrait genres
GENERAL_SUBJECT_PATTERNS = [
    # Animals
    (['horse', 'horses'],                             'paard'),
    (['dog', 'dogs', 'hound'],                        'hond'),
    (['cat', 'cats', 'kitten', 'kittens'],            'kat'),
    (['cow', 'cows', 'cattle'],                       'koe'),
    (['sheep', 'lamb', 'lambs'],                      'schaap'),
    (['goat', 'goats'],                               'geit'),
    (['bird', 'birds'],                               'vogel'),
    (['duck', 'ducks'],                               'eend'),
    (['peacock'],                                     'pauw'),
    (['hen', 'chickens', 'rooster'],                  'kip'),
    # Plants / flowers
    (['roses', 'rose'],                               'roos'),
    (['tulips', 'tulip'],                             'tulp'),
    (['anemones', 'anemone'],                         'anemoon'),
    (['bouquet', 'flowers', 'floral'],                'bloemen'),
    (['fruit', 'grapes', 'peaches', 'apples',
      'pears', 'cherries'],                           'vruchten'),
    (['tree', 'trees'],                               'boom'),
    # Objects (still life)
    (['vase'],                                        'vaas'),
    (['jug', 'pitcher'],                              'kan'),
    (['glass', 'glasses', 'goblet'],                  'glas'),
    (['bowl'],                                        'schaal'),
    (['book', 'books'],                               'boek'),
    (['skull'],                                       'schedel'),
    # Water / marine settings
    (['harbour', 'harbor', 'port'],                   'haven'),
    (['sea', 'ocean', 'waves', 'surf', 'breakers'],   'zee'),
    (['river', 'canal', 'stream'],                    'rivier'),
    (['boat', 'boats', 'vessel'],                     'boot'),
    (['ship', 'ships'],                               'schip'),
    # Landscape settings
    (['forest', 'wood', 'woods'],                     'bos'),
    (['mountain', 'mountains'],                       'berg'),
    (['dunes', 'dune'],                               'duinen'),
    (['meadow', 'field', 'fields'],                   'weiland'),
    (['farm', 'farmhouse', 'farmstead'],              'boerderij'),
    # Urban / buildings
    (['church', 'cathedral'],                         'kerk'),
    (['castle', 'fortress'],                          'kasteel'),
    (['windmill', 'mill'],                            'molen'),
    (['bridge'],                                      'brug'),
    (['street', 'alley'],                             'straat'),
    # Figure / people
    (['peasant', 'farmer'],                           'boer'),
    (['fisherman', 'fisher', 'fishing'],              'visser'),
    (['soldier', 'military', 'officer'],              'militair'),
    (['nude', 'naked'],                               'naakt'),
    (['winter', 'snow', 'ice', 'frozen'],             'winter'),
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

def score_subject_kw(num_kw: int) -> tuple:
    w = FIELD_WEIGHTS['subject_kw']
    if num_kw >= 3:
        return round(w * 0.6), f'{num_kw} keywords auto-extracted - verify against RKD thesaurus'
    if num_kw >= 1:
        return round(w * 0.4), f'{num_kw} keyword(s) auto-extracted - verify against RKD thesaurus'
    return round(w * 0.1), 'No keywords extracted - requires expert review'

def compute_confidence(rkd_artwork_found, rkd_artist_found, qualifier,
                       raw_name, is_school, date_dutch, sig_dated,
                       used_lifespan, circa, genres, num_kw_matched,
                       obj_name, desc, height, width,
                       num_subject_kw=0) -> dict:
    s_art_no, n_art_no = score_artwork_number(rkd_artwork_found)
    s_artist, n_artist = score_artist_name(rkd_artist_found, qualifier,
                                           raw_name, is_school)
    s_date,   n_date   = score_date(date_dutch, sig_dated,
                                    used_lifespan, circa)
    s_genre,  n_genre  = score_genre(genres, num_kw_matched)
    s_obj,    n_obj    = score_object_name(obj_name, desc)
    s_dims,   n_dims   = score_dimensions(height, width)
    s_subj,   n_subj   = score_subject_kw(num_subject_kw)

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
# 8. SUBJECT KEYWORDS
# ============================================================

def extract_subject_keywords(description: str, title_field: str,
                              genres: list) -> list:
    """
    Extracts subject keywords per RKD manual rules:
    1. Classification term derived from the primary detected genre.
    2. For portraits: figure type → attitude/position → clothing/attributes
       (fixed order per manual).
    3. For all other genres: general keyword patterns from description/title.
    Returns a de-duplicated, ordered list of Dutch RKD keyword strings.
    """
    text       = (str(description) + ' ' + str(title_field)).lower()
    primary    = genres[0] if genres else 'undetermined'
    keywords   = []

    # 1. Classification term
    class_term = GENRE_TO_SUBJECT_KW.get(primary)
    if class_term:
        keywords.append(class_term)

    # 2a. Portrait-specific fixed order
    if primary == 'portrait':
        # Figure type (first match only)
        for patterns, term in PORTRAIT_TYPE_PATTERNS:
            if any(p in text for p in patterns):
                keywords.append(term)
                break

        # Attitude / position (all matches, in manual order)
        for patterns, term in PORTRAIT_POSITION_PATTERNS:
            if any(p in text for p in patterns):
                keywords.append(term)

        # Clothing / attributes (all matches)
        for patterns, term in PORTRAIT_ATTRIBUTE_PATTERNS:
            if any(p in text for p in patterns):
                keywords.append(term)

    # 2b. General keywords for non-portrait genres
    else:
        for patterns, term in GENERAL_SUBJECT_PATTERNS:
            if any(p in text for p in patterns):
                keywords.append(term)

    # De-duplicate while preserving order
    seen = set()
    result = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            result.append(kw)
    return result


# ============================================================
# 9. PROVENANCE EXTRACTION
# ============================================================

# Known auction house names (used to classify entries as auction vs. collection)
AUCTION_HOUSES = [
    'christie', 'sotheby', 'bonham', 'phillips', 'dorotheum', 'van ham',
    'lempertz', 'bukowski', 'bruun rasmussen', 'koller', 'de vuyst',
    'drouot', 'cornette de saint cyr', 'tajan', 'artcurial', 'millon',
    'piguet', 'schuler', 'galerie fisher', 'venduehuis', 'glerum',
    'de zon', 'mak van waay', 'sotheby\'s', 'christie\'s',
]

# Currency symbols / abbreviations used in sale price detection
CURRENCY_PATTERN = re.compile(
    r'(?:gbp|hfl|€|£|\$|eur|usd|nl?g\.?|fl\.?)\s*[\d,\.]+|'
    r'[\d,\.]+\s*(?:gbp|hfl|€|£|\$|eur|usd|nl?g\.?|guineas?)',
    re.IGNORECASE
)


def _split_provenance_entries(raw: str) -> list:
    """
    Split a raw provenance block into individual entries.
    Entries are typically separated by semicolons, newlines, or line-break dashes.
    """
    # Normalise line breaks
    raw = raw.replace('\r\n', '\n').replace('\r', '\n')
    # Split on semicolons or newlines that act as separators
    parts = re.split(r';\s*\n?|\n+', raw)
    return [p.strip() for p in parts if p.strip()]


def _classify_entry(entry: str) -> str:
    """Return 'auction', 'collection', or 'unknown'."""
    low = entry.lower()
    if any(ah in low for ah in AUCTION_HOUSES):
        return 'auction'
    if re.search(r'\b(sale|sold|auction|veiling|lot\s*\d+)\b', low):
        return 'auction'
    return 'collection'


def _extract_year(text: str):
    """Return the first 4-digit year found, or None."""
    m = re.search(r'\b(1[4-9]\d{2}|20[0-2]\d)\b', text)
    return int(m.group(1)) if m else None


def _extract_lot(text: str):
    """Return lot number string if present, or None."""
    m = re.search(r'\blot\s*[#no\.]*\s*(\w+)\b', text, re.IGNORECASE)
    return m.group(1) if m else None


def _extract_auction_house(entry: str):
    """Return the first recognised auction house name, or None."""
    low = entry.lower()
    for ah in AUCTION_HOUSES:
        if ah in low:
            # Return title-cased version of what was found in the original text
            idx = low.index(ah)
            return entry[idx: idx + len(ah)].strip(" ,;")
    return None


def extract_provenance(description: str) -> dict:
    """
    Extracts and structures provenance information from an auction description.

    Follows RKDimages field definitions (Screens 6 & 7):
      - Name of collection (cn / collectienaam)
      - Remark collection (cd / opm._verblijfplaats)
      - Date beginning (cb / begindatum_in_collectie)
      - Date end (ce / einddatum_in_collectie)
      - Remark date (ex / opm._datum_uit_collectie)
      - Auction house (yh / veilinghuis-c)
      - Lot number (yl / lotnummer-c)

    Returns a dict with:
      prov_raw         – the raw provenance block extracted from description
      prov_entries     – pipe-separated list of individual provenance statements
      prov_collections – pipe-separated collection/owner names
      prov_dates_begin – pipe-separated begin years
      prov_dates_end   – pipe-separated end years  ('' when unknown)
      prov_remarks     – pipe-separated remark strings
      prov_auction_houses – pipe-separated recognised auction house names
      prov_lot_numbers – pipe-separated lot numbers
      prov_entry_count – total number of entries found
    """
    desc = str(description)

    # --- 1. Isolate the provenance block ---
    # Auction catalogues typically label it "Provenance:", "Provenance :", or
    # "Herkomst:" (Dutch).  Everything up to the next labelled section is kept.
    prov_match = re.search(
        r'(?:Provenance|Herkomst)\s*:?\s*(.*?)(?=\n\s*(?:[A-Z][a-z]+\s*:|\Z)|$)',
        desc, re.IGNORECASE | re.DOTALL
    )
    raw_block = prov_match.group(1).strip() if prov_match else ''

    # Fallback: if no explicit label, try to detect lines that look like
    # provenance (owner name + optional year, or auction house references)
    if not raw_block:
        prov_lines = []
        for line in desc.split('\n'):
            low = line.lower()
            has_ah   = any(ah in low for ah in AUCTION_HOUSES)
            has_sale = bool(re.search(r'\b(sale|sold|auction|lot\s*\d+|collection)\b', low))
            if has_ah or has_sale:
                prov_lines.append(line.strip())
        raw_block = '; '.join(prov_lines)

    if not raw_block:
        return {
            'prov_raw':           '',
            'prov_entries':       '',
            'prov_collections':   '',
            'prov_dates_begin':   '',
            'prov_dates_end':     '',
            'prov_remarks':       '',
            'prov_auction_houses':'',
            'prov_lot_numbers':   '',
            'prov_entry_count':   0,
        }

    # --- 2. Split into individual entries ---
    entries = _split_provenance_entries(raw_block)

    # --- 3. Parse each entry ---
    collections    = []
    dates_begin    = []
    dates_end      = []
    remarks        = []
    auction_houses = []
    lot_numbers    = []

    for entry in entries:
        kind = _classify_entry(entry)
        year = _extract_year(entry)

        if kind == 'auction':
            ah  = _extract_auction_house(entry) or ''
            lot = _extract_lot(entry) or ''
            auction_houses.append(ah)
            lot_numbers.append(lot)
            dates_begin.append(str(year) if year else '')
            dates_end.append('')
            collections.append('')
            # Store the full entry as a remark (per RKD "Remark auction with barcode" field)
            remarks.append(entry)
        else:
            # Collection / owner entry
            # Name: everything before the first date or parenthetical
            name_match = re.match(r'^([^(,\d]+)', entry)
            name = name_match.group(1).strip(' ,;') if name_match else entry
            collections.append(name)
            dates_begin.append(str(year) if year else '')
            dates_end.append('')
            auction_houses.append('')
            lot_numbers.append('')
            # Remainder of entry becomes the remark
            remark = entry[len(name):].strip(' ,;') if name_match else ''
            remarks.append(remark)

    def pipe(lst):
        return ' | '.join(lst)

    return {
        'prov_raw':            raw_block,
        'prov_entries':        pipe(entries),
        'prov_collections':    pipe(collections),
        'prov_dates_begin':    pipe(dates_begin),
        'prov_dates_end':      pipe(dates_end),
        'prov_remarks':        pipe(remarks),
        'prov_auction_houses': pipe(auction_houses),
        'prov_lot_numbers':    pipe(lot_numbers),
        'prov_entry_count':    len(entries),
    }


# ============================================================
# 10. OBJECT NAME
# ============================================================


def detect_object_name(description: str) -> str:
    desc_lower = str(description).lower()
    for keywords, obj_name in OBJECT_NAME_MAP:
        if any(kw in desc_lower for kw in keywords):
            return obj_name
    return 'painting'

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

        # --- Subject keywords ---
        subject_kws = extract_subject_keywords(desc, title_raw, genres)

        # --- Provenance ---
        prov = extract_provenance(desc)

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
            num_subject_kw    = len(subject_kws),
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
            'Subject keyword (ot)':      ' | '.join(subject_kws),
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
            # --- Provenance (Screen 6/7: Herkomst 1/2) ---
            'Prov: raw text':        prov['prov_raw'],
            'Prov: entries':         prov['prov_entries'],
            'Prov: collections (cn)':prov['prov_collections'],
            'Prov: date begin (cb)': prov['prov_dates_begin'],
            'Prov: date end (ce)':   prov['prov_dates_end'],
            'Prov: remarks (cd/ex)': prov['prov_remarks'],
            'Prov: auction houses (yh)': prov['prov_auction_houses'],
            'Prov: lot numbers (yl)':prov['prov_lot_numbers'],
            'Prov: entry count':     prov['prov_entry_count'],
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