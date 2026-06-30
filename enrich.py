"""Catalog enrichment.

The source catalog only has 11 columns and most shopping attributes live in free
text. But ``product_url`` encodes Myntra's full structured title:

    https://www.myntra.com/{TYPE}/{BRAND}/{rich-descriptive-slug}/{id}/buy

This module mines the URL (path = real product type, slug = rich attributes) plus
``product_name`` to manufacture structured, filterable columns ONCE, offline:

    product_type, gender (rewrites `category`), color, material, fit, pattern

All matching reuses ``taxonomy.word_pattern`` (whole-word, plural-tolerant) so e.g.
"cap" never matches "cappuccino". Operations are vectorised for the 100k catalog.
"""

import re

import pandas as pd

import taxonomy

# ---------------------------------------------------------------------------
# Attribute vocabularies (curated, lowercase). Multi-word entries first so the
# longest match wins (e.g. "navy blue" before "blue").
# ---------------------------------------------------------------------------

COLORS = {
    "off white", "navy blue", "sea green", "sky blue", "light blue", "dark blue",
    "black", "white", "blue", "navy", "red", "green", "yellow", "orange", "pink",
    "purple", "grey", "gray", "brown", "beige", "maroon", "olive", "teal", "tan",
    "cream", "gold", "silver", "khaki", "mustard", "magenta", "turquoise",
    "lavender", "peach", "coral", "burgundy", "rust", "charcoal", "rose", "wine",
    "mint", "lime", "indigo", "violet", "fuchsia", "multicoloured", "multicolored",
    "multicolour", "multicolor",
}

MATERIALS = {
    "faux leather", "pure cotton", "cotton", "silk", "leather", "denim", "wool",
    "linen", "polyester", "nylon", "georgette", "chiffon", "velvet", "satin",
    "rayon", "suede", "jute", "canvas", "net", "lace", "khadi", "modal",
    "viscose", "crepe", "organza", "fleece", "corduroy", "cashmere", "acrylic",
    "spandex", "lycra", "mesh",
}

FITS = {
    "super skinny", "wide leg", "mid rise", "high rise", "low rise", "boot cut",
    "slim", "skinny", "regular", "relaxed", "tapered", "straight", "oversized",
    "loose", "bootcut", "cropped", "boxy", "fitted", "flared", "baggy",
}

PATTERNS = {
    "colour blocked", "color blocked", "colourblocked", "colorblocked",
    "self design", "tie dyed", "solid", "printed", "striped", "checked",
    "checkered", "floral", "polka", "graphic", "embroidered", "washed", "ripped",
    "textured", "ribbed", "typography", "camouflage", "camo", "embellished",
    "sequinned", "sequined", "woven", "knitted", "dyed", "ombre", "geometric",
    "abstract",
}

# Gender token -> canonical category. Ordered longest-first in the regex below.
_GENDER_MAP = {
    "womens": "Women", "women": "Women", "girls": "Women", "girl": "Women",
    "ladies": "Women",
    "mens": "Men", "men": "Men", "man": "Men", "boys": "Men", "boy": "Men",
    "kids": "Kids", "kid": "Kids", "infants": "Kids", "infant": "Kids",
    "babies": "Kids", "baby": "Kids",
    "unisex": "Unisex",
}
_GENDER_RE = r"\b(womens|women|girls|girl|ladies|mens|men|man|boys|boy|kids|kid|infants|infant|babies|baby|unisex)\b"

_ENRICHED_COLS = ("product_type", "color", "material", "fit", "pattern")

# Which enriched column a query token belongs to (for structured filtering).
_TOKEN_COLUMN = {}
for _v in COLORS:
    _TOKEN_COLUMN[_v] = "color"
for _v in MATERIALS:
    _TOKEN_COLUMN[_v] = "material"
for _v in FITS:
    _TOKEN_COLUMN[_v] = "fit"
for _v in PATTERNS:
    _TOKEN_COLUMN[_v] = "pattern"
# also index individual words of multi-word vocab terms
for _vocab, _col in ((COLORS, "color"), (MATERIALS, "material"), (FITS, "fit"), (PATTERNS, "pattern")):
    for _v in _vocab:
        for _w in _v.split():
            _TOKEN_COLUMN.setdefault(_w, _col)


def attribute_column(token: str) -> str | None:
    """Return the enriched column a query token filters on ('color'/'material'/
    'fit'/'pattern'), or None if it's not a known attribute word."""
    return _TOKEN_COLUMN.get(str(token).lower().strip())


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------

# myntra.com/{type}/{brand}/{slug}/...  -> capture type (1) and descriptive slug (2)
_URL_RE = re.compile(r"myntra\.com/([^/]+)/[^/]+/([^/?]+)", re.IGNORECASE)


def parse_url(url: str) -> tuple[str, str]:
    """Return (url_type, descriptive_slug) from a Myntra product URL, or ('', '')."""
    m = _URL_RE.search(str(url))
    if not m:
        return "", ""
    return m.group(1).lower(), m.group(2).lower()


def _normalise_type(url_type: str) -> str:
    """Singularise a URL type segment: 'tshirts'->'tshirt', 'nail-polish'->'nail polish'."""
    words = url_type.replace("-", " ").split()
    return " ".join(taxonomy.singularize(w) for w in words)


# ---------------------------------------------------------------------------
# Vectorised attribute extraction
# ---------------------------------------------------------------------------

def _extract_first(text: pd.Series, vocab: set[str]) -> pd.Series:
    """First whole-word vocab hit per row (longest term wins), else ''."""
    ordered = sorted(vocab, key=lambda v: (-len(v), v))
    pattern = r"\b(" + "|".join(re.escape(v) for v in ordered) + r")s?\b"
    return text.str.extract(pattern, flags=re.IGNORECASE, expand=False).fillna("")


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add structured columns derived from product_url + product_name.

    Idempotent: if the enriched columns already exist (e.g. persisted by the
    preprocessing step) the frame is returned unchanged.
    """
    if all(c in df.columns for c in _ENRICHED_COLS) and "product_name" in df.columns:
        return df
    if "product_name" not in df.columns:
        return df

    out = df.copy()

    # --- parse URL (type + slug) ---
    if "product_url" in out.columns:
        parsed = out["product_url"].astype(str).str.extract(_URL_RE)
        url_type = parsed[0].fillna("").str.lower()
        slug = parsed[1].fillna("").str.lower().str.replace("-", " ", regex=False)
    else:
        url_type = pd.Series("", index=out.index)
        slug = pd.Series("", index=out.index)

    # Rich text used for attribute extraction = name + URL slug.
    text = (out["product_name"].astype(str) + " " + slug).str.lower()

    # --- product_type (from URL, fallback to name taxonomy) ---
    type_from_url = url_type.map(_normalise_type)
    type_from_name = out["product_name"].astype(str).map(taxonomy.derive_product_type)
    out["product_type"] = type_from_url.where(type_from_url.str.len() > 0, type_from_name)
    out.loc[out["product_type"] == "other", "product_type"] = type_from_name[out["product_type"] == "other"]

    # --- gender -> category (URL slug is far more reliable than the name guess) ---
    gtok = text.str.extract(_GENDER_RE, flags=re.IGNORECASE, expand=False).fillna("").str.lower()
    gender = gtok.map(_GENDER_MAP).fillna("")
    fallback_cat = out["category"] if "category" in out.columns else "Unisex"
    out["category"] = gender.where(gender.str.len() > 0, fallback_cat)

    # --- soft attributes ---
    out["color"] = _extract_first(text, COLORS)
    out["material"] = _extract_first(text, MATERIALS)
    out["fit"] = _extract_first(text, FITS)
    out["pattern"] = _extract_first(text, PATTERNS)

    return out
