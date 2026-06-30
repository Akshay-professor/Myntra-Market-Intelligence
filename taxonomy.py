"""Product-type taxonomy.

The cleaned dataset only carries a coarse `category` (Men/Women/Kids/Unisex)
derived from the product name. This module adds a real *product type* layer
(jeans, tshirt, kurta, shoes, lipstick, ...) so search and classification can
reason about what a product actually is instead of doing blind string matching.

It also exposes:
- ``PRODUCT_VOCAB`` : every keyword that signals a product (used by the classifier).
- ``SYNONYMS``      : common synonym -> canonical keyword map (denim -> jeans, etc.)
                      so a search for "denim" still finds "jeans".
"""

import re

# Ordered (canonical_type, keyword_patterns). ORDER MATTERS: more specific types
# must come before more generic ones (e.g. "sweatshirt"/"tshirt" before "shirt")
# because the first matching rule wins.
_TYPE_RULES: list[tuple[str, list[str]]] = [
    # --- topwear (specific first) ---
    ("tshirt", ["t-shirt", "t shirt", "tshirt", "tee"]),
    ("sweatshirt", ["sweatshirt"]),
    ("sweater", ["sweater", "pullover"]),
    ("hoodie", ["hoodie", "hooded"]),
    ("kurta", ["kurta", "kurti"]),
    ("shirt", ["shirt"]),
    ("top", ["top ", "tank top", "crop top", "tunic", "blouse"]),
    ("jacket", ["jacket", "windcheater"]),
    ("blazer", ["blazer"]),
    ("coat", ["overcoat", "coat"]),
    # --- bottomwear ---
    ("jeans", ["jeans", "jean", "denim"]),
    ("joggers", ["jogger", "track pant", "trackpant"]),
    ("trousers", ["trouser", "chino"]),
    ("shorts", ["shorts"]),
    ("skirt", ["skirt"]),
    ("leggings", ["legging", "jegging"]),
    ("pants", ["pants", "pant"]),
    # --- dresses / ethnic ---
    ("dress", ["dress", "gown", "frock"]),
    ("saree", ["saree", "sari"]),
    ("lehenga", ["lehenga", "lehanga"]),
    ("salwar", ["salwar", "salwar suit", "kurta set"]),
    ("dupatta", ["dupatta"]),
    # --- innerwear / sleep / swim ---
    ("bikini", ["bikini"]),
    ("lingerie", ["lingerie", "babydoll"]),
    ("bra", ["bra ", "bralette", "bras"]),
    ("briefs", ["brief", "panty", "panties"]),
    ("boxers", ["boxer", "trunk"]),
    ("vest", ["vest", "innerwear"]),
    ("nightwear", ["nightwear", "nightdress", "pyjama", "pajama", "night suit"]),
    ("swimwear", ["swimwear", "swimsuit"]),
    # --- footwear ---
    ("sneakers", ["sneaker", "trainers"]),
    ("heels", ["heel", "stiletto", "pump"]),
    ("sandals", ["sandal", "floater"]),
    ("flats", ["flats", "ballerina"]),
    ("boots", ["boots", "boot"]),
    ("slippers", ["slipper", "flip-flop", "flip flop"]),
    ("loafers", ["loafer", "moccasin"]),
    ("shoes", ["shoe", "footwear"]),
    # --- accessories ---
    ("watch", ["watch"]),
    ("handbag", ["handbag", "hand bag", "tote", "clutch"]),
    ("backpack", ["backpack", "rucksack"]),
    ("bag", ["bag", "sling", "satchel"]),
    ("wallet", ["wallet"]),
    ("belt", ["belt"]),
    ("sunglasses", ["sunglass", "sunglasses", "eyewear"]),
    ("cap", ["cap", "hat", "beanie"]),
    ("scarf", ["scarf", "stole", "muffler"]),
    ("socks", ["socks", "sock"]),
    ("jewellery", ["earring", "necklace", "bracelet", "pendant", "jewellery",
                   "jewelry", "bangle", "anklet", "ring "]),
    # --- beauty / personal care ---
    ("lipstick", ["lipstick", "lip color", "lip colour"]),
    ("lip balm", ["lip balm", "lip care"]),
    ("kajal", ["kajal", "kohl"]),
    ("mascara", ["mascara"]),
    ("foundation", ["foundation", "concealer", "compact"]),
    ("perfume", ["perfume", "fragrance", "deodorant", "deo ", "eau de"]),
    ("moisturizer", ["moisturizer", "moisturiser", "cream", "lotion"]),
    ("sunscreen", ["sunscreen", "spf"]),
    ("shampoo", ["shampoo", "conditioner"]),
    ("serum", ["serum"]),
    ("nail", ["nail polish", "nail paint", "nail enamel"]),
    ("makeup", ["eyeliner", "eyeshadow", "blush", "highlighter", "primer"]),
    ("skincare", ["face wash", "toner", "scrub", "mask", "gel"]),
]

# Synonyms a user might type -> a canonical keyword present in product names.
SYNONYMS: dict[str, str] = {
    "denim": "jeans",
    "tee": "tshirt",
    "tees": "tshirt",
    "trainers": "sneakers",
    "kicks": "sneakers",
    "specs": "sunglasses",
    "shades": "sunglasses",
    "frock": "dress",
    "gown": "dress",
    "sari": "saree",
    "perfume": "fragrance",
    "deo": "deodorant",
    "trousers": "trouser",
    "trackpants": "track pant",
    "joggers": "jogger",
    "panties": "panty",
    "footwears": "footwear",
}


def _compile_patterns() -> list[tuple[str, re.Pattern]]:
    """Pre-compile a word-start regex for each (type, keyword)."""
    compiled = []
    for ptype, kws in _TYPE_RULES:
        # Strip trailing spaces used in rules to force token boundaries.
        parts = [re.escape(kw.strip()) for kw in kws if kw.strip()]
        # Whole-word match (+ optional plural) so "cap" matches "cap"/"caps" but
        # NOT "cappuccino"/"captain"/"capris".
        pattern = re.compile(r"\b(?:" + "|".join(parts) + r")s?\b", re.IGNORECASE)
        compiled.append((ptype, pattern))
    return compiled


_COMPILED = _compile_patterns()


def derive_product_type(name: str) -> str:
    """Return the canonical product type for a product name, or 'other'."""
    text = str(name).lower()
    for ptype, pattern in _COMPILED:
        if pattern.search(text):
            return ptype
    return "other"


def singularize(word: str) -> str:
    """Best-effort singularizer used to normalise product types ("tshirts"->"tshirt",
    "watches"->"watch", "accessories"->"accessory"). Conservative on short words."""
    w = word.lower().strip()
    if len(w) <= 3:
        return w
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "sses", "xes", "zes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def word_pattern(term: str) -> str:
    """Whole-word, plural-tolerant regex for matching a search term in text.

    Depluralizes the term so both "cap" and "caps" match "cap"/"caps", while a
    trailing boundary keeps "cap" from matching "cappuccino"/"captain"/"capris".
    """
    t = term.lower().strip()
    if len(t) > 3 and t.endswith("s"):
        t = t[:-1]
    return r"\b" + re.escape(t) + r"s?\b"


def expand_synonyms(keywords: list[str]) -> list[str]:
    """Map known synonyms in a keyword list to their canonical form (deduped)."""
    out: list[str] = []
    for kw in keywords:
        canon = SYNONYMS.get(kw.lower(), kw)
        if canon not in out:
            out.append(canon)
    return out


# Every keyword that signals "this is a product" — used by the classifier so its
# vocabulary stays in sync with the taxonomy.
def _build_vocab() -> set[str]:
    vocab: set[str] = set()
    for ptype, kws in _TYPE_RULES:
        vocab.add(ptype)
        for kw in kws:
            kw = kw.strip()
            # add each whole word of multi-word patterns
            for token in kw.replace("-", " ").split():
                if len(token) >= 3:
                    vocab.add(token)
    vocab |= set(SYNONYMS.keys())
    return vocab


PRODUCT_VOCAB: set[str] = _build_vocab()
