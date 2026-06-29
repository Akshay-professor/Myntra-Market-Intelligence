import os
import base64
import random
import logging
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent import get_agent_response

logger = logging.getLogger(__name__)

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DATA_FILE = BASE_DIR / "data" / "sample" / "myntra_sample_data.csv"
LOCAL_FULL_DATA_FILE = BASE_DIR / "cleaned_myntra_data.csv"

st.set_page_config(page_title="Myntra Market Intelligence", page_icon="🛍️", layout="wide")

MYNTRA_PINK = "#ff3f6c"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Assistant:wght@400;600;700;800&display=swap');

    html, body, .stApp, [data-testid="stAppViewContainer"], [class*="css"] {
        font-family: 'Assistant','Segoe UI',Roboto,Helvetica,Arial,sans-serif;
    }
    .stApp, [data-testid="stAppViewContainer"] { background:#ffffff; }

    /* hide Streamlit chrome */
    #MainMenu {visibility:hidden;}
    header[data-testid="stHeader"] { background:transparent; height:0; }
    footer {visibility:hidden;}
    [data-testid="stToolbar"] {display:none;}

    .block-container { padding-top:0.6rem; padding-bottom:7rem; max-width:1280px; }

    /* sidebar */
    [data-testid="stSidebar"] { background:#fafafa; border-right:1px solid #f0f0f3; }
    [data-testid="stSidebar"] * { color:#282c3f; }

    /* buttons -> Myntra pink */
    .stButton > button, .stDownloadButton > button {
        background:#ff3f6c; color:#fff !important; border:none; border-radius:4px;
        font-weight:700; letter-spacing:.3px;
    }
    .stButton > button:hover, .stDownloadButton > button:hover { background:#e22e5b; color:#fff !important; }

    /* tabs */
    [data-testid="stTabs"] button[role="tab"] { font-weight:700; color:#282c3f; }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color:#ff3f6c; }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:#ff3f6c; }

    /* chat input -> Myntra search pill */
    [data-testid="stChatInput"] textarea { background:#f5f5f6; color:#282c3f; }
    [data-testid="stChatInput"] textarea::placeholder { color:#9b9ba5; }

    /* chat messages: clean, not heavy bubbles */
    .stChatMessage { background:transparent !important; border:none !important; padding:6px 0 !important; }
    .stChatMessage p, .stChatMessage li, .stChatMessage span, .stChatMessage div { color:#282c3f; }
    .stChatMessage a { color:#ff3f6c; }

    /* metric cards */
    [data-testid="stMetric"] {
        background:#fff; border:1px solid #f0f0f3; border-radius:10px;
        padding:14px; box-shadow:0 2px 8px rgba(0,0,0,.05);
    }

    /* ---- Myntra nav bar ---- */
    .myntra-nav {
        position:sticky; top:0; z-index:1000; background:#fff;
        display:flex; align-items:center; gap:34px;
        padding:8px 20px; border-bottom:1px solid #f0f0f3;
        box-shadow:0 2px 6px rgba(0,0,0,.05); margin-bottom:14px;
    }
    .myntra-wordmark { font-weight:800; font-size:27px; color:#ff3f6c; font-style:italic; letter-spacing:-1px; }
    .myntra-nav-logo img { height:44px; display:block; }
    .myntra-nav-links { display:flex; gap:24px; }
    .myntra-nav-links a {
        color:#282c3f; text-decoration:none; font-weight:700; font-size:13px;
        text-transform:uppercase; letter-spacing:.3px; padding:14px 0;
        border-bottom:4px solid transparent;
    }
    .myntra-nav-links a:hover { color:#ff3f6c; border-bottom:4px solid #ff3f6c; }
    .myntra-nav-spacer { flex:1; }
    .myntra-nav-icons { display:flex; gap:26px; }
    .myntra-nav-icons div {
        display:flex; flex-direction:column; align-items:center;
        font-size:11px; font-weight:700; color:#282c3f; cursor:pointer; line-height:1.3;
    }
    .myntra-nav-icons div span { font-size:17px; }

    /* product grid */
    .product-grid {
        display:grid; grid-template-columns:repeat(auto-fill,minmax(185px,1fr));
        gap:16px; margin-top:10px;
    }

    /* shop hero */
    .shop-hero { text-align:center; padding:26px 0 10px; }
    .shop-hero h2 { color:#282c3f; margin:0; font-weight:800; font-size:26px; }
    .shop-hero p { color:#7e818c; margin:6px 0 0; font-size:15px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _logo_html() -> str:
    """Return the nav-bar logo: an image from assets/ if present, else a wordmark."""
    for ext in ("png", "svg", "jpg", "jpeg", "webp"):
        path = BASE_DIR / "assets" / f"myntra_logo.{ext}"
        if path.exists():
            data = base64.b64encode(path.read_bytes()).decode()
            mime = "svg+xml" if ext == "svg" else ext
            return f'<span class="myntra-nav-logo"><img src="data:image/{mime};base64,{data}"/></span>'
    return '<span class="myntra-wordmark">Myntra</span>'


def render_navbar() -> None:
    """Render the sticky, Myntra-style top navigation bar."""
    links = "".join(
        f'<a href="#">{c}</a>'
        for c in ["Men", "Women", "Kids", "Home &amp; Living", "Beauty", "Studio"]
    )
    icons = (
        '<div><span>👤</span>Profile</div>'
        '<div><span>❤️</span>Wishlist</div>'
        '<div><span>🛍️</span>Bag</div>'
    )
    st.markdown(
        f'<div class="myntra-nav">{_logo_html()}'
        f'<div class="myntra-nav-links">{links}</div>'
        f'<div class="myntra-nav-spacer"></div>'
        f'<div class="myntra-nav-icons">{icons}</div></div>',
        unsafe_allow_html=True,
    )


# ---------- Data Loading (Cached) ----------

def _ensure_product_type(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a derived `product_type` column if the dataset doesn't already have one."""
    import taxonomy
    if "product_type" not in frame.columns and "product_name" in frame.columns:
        frame = frame.copy()
        frame["product_type"] = frame["product_name"].apply(taxonomy.derive_product_type)
    return frame


@st.cache_data
def load_csv_data(filepath: str) -> pd.DataFrame:
    """Load and cache a CSV file so it's only parsed once."""
    return _ensure_product_type(pd.read_csv(filepath))


@st.cache_data
def generate_sample_data() -> pd.DataFrame:
    random.seed(42)  # Fix #7: deterministic sample data
    brands = ["Nike", "Adidas", "H&M", "Zara", "Levis", "Puma", "UCB", "Roadster", "HRX", "Biba"]
    categories = ["Men", "Women", "Kids"]
    data = []

    for i in range(1, 51):
        brand = random.choice(brands)
        category = random.choice(categories)
        original_price = random.randint(200, 5000)
        discount_pct = random.randint(10, 70)
        discounted_price = round(original_price * (1 - discount_pct / 100), 2)
        rating = round(random.uniform(2.5, 5.0), 1)

        data.append(
            {
                "product_id": f"P{10000 + i}",
                "product_name": f"{brand} {category} Apparel {i}",
                "brand": brand,
                "category": category,
                "original_price": original_price,
                "discount_pct": discount_pct,
                "discounted_price": discounted_price,
                "rating": rating,
                "num_reviews": random.randint(0, 5000),
            }
        )

    return _ensure_product_type(pd.DataFrame(data))


def validate_dataframe(df: pd.DataFrame) -> tuple[bool, str]:
    required_cols = {
        "product_id",
        "product_name",
        "brand",
        "category",
        "original_price",
        "discount_pct",
        "discounted_price",
        "rating",
        "num_reviews",
    }
    missing = required_cols - set(df.columns)
    if missing:
        return False, f"Missing required columns: {', '.join(sorted(missing))}"
    if df.empty:
        return False, "Dataset is empty."
    return True, ""


def prepare_rating_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    rating_bins = [0, 3.0, 3.5, 4.0, 4.5, 5.01]
    rating_labels = ["<3.0", "3.0-3.5", "3.5-4.0", "4.0-4.5", "4.5+"]
    bucketed = frame.copy()
    bucketed["rating_bucket"] = pd.cut(
        bucketed["rating"],
        bins=rating_bins,
        labels=rating_labels,
        include_lowest=True,
        right=False,
    )
    bucket_counts = (
        bucketed.groupby("rating_bucket", observed=False)
        .size()
        .reindex(rating_labels, fill_value=0)
        .reset_index(name="product_count")
    )
    return bucket_counts


# ---------- Intent Routing (classifier-driven) ----------

import re
import agent
import data_tools
import classifier
import taxonomy
import semantic_search


@st.cache_resource(show_spinner=False)
def get_semantic_index(_dataframe, signature):
    """Build (and cache) the TF-IDF semantic index for the active dataset.

    ``_dataframe`` is excluded from the cache key (leading underscore); ``signature``
    keys the cache so the index rebuilds only when the dataset actually changes.
    """
    if not semantic_search.is_available():
        return None
    corpus = semantic_search.make_corpus(_dataframe)
    return semantic_search.build_index(corpus)


@st.cache_data(show_spinner=False)
def get_brand_set(_dataframe, signature) -> set:
    """Lowercased set of brand names in the active dataset (for brand detection)."""
    if "brand" not in _dataframe.columns:
        return set()
    return set(_dataframe["brand"].dropna().astype(str).str.lower().str.strip())

_GREETING_RESPONSE = (
    "Hey there! 👋 Welcome to the Myntra Market Intelligence Agent!\n\n"
    "I can help you with:\n"
    "- 🔍 **Finding products** — *'show me jeans under ₹2000'*\n"
    "- 📊 **Brand analytics** — *'top brands by discount'*\n"
    "- ⭐ **Ratings & reviews** — *'best rated products'*\n"
    "- 📈 **Category insights** — *'category pricing summary'*\n\n"
    "What would you like to explore?"
)

_OUT_OF_SCOPE_RESPONSE = (
    "I'm the **Myntra Market Intelligence** assistant, so I can only help with "
    "shopping and Myntra catalog data — not general questions. 🙂\n\n"
    "Try something like:\n"
    "- 🔍 *'black jeans under ₹2000'*\n"
    "- 📊 *'top brands by discount'*\n"
    "- ⭐ *'most reviewed products'*\n"
    "- 📈 *'category pricing summary'*"
)

# Filler phrases/words stripped from a product query to isolate the keywords.
_FILLER_PHRASES = [
    "show me", "find me", "i want to buy", "i want to purchase",
    "i want", "i need", "search for", "looking for", "get me",
    "up to", "less than", "more than", "greater than",
]
_FILLER_WORDS = [
    "show", "find", "buy", "purchase", "suggest", "recommend",
    "under", "below", "within", "budget", "max", "upto",
    "above", "over", "min", "minimum",
    "expensive", "cheap", "cheaper", "affordable", "costly",
    "please", "pls", "plz", "some", "any", "the", "a", "an",
    "rs", "₹", "inr", "rupees", "rupee",
    # discount / sorting noise
    "discount", "off", "percent", "percentage",
    "best", "rated", "rating", "top", "good", "nice", "popular",
    "highest", "great",
    # generic noise
    "with", "for", "size", "sizes", "and", "or", "free", "to",
    "hack", "hacks", "get", "products", "product", "item", "items",
]


def _extract_search_params(query: str):
    """Parse a product query into structured search parameters.

    Returns (keywords, min_price, max_price, min_discount, sort_by_rating, search_term).
    """
    q = query.lower().strip()

    # --- Discount filter ("70% off", "above 50% discount", "discount over 60") ---
    min_discount = None
    dm = re.search(r'(\d+)\s*%?\s*(?:off|discount)', q)
    if dm:
        min_discount = float(dm.group(1))
    else:
        dm = re.search(
            r'(?:off|discount)\s*(?:above|over|more than|greater than|of|at\s*least)?\s*(\d+)', q
        )
        if dm:
            min_discount = float(dm.group(1))

    # Remove the discount phrase so it isn't mistaken for a price below.
    q_price = q
    if min_discount is not None:
        q_price = re.sub(r'\d+\s*%?\s*(?:off|discount)', ' ', q_price)
        q_price = re.sub(
            r'(?:off|discount)\s*(?:above|over|more than|greater than|of|at\s*least)?\s*\d+', ' ', q_price
        )

    # --- Price filters ---
    max_price = None
    min_price = None
    m = re.search(
        r'(?:under|below|within|budget|max|upto|up to|less than)\s*(?:rs\.?|₹|inr)?\s*(\d+)', q_price
    )
    if m:
        max_price = float(m.group(1))
    m = re.search(
        r'(?:above|over|more than|min|minimum)\s*(?:rs\.?|₹|inr)?\s*(\d+)', q_price
    )
    if m:
        min_price = float(m.group(1))

    # --- Sort hint ---
    sort_by_rating = bool(re.search(r'(best|top|highest|good)\s*rated|\brating\b', q))

    # --- Keywords (strip fillers, numbers, punctuation) ---
    search_term = q
    for filler in _FILLER_PHRASES:
        search_term = search_term.replace(filler, " ")
    for word in _FILLER_WORDS:
        search_term = re.sub(r'\b' + re.escape(word) + r'\b', ' ', search_term)
    search_term = re.sub(r'\d+', '', search_term)
    search_term = re.sub(r'[%₹]', ' ', search_term)
    search_term = re.sub(r'\s+', ' ', search_term).strip()

    keywords = search_term.split()
    return keywords, min_price, max_price, min_discount, sort_by_rating, search_term


def _product_search_message(query: str, dataframe) -> dict:
    """Run a product search and return a structured chat message.

    Returns either a {"kind": "products", ...} message or a plain text message
    (for the "no results" case).
    """
    keywords, min_price, max_price, min_discount, sort_by_rating, search_term = \
        _extract_search_params(query)

    # Map synonyms (denim -> jeans, tee -> tshirt, ...) so phrasing differences
    # still hit the catalog.
    keywords = taxonomy.expand_synonyms(keywords)

    has_filter = min_price is not None or max_price is not None or min_discount is not None

    # Need either a keyword or a numeric filter to run a search.
    if not keywords and not has_filter:
        return {"role": "assistant", "content": _OUT_OF_SCOPE_RESPONSE}

    results = data_tools.find_products(
        dataframe, keywords, min_price=min_price, max_price=max_price,
        min_discount=min_discount, sort_by_rating=sort_by_rating, limit=24,
    )

    # Semantic fallback: if keyword search found nothing (typo / unusual phrasing),
    # rank the whole catalog by TF-IDF similarity, then re-apply the numeric filters.
    if results.empty and keywords:
        index = get_semantic_index(dataframe, (len(dataframe), tuple(dataframe.columns)))
        positions = semantic_search.rank(index, search_term or query, top_n=60)
        if positions:
            candidate = dataframe.iloc[positions]
            results = data_tools.find_products(
                candidate, [], min_price=min_price, max_price=max_price,
                min_discount=min_discount, sort_by_rating=sort_by_rating, limit=24,
            )
            # find_products with no keywords + no filter returns empty; in that case
            # keep the semantic order directly.
            if results.empty and not has_filter:
                results = candidate.head(24)

    label = search_term or "products"
    if results.empty:
        msg = f"Sorry, I couldn't find any **{label}**"
        if min_discount:
            msg += f" with {int(min_discount)}%+ discount"
        if max_price:
            msg += f" under ₹{int(max_price)}"
        if min_price:
            msg += f" above ₹{int(min_price)}"
        return {"role": "assistant", "content": msg + " in the catalog. Try a different keyword!"}

    prefix = "Top-rated" if sort_by_rating else "Top"
    header = f"{prefix} results for **{label}**"
    if min_discount:
        header += f" with **{int(min_discount)}%+ off**"
    if max_price:
        header += f" under **₹{int(max_price)}**"
    if min_price:
        header += f" above **₹{int(min_price)}**"

    return {"role": "assistant", "kind": "products", "header": header,
            "items": _results_to_items(results)}


def _results_to_items(results) -> list[dict]:
    """Convert a results DataFrame into the carousel's item dicts."""
    items = []
    for _, row in results.iterrows():
        img_url = row.get("image_url", "")
        items.append({
            "name": str(row.get("product_name", "Product")),
            "brand": str(row.get("brand", "Unknown")),
            "price": row.get("discounted_price", "N/A"),
            "original_price": row.get("original_price", None),
            "discount": row.get("discount_pct", 0),
            "rating": row.get("rating", None),
            "img": str(img_url) if pd.notna(img_url) else "",
            "url": str(row.get("product_url", "#")),
        })
    return items


def _structured_product_message(dataframe, facts: dict, query: str) -> dict:
    """Build a product grid from LLM-extracted facts (primary search path)."""
    # A bare brand name ("roadster") -> show that brand's catalog, not a guessed item.
    brand_set = get_brand_set(dataframe, (len(dataframe), tuple(dataframe.columns)))
    facts = agent.apply_brand_override(facts, query, brand_set)

    item = facts.get("item")
    color = facts.get("color")
    attributes = facts.get("attributes") or []
    brand = facts.get("brand")
    gender = facts.get("gender")
    min_price = facts.get("min_price")
    max_price = facts.get("max_price")
    min_discount = facts.get("min_discount")
    sort_by_rating = bool(re.search(r'(best|top|highest|good)\s*rated|\brating\b', query.lower()))

    results = data_tools.structured_search(
        dataframe, item=item, color=color, attributes=attributes, brand=brand,
        gender=gender, min_price=min_price, max_price=max_price,
        min_discount=min_discount, sort_by_rating=sort_by_rating, limit=24,
    )

    # Semantic fallback if hard filters left nothing (typos / unusual phrasing).
    if results.empty:
        index = get_semantic_index(dataframe, (len(dataframe), tuple(dataframe.columns)))
        sem_query = " ".join(p for p in [color, item, *attributes] if p) or query
        positions = semantic_search.rank(index, sem_query, top_n=60)
        if positions:
            candidate = dataframe.iloc[positions]
            if min_price is not None:
                candidate = candidate[candidate["discounted_price"] >= min_price]
            if max_price is not None:
                candidate = candidate[candidate["discounted_price"] <= max_price]
            if min_discount is not None and "discount_pct" in candidate.columns:
                candidate = candidate[candidate["discount_pct"] >= min_discount]
            results = candidate.head(24)

    # Build a readable label from the facts (brand included).
    label_bits = [b for b in [brand, gender, color, *attributes, item] if b]
    label = " ".join(label_bits).title() if label_bits else "products"
    generic = not label_bits and (min_price is not None or max_price is not None or min_discount is not None)

    def _price_suffix(sep_disc=" with ", sep_max=" under ", sep_min=" above ", bold=False):
        b = "**" if bold else ""
        s = ""
        if min_discount:
            s += f"{sep_disc}{b}{int(min_discount)}%+ off{b}"
        if max_price:
            s += f"{sep_max}{b}₹{int(max_price)}{b}"
        if min_price:
            s += f"{sep_min}{b}₹{int(min_price)}{b}"
        return s

    if results.empty:
        return {"role": "assistant",
                "content": f"Sorry, I couldn't find any **{label}**" + _price_suffix()
                + " in the catalog. Try a different search!"}

    if generic:
        header = ("Showing a budget selection" + _price_suffix(bold=True)
                  + ".\n\n*Tip: add a type like 'jeans under 300' to narrow it down.*")
    else:
        prefix = "Top-rated" if sort_by_rating else "Top"
        header = f"{prefix} results for **{label}**" + _price_suffix(bold=True)

    return {"role": "assistant", "kind": "products", "header": header,
            "items": _results_to_items(results)}


def build_chat_context(messages: list, current_query: str) -> str:
    """Build a compact context string from recent user messages only."""
    recent_user_msgs = [
        m["content"][:120] for m in messages
        if m["role"] == "user" and "content" in m
    ][-3:]

    if not recent_user_msgs:
        return current_query

    context_lines = [f"- {msg}" for msg in recent_user_msgs]
    return (
        f"Recent user queries for context:\n"
        + "\n".join(context_lines)
        + f"\n\nCurrent question: {current_query}"
    )


def _agent_message(query: str, dataframe, messages: list) -> dict:
    full_prompt = build_chat_context(messages, query)
    return {"role": "assistant", "content": get_agent_response(dataframe, full_prompt)}


def get_smart_response(query: str, dataframe, messages: list) -> dict:
    """Understand the query, then route to the matching handler.

    Cheap rule fast-paths handle obvious greetings/analytics with no LLM. Anything
    else goes through one LLM call (``understand_query``) that extracts structured
    facts AND the intent, which drives an accurate structured product search. If the
    LLM is unavailable, we fall back to the rule classifier + keyword search.

    Always returns a chat *message dict* so the render loop can draw it consistently.
    """
    rule = classifier.rule_intent(query)
    if rule == classifier.GREETING:
        return {"role": "assistant", "content": _GREETING_RESPONSE}
    if rule == classifier.ANALYTICS:
        return _agent_message(query, dataframe, messages)

    # Primary path: LLM understands the query (intent + search facts).
    facts = agent.understand_query(query)

    if facts is None:
        # LLM unavailable / failed -> degrade to rules + keyword search.
        fallback = rule if rule is not None else classifier.classify_query(query)
        if fallback == classifier.PRODUCT_SEARCH:
            return _product_search_message(query, dataframe)
        if fallback == classifier.ANALYTICS:
            return _agent_message(query, dataframe, messages)
        if fallback == classifier.GREETING:
            return {"role": "assistant", "content": _GREETING_RESPONSE}
        return {"role": "assistant", "content": _OUT_OF_SCOPE_RESPONSE}

    intent = facts.get("intent")
    if intent == "greeting":
        return {"role": "assistant", "content": _GREETING_RESPONSE}
    if intent == "analytics":
        return _agent_message(query, dataframe, messages)
    if intent == "product_search":
        return _structured_product_message(dataframe, facts, query)
    return {"role": "assistant", "content": _OUT_OF_SCOPE_RESPONSE}


# ---------- Product Card / Grid Renderer ----------

_MYNTRA_PINK = MYNTRA_PINK


def _fmt_price(value) -> str | None:
    try:
        if value is None or pd.isna(value):
            return None
        return f"{int(round(float(value))):,}"
    except (ValueError, TypeError):
        return None


def _product_card_html(item: dict) -> str:
    """Build one Myntra-style product card as an HTML string."""
    img = item.get("img", "")
    img = img.strip() if isinstance(img, str) else ""
    has_img = bool(img) and img != "-" and img.lower() != "nan"

    brand = str(item.get("brand", "")).strip()
    name = str(item.get("name", "")).strip()
    url = item.get("url", "#") or "#"

    price = _fmt_price(item.get("price"))
    mrp = _fmt_price(item.get("original_price"))
    try:
        discount = float(item.get("discount", 0) or 0)
    except (ValueError, TypeError):
        discount = 0.0
    try:
        rating = float(item.get("rating")) if item.get("rating") is not None else None
        if rating is not None and pd.isna(rating):
            rating = None
    except (ValueError, TypeError):
        rating = None

    image_block = (
        f'<img src="{img}" style="width:100%;height:210px;object-fit:cover;'
        'border-radius:6px 6px 0 0;display:block;" '
        'onerror="this.style.display=\'none\'"/>'
        if has_img else
        '<div style="width:100%;height:210px;background:#f5f5f6;border-radius:6px 6px 0 0;'
        'display:flex;align-items:center;justify-content:center;color:#bbb;font-size:13px;">'
        'No image</div>'
    )

    rating_badge = ""
    if rating is not None:
        rating_badge = (
            f'<span style="background:#fff !important;border:1px solid #eaeaec;border-radius:3px;'
            'padding:1px 5px;font-size:11px;font-weight:700;color:#282c3f !important;'
            f'box-shadow:0 1px 2px rgba(0,0,0,.08);">{rating:.1f} '
            '<span style="color:#14958f !important;">★</span></span>'
        )

    price_line = ""
    if price is not None:
        price_line = f'<span style="font-weight:700;color:#282c3f !important;">₹{price}</span>'
        if mrp is not None and discount > 0:
            price_line += (
                f' <span style="color:#7e818c !important;text-decoration:line-through;'
                f'font-size:12px;">₹{mrp}</span>'
            )
        if discount > 0:
            price_line += (
                f' <span style="color:#ff905a !important;font-size:12px;font-weight:700;">'
                f'({discount:.0f}% OFF)</span>'
            )

    return (
        '<div style="width:100%;background:#fff !important;border:1px solid #f0f0f3;'
        'border-radius:8px;box-shadow:0 1px 6px rgba(0,0,0,.06);overflow:hidden;'
        'display:flex;flex-direction:column;transition:box-shadow .2s;">'
        f'{image_block}'
        '<div style="padding:10px 12px 12px;display:flex;flex-direction:column;gap:4px;">'
        f'<div style="font-weight:700;color:#282c3f !important;font-size:14px;">{brand}</div>'
        f'<div style="color:#7e818c !important;font-size:12px;white-space:nowrap;overflow:hidden;'
        f'text-overflow:ellipsis;">{name}</div>'
        f'<div style="margin-top:2px;">{price_line}</div>'
        f'<div style="margin-top:4px;">{rating_badge}</div>'
        f'<a href="{url}" target="_blank" style="margin-top:8px;text-align:center;'
        f'background:{_MYNTRA_PINK} !important;color:#fff !important;font-weight:700;font-size:12px;'
        'padding:7px 0;border-radius:4px;text-decoration:none;letter-spacing:.3px;">VIEW ON MYNTRA</a>'
        '</div></div>'
    )


_GRID_LIMIT = 12


def render_product_grid(msg: dict, msg_key: int) -> None:
    """Render results as a responsive Myntra-style product grid."""
    header = msg.get("header", "")
    if header:
        st.markdown(header)

    items = msg.get("items", [])
    if not items:
        return

    display = items[:_GRID_LIMIT]
    cards = "".join(_product_card_html(it) for it in display)
    st.markdown(f'<div class="product-grid">{cards}</div>', unsafe_allow_html=True)

    if len(items) > len(display):
        st.markdown(
            f"<div style='color:#9aa0b4;font-size:12px;margin-top:8px;'>"
            f"Showing top {len(display)} of {len(items)} matches</div>",
            unsafe_allow_html=True,
        )


def render_message(msg: dict, msg_key: int) -> None:
    """Dispatch a stored chat message to the correct renderer."""
    if msg.get("kind") == "products":
        render_product_grid(msg, msg_key)
    else:
        st.markdown(msg.get("content", ""))


# ---------- Session State ----------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "carousel_offsets" not in st.session_state:
    st.session_state.carousel_offsets = {}

# ---------- Sidebar ----------

with st.sidebar:
    st.markdown("### 🛍️ Market Intelligence")
    st.caption("Powered by Groq AI + LangChain")
    st.write("Upload your own catalog or use the built-in dataset.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = _ensure_product_type(pd.read_csv(uploaded_file))
            st.success("File uploaded successfully.")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            df = pd.DataFrame()
    else:
        # Fix #1 & #13: Cached loading with spinner
        if LOCAL_FULL_DATA_FILE.exists():
            with st.spinner("Loading full dataset..."):
                df = load_csv_data(str(LOCAL_FULL_DATA_FILE))
            st.info(f"Using local cleaned dataset ({len(df):,} products).")
        elif SAMPLE_DATA_FILE.exists():
            df = load_csv_data(str(SAMPLE_DATA_FILE))
            st.info("Using default sample dataset (50 rows).")
        else:
            df = generate_sample_data()
            st.info("Using generated sample dataset (50 rows).")

    if not df.empty:
        with st.expander("Dataset Preview"):
            st.dataframe(df.head(5), width="stretch")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", width="stretch"):
        st.session_state.messages = []
        st.session_state.carousel_offsets = {}
        st.rerun()

    if st.session_state.messages:
        chat_export = "Myntra AI Insights Report\n" + "=" * 30 + "\n\n"
        for msg in st.session_state.messages:
            role = "User Query" if msg["role"] == "user" else "AI Analyst"
            if msg.get("kind") == "products":
                lines = [msg.get("header", "Product results")]
                for it in msg.get("items", []):
                    lines.append(
                        f"- {it.get('brand', '')} - {it.get('name', '')} | "
                        f"₹{it.get('price', 'N/A')} | {it.get('discount', 0)}% off | "
                        f"{it.get('url', '')}"
                    )
                body = "\n".join(lines)
            else:
                body = msg.get("content", "")
            chat_export += f"{role}:\n{body}\n{'-' * 30}\n"

        st.download_button(
            label="📄 Download Insights Report",
            data=chat_export,
            file_name="myntra_ai_insights.txt",
            mime="text/plain",
            width="stretch",
        )

# ---------- Main Content ----------

render_navbar()

if not os.getenv("GROQ_API_KEY"):
    st.error(
        "GROQ_API_KEY not found. Create a .env file in the project root with:\n\n"
        "GROQ_API_KEY=your_groq_api_key_here"
    )
    st.stop()

is_valid, validation_msg = validate_dataframe(df)
if not is_valid:
    st.warning(f"Dataset issue: {validation_msg}")
    st.stop()

quick_query = None
shop_tab, insights_tab = st.tabs(["🛍️  Shop", "📊  Insights"])

# ===== SHOP TAB: search + chat + product grid =====
with shop_tab:
    if not st.session_state.messages:
        st.markdown(
            '<div class="shop-hero"><h2>What are you looking for?</h2>'
            '<p>Search thousands of products or ask for market insights — powered by AI.</p></div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Popular searches**")
        s1, s2, s3, s4 = st.columns(4)
        if s1.button("👖  Jeans under ₹1500", width="stretch"):
            quick_query = "jeans under 1500"
        if s2.button("👟  Running shoes", width="stretch"):
            quick_query = "running shoes"
        if s3.button("💄  Lipstick", width="stretch"):
            quick_query = "lipstick"
        if s4.button("🏆  Top brands by discount", width="stretch"):
            quick_query = "What are the top 10 brands with the highest average discount?"

    for idx, message in enumerate(st.session_state.messages):
        avatar = "🧑" if message["role"] == "user" else "🛍️"
        with st.chat_message(message["role"], avatar=avatar):
            render_message(message, idx)

# ===== INSIGHTS TAB: stats + charts =====
with insights_tab:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Products", f"{len(df):,}")
    m2.metric("Brands", f"{df['brand'].nunique() if 'brand' in df.columns else 0:,}")
    m3.metric("Categories", f"{df['category'].nunique() if 'category' in df.columns else 0}")
    m4.metric("Avg Discount %", f"{df['discount_pct'].mean() if 'discount_pct' in df.columns else 0:.1f}%")

    st.markdown("###")
    if st.button("🏆 Ask AI: top brands by discount"):
        quick_query = "What are the top 10 brands with the highest average discount?"

    visual_tabs = st.tabs([
        "Brand Performance",
        "Category Pricing",
        "Rating Insights",
        "Discount Strategy",
    ])

    with visual_tabs[0]:
        st.caption("Top 10 brands by average discount, and the brands with the largest catalogs.")
        brand_discount_df = (
            df.groupby("brand", as_index=False)
            .agg(avg_discount_pct=("discount_pct", "mean"), total_products=("product_id", "count"))
            .round(2)
        )
        left_brand, right_brand = st.columns(2)
        with left_brand:
            st.markdown("**Top 10 Brands by Average Discount**")
            st.bar_chart(
                brand_discount_df.sort_values("avg_discount_pct", ascending=False).head(10),
                x="brand", y="avg_discount_pct", width="stretch", color=MYNTRA_PINK,
            )
        with right_brand:
            st.markdown("**Top 10 Brands by Product Count**")
            st.bar_chart(
                brand_discount_df.sort_values("total_products", ascending=False).head(10),
                x="brand", y="total_products", width="stretch", color=MYNTRA_PINK,
            )

    with visual_tabs[1]:
        st.caption("Average original vs discounted price by category.")
        category_pricing_df = (
            df.groupby("category", as_index=False)
            .agg(
                mean_original_price=("original_price", "mean"),
                mean_discounted_price=("discounted_price", "mean"),
            )
            .round(2)
        )
        st.bar_chart(
            category_pricing_df, x="category",
            y=["mean_original_price", "mean_discounted_price"], width="stretch",
        )

    with visual_tabs[2]:
        st.caption("Product counts across rating tiers.")
        st.bar_chart(
            prepare_rating_buckets(df), x="rating_bucket", y="product_count",
            width="stretch", color=MYNTRA_PINK,
        )

    with visual_tabs[3]:
        st.caption("Each dot is a product: discount % vs rating, colored by category.")
        st.scatter_chart(
            df[["discount_pct", "rating", "category"]].copy(),
            x="discount_pct", y="rating", color="category", width="stretch",
        )

# ---------- Chat Handler (global search bar) ----------

typed_query = st.chat_input("Search for products, brands and more…")
user_query = quick_query or typed_query

if user_query:
    with st.spinner("Searching…"):
        try:
            # Pass prior history (without the current turn) for LLM context.
            response = get_smart_response(user_query, df, st.session_state.messages)
        except Exception as e:
            logger.error("Chat query failed: %s", e)
            response = {
                "role": "assistant",
                "content": "Something went wrong while processing your request. Please try again!",
            }

    st.session_state.messages.append({"role": "user", "content": user_query})
    st.session_state.messages.append(response)
    st.rerun()

