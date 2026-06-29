"""Query intent classifier.

Hybrid design: cheap rule-based fast-paths first (zero tokens), then a single
small-model LLM call only for genuinely ambiguous queries. Returns one of:

    GREETING | PRODUCT_SEARCH | ANALYTICS | OUT_OF_SCOPE
"""

import re

from agent import classify_intent_llm
from taxonomy import PRODUCT_VOCAB as _TAXONOMY_VOCAB

GREETING = "GREETING"
PRODUCT_SEARCH = "PRODUCT_SEARCH"
ANALYTICS = "ANALYTICS"
OUT_OF_SCOPE = "OUT_OF_SCOPE"

# ---------------------------------------------------------------------------
# Greeting detection
# ---------------------------------------------------------------------------

_GREETING_WORDS = {
    "hi", "hello", "hey", "hii", "hiii", "hola", "sup", "yo", "greetings",
    "thanks", "thank you", "thankyou", "ok", "okay", "cool", "nice", "bye",
}
_GREETING_PHRASES = {
    "how are you", "how r u", "what's up", "whats up",
    "good morning", "good evening", "good afternoon", "thank you",
}


def _is_greeting(text: str) -> bool:
    cleaned = text.strip().lower().rstrip("?!.,")
    if cleaned in _GREETING_WORDS:
        return True
    for phrase in _GREETING_PHRASES:
        if phrase in cleaned and len(cleaned) < 30:
            return True
    # Short messages that start with a greeting word ("hey there", "hello!").
    tokens = re.findall(r"[a-z']+", cleaned)
    if tokens and tokens[0] in _GREETING_WORDS and len(tokens) <= 3:
        return True
    return False


# ---------------------------------------------------------------------------
# Analytics detection
# ---------------------------------------------------------------------------

_ANALYTICAL_KEYWORDS = [
    "top brands", "brand performance", "average", "summary", "compare",
    "distribution", "category", "rating", "discount strategy", "most reviewed",
    "highest", "lowest", "how many", "total", "count", "percentage",
    "analytics", "analysis", "insight", "trend", "overall", "revenue",
    "concentration", "per category", "by category",
]


_ANALYTICAL_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(kw) for kw in _ANALYTICAL_KEYWORDS) + r")\b"
)


def _is_analytical(text: str) -> bool:
    # Word-boundary match so "discount" doesn't trip the "count" keyword, etc.
    return bool(_ANALYTICAL_RE.search(text))


# ---------------------------------------------------------------------------
# Product-search detection
# ---------------------------------------------------------------------------

# Product-type vocabulary comes from the shared taxonomy so the classifier and
# the search layer always agree on what counts as a product term. A few extra
# generic terms are added that aren't specific product types.
PRODUCT_VOCAB = set(_TAXONOMY_VOCAB) | {
    "outfit", "apparel", "clothing", "wear", "accessory", "accessories",
}

# Shopping intent phrases.
_SHOPPING_PHRASES = [
    "show me", "show", "find me", "find", "i want", "i need", "looking for",
    "search for", "get me", "buy", "purchase", "recommend", "suggest",
]

# Price-range patterns (under/above/below ₹1000 etc.).
_PRICE_PATTERN = re.compile(
    r"(?:under|below|within|budget|max|upto|up to|less than|above|over|"
    r"more than|min|minimum|cheaper than)\s*(?:rs\.?|₹|inr)?\s*\d+",
    re.IGNORECASE,
)


def _has_product_signal(text: str) -> bool:
    if _PRICE_PATTERN.search(text):
        return True
    if any(phrase in text for phrase in _SHOPPING_PHRASES):
        return True
    words = set(re.findall(r"[a-zA-Z\-]+", text))
    return bool(words & PRODUCT_VOCAB)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rule_intent(query: str) -> str | None:
    """Classify using deterministic rules only (no LLM).

    Returns an intent, or None when the rules can't decide.
    """
    if not query or not query.strip():
        return OUT_OF_SCOPE

    text = query.lower().strip()

    if _is_greeting(text):
        return GREETING

    # Analytics keywords take priority over product-vocab (e.g. "average rating
    # of shirts" is analytics, not a product search).
    if _is_analytical(text):
        return ANALYTICS

    if _has_product_signal(text):
        return PRODUCT_SEARCH

    return None


def classify_query(query: str) -> str:
    """Return the intent of a user query.

    Cheap deterministic rules first, then a small LLM call only when the rules
    can't decide (this is what reliably catches off-topic questions vs. uncommon
    product terms).
    """
    intent = rule_intent(query)
    if intent is not None:
        return intent

    # Ambiguous -> spend one cheap LLM call.
    return classify_intent_llm(query)
