"""Lightweight semantic / fuzzy product search via TF-IDF.

Uses character n-gram TF-IDF vectors (scikit-learn) so the search tolerates
typos ("jeanss"), partial words and loose phrasing — without downloading any
neural model, keeping it deployable on a free Streamlit tier.

sklearn is imported lazily: if it is unavailable the module degrades gracefully
(``is_available()`` returns False and ``rank`` returns no matches), so the app
falls back to plain keyword search instead of crashing.
"""

import pandas as pd


def is_available() -> bool:
    try:
        import sklearn  # noqa: F401
        return True
    except Exception:
        return False


def make_corpus(df: pd.DataFrame) -> list[str]:
    """Combine the searchable text columns into one lowercased string per row."""
    parts = df["product_name"].astype(str)
    if "brand" in df.columns:
        parts = parts + " " + df["brand"].astype(str)
    if "product_type" in df.columns:
        parts = parts + " " + df["product_type"].astype(str)
    if "category" in df.columns:
        parts = parts + " " + df["category"].astype(str)
    return parts.str.lower().tolist()


def build_index(corpus: list[str]):
    """Fit a char n-gram TF-IDF vectorizer over the corpus.

    Returns (vectorizer, matrix) or None if sklearn is unavailable.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception:
        return None

    # Drop ultra-rare n-grams on large catalogs (noise + memory), but keep every
    # n-gram on small datasets so tiny uploads still match.
    min_df = 2 if len(corpus) > 1000 else 1
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=min_df,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


def rank(index, query: str, top_n: int = 24, min_score: float = 0.05) -> list[int]:
    """Return positional row indices of the best semantic matches for ``query``.

    ``index`` is the (vectorizer, matrix) tuple from :func:`build_index`.
    Results are ordered best-first and filtered to those scoring above
    ``min_score``. Returns an empty list if the index is missing.
    """
    if not index or not query or not query.strip():
        return []
    try:
        from sklearn.metrics.pairwise import linear_kernel
    except Exception:
        return []

    vectorizer, matrix = index
    query_vec = vectorizer.transform([query.lower()])
    scores = linear_kernel(query_vec, matrix).ravel()

    # Top candidates by score, descending.
    n = min(top_n, scores.shape[0])
    if n == 0:
        return []
    top_idx = scores.argsort()[::-1][:n]
    return [int(i) for i in top_idx if scores[i] >= min_score]
