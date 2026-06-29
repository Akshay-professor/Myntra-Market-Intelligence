import re

import pandas as pd

import taxonomy


def get_top_brands_by_discount(df: pd.DataFrame) -> str:
    """Returns top 10 brands with highest average discount %."""
    res = df.groupby("brand")["discount_pct"].mean().nlargest(10).round(2)
    return res.reset_index().to_markdown(index=False)


def get_category_summary(df: pd.DataFrame) -> str:
    """Returns per category: avg price, avg discount, product count."""
    res = (
        df.groupby("category")
        .agg(
            avg_price=("discounted_price", "mean"),
            avg_discount=("discount_pct", "mean"),
            product_count=("product_id", "count"),
        )
        .round(2)
    )
    return res.reset_index().to_markdown(index=False)


def get_most_reviewed_products(df: pd.DataFrame) -> str:
    """Returns top 10 products by num_reviews with their ratings."""
    cols = ["product_name", "num_reviews", "rating"]
    if "image_url" in df.columns: cols.append("image_url")
    if "product_url" in df.columns: cols.append("product_url")
    res = df.nlargest(10, "num_reviews")[cols]
    return res.to_markdown(index=False)


def get_high_discount_products(df: pd.DataFrame, threshold=50) -> str:
    """Returns products with discount > threshold%."""
    try:
        thresh = float(threshold)
    except (ValueError, TypeError):
        thresh = 50.0

    cols = ["product_name", "brand", "discount_pct", "discounted_price"]
    if "image_url" in df.columns: cols.append("image_url")
    if "product_url" in df.columns: cols.append("product_url")

    res = df[df["discount_pct"] > thresh][cols].head(10)

    if res.empty:
        return f"No products found with a discount greater than {thresh}%."
    return res.to_markdown(index=False)


def get_rating_distribution(df: pd.DataFrame) -> str:
    """Returns count of products in rating buckets: <3, 3-3.5, 3.5-4, 4-4.5, 4.5+."""
    bins = [0, 2.99, 3.49, 3.99, 4.49, 5.0]
    labels = ["<3", "3-3.5", "3.5-4", "4-4.5", "4.5+"]
    res = pd.cut(df["rating"], bins=bins, labels=labels).value_counts().sort_index()
    return (
        res.reset_index(name="product_count")
        .rename(columns={"index": "rating_bucket"})
        .to_markdown(index=False)
    )


def get_brand_performance(df: pd.DataFrame) -> str:
    """Returns brand-wise: avg rating, avg discount, total products, sorted by avg rating desc."""
    res = (
        df.groupby("brand")
        .agg(
            avg_rating=("rating", "mean"),
            avg_discount=("discount_pct", "mean"),
            total_products=("product_id", "count"),
        )
        .sort_values("avg_rating", ascending=False)
        .round(2)
    )
    return res.head(20).reset_index().to_markdown(index=False)


# ---------------------------------------------------------------------------
# Advanced product search — used by both LLM agent and direct search
# ---------------------------------------------------------------------------

def find_products(df: pd.DataFrame, keywords: list[str],
                  min_price: float | None = None,
                  max_price: float | None = None,
                  limit: int = 10,
                  match_all: bool = True,
                  min_discount: float | None = None,
                  sort_by_rating: bool = False) -> pd.DataFrame:
    """Search products by matching keywords against product_name, brand, and category.

    Keywords are matched on a WORD-BOUNDARY basis (``\\bkeyword``) so "men" does not
    match "women" and "tshirt" does not match "sweatshirt", while plurals like
    "tshirts" still match "tshirt".

    With ``match_all=True`` (default), a product must contain EVERY meaningful keyword
    (AND logic); if that strict pass returns nothing it gracefully falls back to OR
    logic so rare or misspelled queries still surface something.

    ``min_price`` / ``max_price`` filter on discounted price; ``min_discount`` filters on
    discount percentage. When no keywords are given but a filter is, the whole catalog is
    browsed through that filter (e.g. "products above 70% discount").

    Results are sorted by review count (or rating first when ``sort_by_rating``).
    """
    has_filter = (
        min_price is not None or max_price is not None or min_discount is not None
    )

    meaningful = [kw.lower().strip() for kw in (keywords or []) if len(kw.lower().strip()) >= 2]

    # One combined, lowercased searchable string per row (includes product_type
    # when present so type keywords match even if absent from the product name).
    combined = df["product_name"].astype(str) + " " + df["brand"].astype(str)
    if "product_type" in df.columns:
        combined = combined + " " + df["product_type"].astype(str)
    combined = (combined + " " + df["category"].astype(str)).str.lower()

    def _apply(mask: pd.Series) -> pd.DataFrame:
        res = df[mask]
        if min_price is not None:
            res = res[res["discounted_price"] >= min_price]
        if max_price is not None:
            res = res[res["discounted_price"] <= max_price]
        if min_discount is not None and "discount_pct" in res.columns:
            res = res[res["discount_pct"] >= min_discount]
        sort_cols = ["rating", "num_reviews"] if sort_by_rating else ["num_reviews", "rating"]
        avail = [c for c in sort_cols if c in res.columns]
        if avail:
            res = res.sort_values(by=avail, ascending=[False] * len(avail))
        return res

    # No keywords: only browse the catalog if a numeric filter was supplied.
    if not meaningful:
        if has_filter:
            return _apply(pd.Series(True, index=df.index)).head(limit)
        return pd.DataFrame()

    # Per-keyword hit masks: whole-word, plural-tolerant match.
    hits = [
        combined.str.contains(taxonomy.word_pattern(kw), na=False, regex=True)
        for kw in meaningful
    ]

    # AND mask: every keyword must hit.
    and_mask = hits[0].copy()
    for h in hits[1:]:
        and_mask = and_mask & h

    results = _apply(and_mask) if match_all else pd.DataFrame()

    # Graceful fallback to OR if the strict pass found nothing.
    if results.empty:
        or_mask = hits[0].copy()
        for h in hits[1:]:
            or_mask = or_mask | h
        results = _apply(or_mask)

    return results.head(limit)


def _name_matches(series: pd.Series, term: str) -> pd.Series:
    """Whole-word, plural-tolerant match of a term against a lowercased text series."""
    return series.str.contains(taxonomy.word_pattern(term), na=False, regex=True)


def structured_search(df: pd.DataFrame,
                      item: str | None = None,
                      color: str | None = None,
                      attributes: list[str] | None = None,
                      brand: str | None = None,
                      gender: str | None = None,
                      min_price: float | None = None,
                      max_price: float | None = None,
                      min_discount: float | None = None,
                      sort_by_rating: bool = False,
                      limit: int = 24) -> pd.DataFrame:
    """Precise, fact-based product search.

    HARD filters (define the candidate set): ``gender`` -> category, ``item`` ->
    product_type, ``brand``, and the numeric price/discount filters.
    SOFT signals (``color`` + ``attributes``): used to rank within candidates, and
    to filter only when at least one candidate actually matches — so a request for
    "brown caps" still shows caps even when no brown one exists (soft relaxation).

    Within an item type, products that only mention the item as an accessory
    ("Bath Robe With Belt") are ranked below genuine items ("Men Solid Belt").
    """
    res = df

    # gender -> category (include Unisex so men/women queries keep neutral items).
    if gender:
        g = str(gender).lower()
        if g in ("men", "women", "kids"):
            res = res[res["category"].astype(str).str.lower().isin([g, "unisex"])]

    # item -> product_type (accurate) with a name-contains fallback for unknowns.
    if item:
        canon = taxonomy.derive_product_type(str(item))
        typed = res.iloc[0:0]
        if "product_type" in res.columns and canon != "other":
            typed = res[res["product_type"] == canon]
        res = typed if not typed.empty else res[_name_matches(res["product_name"].astype(str).str.lower(), str(item))]

    # brand (only if it actually narrows things).
    if brand:
        narrowed = res[res["brand"].astype(str).str.lower().str.contains(str(brand).lower(), na=False, regex=False)]
        if not narrowed.empty:
            res = narrowed

    # numeric filters
    if min_price is not None:
        res = res[res["discounted_price"] >= min_price]
    if max_price is not None:
        res = res[res["discounted_price"] <= max_price]
    if min_discount is not None and "discount_pct" in res.columns:
        res = res[res["discount_pct"] >= min_discount]

    # soft tokens: color + attribute words.
    soft: list[str] = []
    if color:
        soft.append(str(color).lower())
    for attr in (attributes or []):
        soft.extend(t for t in str(attr).lower().split() if len(t) >= 2)

    base_sort = ["rating", "num_reviews"] if sort_by_rating else ["num_reviews", "rating"]
    sort_cols: list[str] = []

    if not res.empty:
        name_lower = res["product_name"].astype(str).str.lower()

        # Accessory down-ranking: "... with <item>" (e.g. robe with belt) ranks last.
        if item:
            accessory = (
                name_lower.str.contains(r"\bwith\b", na=False, regex=True)
                & name_lower.str.contains(taxonomy.word_pattern(str(item)), na=False, regex=True)
            )
            res = res.assign(_primary=(~accessory).astype(int))
            sort_cols.append("_primary")

        # Soft-token relevance score (color/attributes).
        if soft:
            score = sum(_name_matches(name_lower, t).astype(int) for t in soft)
            res = res.assign(_score=score)
            if (res["_score"] > 0).any():
                res = res[res["_score"] > 0]
            sort_cols.append("_score")

    sort_cols += base_sort
    sort_cols = [c for c in sort_cols if c in res.columns]
    if sort_cols and not res.empty:
        res = res.sort_values(by=sort_cols, ascending=False)

    res = res.drop(columns=[c for c in ("_primary", "_score") if c in res.columns])
    return res.head(limit)


def search_products(df: pd.DataFrame, query_input: str) -> str:
    """LLM-agent-compatible search: takes a string, returns markdown table.
    Format: 'keyword' OR 'keyword|max_price'
    """
    if not query_input or not isinstance(query_input, str) or query_input.strip() == "":
        return "Please provide a valid search keyword."

    parts = [p.strip() for p in query_input.split("|")]
    keywords = parts[0].split()
    max_price = None
    if len(parts) > 1:
        try:
            max_price = float(parts[1])
        except ValueError:
            pass

    results = find_products(df, keywords, max_price=max_price)

    if results.empty:
        msg = f"No products found matching '{parts[0]}'"
        if max_price:
            msg += f" under ₹{max_price}"
        return msg + "."

    cols = ["product_name", "brand", "discounted_price", "discount_pct", "rating"]
    if "image_url" in df.columns: cols.append("image_url")
    if "product_url" in df.columns: cols.append("product_url")

    return results[cols].to_markdown(index=False)
