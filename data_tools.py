import pandas as pd


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
                  limit: int = 10) -> pd.DataFrame:
    """Search products by matching keywords against product_name, brand, and category.

    Each keyword is matched independently (OR logic across columns, AND across keywords
    would be too restrictive, so we use OR across all keywords too — the more keywords
    match, the more likely it appears at the top via review-count sorting).

    Returns a DataFrame of matching products, sorted by num_reviews descending.
    """
    if not keywords:
        return pd.DataFrame()

    # Build a combined mask: product matches if ANY keyword hits ANY searchable column
    mask = pd.Series(False, index=df.index)
    for kw in keywords:
        kw = kw.lower().strip()
        if len(kw) < 2:
            continue
        name_hit = df["product_name"].str.lower().str.contains(kw, na=False, regex=False)
        brand_hit = df["brand"].str.lower().str.contains(kw, na=False, regex=False)
        category_hit = df["category"].str.lower().str.contains(kw, na=False, regex=False)
        mask = mask | name_hit | brand_hit | category_hit

    results = df[mask]

    # Apply price filters
    if min_price is not None:
        results = results[results["discounted_price"] >= min_price]
    if max_price is not None:
        results = results[results["discounted_price"] <= max_price]

    # Sort by popularity (num_reviews) then rating
    if "num_reviews" in results.columns:
        results = results.sort_values(
            by=["num_reviews", "rating"], ascending=[False, False]
        )

    return results.head(limit)


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
