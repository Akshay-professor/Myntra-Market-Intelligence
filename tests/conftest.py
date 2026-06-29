"""Shared pytest fixtures."""

import os
import sys

import pandas as pd
import pytest

# Make project modules importable when running `pytest` from any directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import taxonomy  # noqa: E402


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A small but representative catalog covering the tricky cases."""
    rows = [
        # name, brand, category, discounted_price, original_price, discount_pct, rating, reviews
        ("Roadster Men Slim Fit Jeans", "Roadster", "Men", 1199, 1999, 40, 4.1, 500),
        ("HIGHLANDER Men Black Tapered Jeans", "HIGHLANDER", "Men", 849, 1699, 50, 4.0, 300),
        ("FACES CANADA Black Magneteyes Mascara", "FACES CANADA", "Unisex", 199, 399, 50, 3.9, 200),
        ("Lakme Kajal - Black", "Lakme", "Unisex", 72, 80, 10, 3.8, 150),
        ("HRX Pack Of 2 Printed T-shirts", "HRX", "Unisex", 1598, 1600, 0, 4.2, 900),
        ("Puma Men Printed Polo T-shirts", "Puma", "Men", 1374, 2499, 45, 4.3, 700),
        ("HRX Women Solid Sweatshirt", "HRX", "Women", 1249, 2499, 50, 4.5, 600),
        ("SUGR Girl Girls Pack of 3 Cotton Tshirts", "SUGR Girl", "Women", 474, 1899, 75, 3.7, 80),
        ("Jockey Women Solid Bikini Briefs", "Jockey", "Women", 219, 219, 0, 4.0, 400),
        ("Nike Men Running Shoes", "Nike", "Men", 2799, 3999, 30, 4.6, 1200),
    ]
    cols = ["product_name", "brand", "category", "discounted_price",
            "original_price", "discount_pct", "rating", "num_reviews"]
    df = pd.DataFrame(rows, columns=cols)
    df["product_id"] = range(len(df))
    df["image_url"] = "-"
    df["product_url"] = "http://example.com"
    df["product_type"] = df["product_name"].apply(taxonomy.derive_product_type)
    return df
