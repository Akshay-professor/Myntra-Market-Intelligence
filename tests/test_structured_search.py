"""Tests for the fact-based structured search."""

import pandas as pd
import pytest

import taxonomy
import data_tools


@pytest.fixture
def catalog() -> pd.DataFrame:
    rows = [
        ("HRX Men Black Cap", "HRX", "Men", 499, 50),
        ("Puma Men Blue Cap", "Puma", "Men", 599, 40),
        ("Black Panther Women Capris", "Black Panther", "Women", 419, 60),
        ("Bobbi Brown Crushed Lip Colour", "Bobbi Brown", "Women", 2200, 0),
        ("Roadster Men Slim Fit Jeans", "Roadster", "Men", 1199, 40),
        ("Levis Women Skinny Jeans", "Levis", "Women", 1499, 30),
    ]
    df = pd.DataFrame(rows, columns=["product_name", "brand", "category",
                                     "discounted_price", "discount_pct"])
    df["rating"] = [4.1, 4.3, 3.9, 4.6, 4.2, 4.0]
    df["num_reviews"] = [100, 200, 150, 300, 500, 250]
    df["original_price"] = df["discounted_price"] * 2
    df["product_id"] = range(len(df))
    df["product_type"] = df["product_name"].apply(taxonomy.derive_product_type)
    return df


def test_item_uses_product_type_not_prefix(catalog):
    # "cap" must return caps only -- never Capris (prefix) or Bobbi Brown (brand).
    res = data_tools.structured_search(catalog, item="cap")
    names = list(res["product_name"])
    assert names, "should find caps"
    assert all("Cap" in n and "Capris" not in n for n in names)
    assert all("Lip" not in n for n in names)


def test_color_ranks_but_relaxes(catalog):
    # "black cap": black cap first, but blue cap still shown (soft relaxation).
    res = data_tools.structured_search(catalog, item="cap", color="black")
    names = list(res["product_name"])
    assert names[0] == "HRX Men Black Cap"


def test_color_with_no_match_still_returns_item(catalog):
    # No brown cap exists -> still return caps (don't go empty).
    res = data_tools.structured_search(catalog, item="cap", color="brown")
    assert not res.empty
    assert all("Cap" in n for n in res["product_name"])


def test_gender_filter(catalog):
    res = data_tools.structured_search(catalog, item="jeans", gender="women")
    assert list(res["product_name"]) == ["Levis Women Skinny Jeans"]


def test_price_filter(catalog):
    res = data_tools.structured_search(catalog, item="jeans", max_price=1300)
    assert all(p <= 1300 for p in res["discounted_price"])


def test_min_discount_filter(catalog):
    res = data_tools.structured_search(catalog, item="cap", min_discount=45)
    assert all(d >= 45 for d in res["discount_pct"])
