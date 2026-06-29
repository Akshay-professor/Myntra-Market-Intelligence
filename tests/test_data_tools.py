"""Tests for the product search engine (data_tools.find_products)."""

import data_tools


def _names(df):
    return list(df["product_name"])


def test_word_boundary_men_not_women(sample_df):
    # "men" must not match "women"/"girls" products.
    res = data_tools.find_products(sample_df, ["men", "tshirt"])
    assert res is not None
    for name in _names(res):
        assert "Women" not in name and "Girls" not in name


def test_word_boundary_tshirt_not_sweatshirt(sample_df):
    res = data_tools.find_products(sample_df, ["tshirt"])
    names = _names(res)
    assert names, "should find t-shirts"
    assert all("Sweatshirt" not in n for n in names)


def test_black_jeans_excludes_cosmetics(sample_df):
    # The classic bug: "black jeans" should not return black mascara/kajal.
    res = data_tools.find_products(sample_df, ["black", "jeans"])
    names = _names(res)
    assert names == ["HIGHLANDER Men Black Tapered Jeans"]


def test_and_then_or_fallback(sample_df):
    # "zzz" matches nothing; AND yields empty, OR fallback should still find jeans.
    res = data_tools.find_products(sample_df, ["zzz", "jeans"])
    assert any("Jeans" in n for n in _names(res))


def test_max_price_filter(sample_df):
    res = data_tools.find_products(sample_df, ["jeans"], max_price=1000)
    assert all(p <= 1000 for p in res["discounted_price"])


def test_min_price_filter(sample_df):
    res = data_tools.find_products(sample_df, ["tshirt"], min_price=1000)
    assert all(p >= 1000 for p in res["discounted_price"])


def test_min_discount_filter(sample_df):
    res = data_tools.find_products(sample_df, ["tshirt"], min_discount=40)
    assert all(d >= 40 for d in res["discount_pct"])


def test_keywordless_browse_requires_filter(sample_df):
    # No keywords and no filter -> empty.
    assert data_tools.find_products(sample_df, []).empty
    # No keywords but a discount filter -> browse the whole catalog.
    res = data_tools.find_products(sample_df, [], min_discount=70)
    assert not res.empty
    assert all(d >= 70 for d in res["discount_pct"])


def test_sort_by_rating(sample_df):
    res = data_tools.find_products(sample_df, ["tshirt"], sort_by_rating=True)
    ratings = list(res["rating"])
    assert ratings == sorted(ratings, reverse=True)


def test_limit_respected(sample_df):
    res = data_tools.find_products(sample_df, [], min_discount=0, limit=3)
    assert len(res) <= 3
