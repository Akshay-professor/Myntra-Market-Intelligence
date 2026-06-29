"""Tests for the bare-brand-query override."""

import agent


def _facts(**kw):
    base = {"intent": "product_search", "item": None, "color": None,
            "attributes": [], "brand": None, "gender": None,
            "min_price": None, "max_price": None, "min_discount": None}
    base.update(kw)
    return base


def test_bare_brand_overrides_hallucinated_item():
    facts = _facts(item="watch")  # LLM wrongly guessed "watch" for "roadster"
    out = agent.apply_brand_override(facts, "roadster", {"roadster", "puma", "nike"})
    assert out["brand"] == "roadster"
    assert out["item"] is None
    assert out["intent"] == "product_search"


def test_non_brand_query_unchanged():
    facts = _facts(item="jeans")
    out = agent.apply_brand_override(facts, "blue jeans", {"roadster", "puma"})
    assert out == facts


def test_brand_plus_item_not_overridden():
    # "puma shoes" is not a bare brand, so keep the LLM's parse.
    facts = _facts(item="shoes", brand="puma")
    out = agent.apply_brand_override(facts, "puma shoes", {"puma"})
    assert out["item"] == "shoes"


def test_empty_brand_set_is_safe():
    facts = _facts(item="watch")
    assert agent.apply_brand_override(facts, "roadster", set()) == facts
