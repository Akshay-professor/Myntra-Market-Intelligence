"""Tests for the LLM fact-normalization (pure, no network)."""

import agent


def test_normalize_full_facts():
    raw = {
        "intent": "product_search", "item": "Jeans", "color": "Black",
        "attributes": ["Slim Fit"], "brand": None, "gender": "Men",
        "min_price": None, "max_price": "2000", "min_discount": None,
    }
    facts = agent._normalize_facts(raw)
    assert facts["intent"] == "product_search"
    assert facts["item"] == "jeans"
    assert facts["color"] == "black"
    assert facts["attributes"] == ["slim fit"]
    assert facts["gender"] == "men"
    assert facts["max_price"] == 2000


def test_normalize_handles_string_nulls():
    raw = {"intent": "product_search", "item": "null", "color": "None", "gender": "unknown"}
    facts = agent._normalize_facts(raw)
    assert facts["item"] is None
    assert facts["color"] is None
    assert facts["gender"] is None


def test_normalize_attributes_as_string():
    facts = agent._normalize_facts({"intent": "product_search", "attributes": "cotton"})
    assert facts["attributes"] == ["cotton"]


def test_normalize_invalid_intent_defaults():
    # No item, bad intent -> out_of_scope.
    assert agent._normalize_facts({"intent": "weird"})["intent"] == "out_of_scope"
    # Bad intent but has an item -> product_search.
    assert agent._normalize_facts({"intent": "weird", "item": "shoes"})["intent"] == "product_search"


def test_normalize_non_dict():
    assert agent._normalize_facts("garbage")["intent"] == "out_of_scope"
