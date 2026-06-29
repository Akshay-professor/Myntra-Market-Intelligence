"""Tests for the intent classifier rule fast-paths.

The cheap LLM fallback is monkeypatched so tests are deterministic and offline.
"""

import pytest

import classifier


@pytest.fixture(autouse=True)
def stub_llm(monkeypatch):
    """Replace the LLM fallback with a sentinel so we can see when rules fall through."""
    monkeypatch.setattr(classifier, "classify_intent_llm", lambda q: "LLM_FALLBACK")


@pytest.mark.parametrize("text", ["hi", "hello", "hey there", "how are you", "thanks"])
def test_greetings(text):
    assert classifier.classify_query(text) == classifier.GREETING


@pytest.mark.parametrize("text", [
    "black jeans",
    "shirts under 2500",
    "expensive shirts above 2000",
    "bikini",
    "show me kurtas",
    "men tshirts under 500",
    "above 70% discount",
])
def test_product_searches(text):
    assert classifier.classify_query(text) == classifier.PRODUCT_SEARCH


@pytest.mark.parametrize("text", [
    "top brands by discount",
    "average price per category",
    "rating distribution",
    "brand performance summary",
])
def test_analytics(text):
    assert classifier.classify_query(text) == classifier.ANALYTICS


@pytest.mark.parametrize("text", [
    "who is india's pm?",
    "tell me a joke",
    "what is 2+2",
])
def test_ambiguous_falls_through_to_llm(text):
    # Rules can't decide -> LLM fallback is invoked.
    assert classifier.classify_query(text) == "LLM_FALLBACK"


def test_empty_query_is_out_of_scope():
    assert classifier.classify_query("   ") == classifier.OUT_OF_SCOPE
