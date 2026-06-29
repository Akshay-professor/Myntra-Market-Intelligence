"""Tests for the TF-IDF semantic search layer."""

import pytest

import semantic_search

# Skip the whole module gracefully if sklearn isn't installed.
pytestmark = pytest.mark.skipif(
    not semantic_search.is_available(), reason="scikit-learn not installed"
)


def test_make_corpus_includes_columns(sample_df):
    corpus = semantic_search.make_corpus(sample_df)
    assert len(corpus) == len(sample_df)
    assert "jeans" in corpus[0].lower()


def test_typo_tolerance(sample_df):
    index = semantic_search.build_index(semantic_search.make_corpus(sample_df))
    # "jeanss" (typo) should still rank the jeans products at the top.
    positions = semantic_search.rank(index, "jeanss", top_n=3)
    assert positions
    top_names = [sample_df.iloc[p]["product_name"] for p in positions]
    assert any("Jeans" in n for n in top_names)


def test_rank_empty_for_blank_query(sample_df):
    index = semantic_search.build_index(semantic_search.make_corpus(sample_df))
    assert semantic_search.rank(index, "") == []


def test_rank_handles_missing_index():
    assert semantic_search.rank(None, "jeans") == []
