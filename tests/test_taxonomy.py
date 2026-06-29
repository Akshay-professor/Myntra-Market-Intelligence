"""Tests for the product-type taxonomy."""

import taxonomy


def test_derive_basic_types():
    cases = {
        "Roadster Men Slim Fit Jeans": "jeans",
        "HRX Pack Of 2 Printed T-shirts": "tshirt",
        "HRX Women Solid Sweatshirt": "sweatshirt",
        "Anouk Women Kurta": "kurta",
        "Jockey Women Bikini Briefs": "bikini",
        "Lakme Kajal - Black": "kajal",
        "Nike Men Running Shoes": "shoes",
        "Titan Analog Watch": "watch",
    }
    for name, expected in cases.items():
        assert taxonomy.derive_product_type(name) == expected, name


def test_specific_type_wins_over_generic():
    # "sweatshirt" must not be classified as the generic "shirt".
    assert taxonomy.derive_product_type("Solid Sweatshirt") == "sweatshirt"
    # A denim *jacket* is a jacket, not jeans.
    assert taxonomy.derive_product_type("Levis Denim Jacket") == "jacket"


def test_unknown_returns_other():
    assert taxonomy.derive_product_type("Mystery Gadget XYZ") == "other"


def test_expand_synonyms():
    assert taxonomy.expand_synonyms(["denim"]) == ["jeans"]
    assert taxonomy.expand_synonyms(["tee"]) == ["tshirt"]
    # Dedupes and preserves unknown words.
    assert taxonomy.expand_synonyms(["denim", "jeans", "blue"]) == ["jeans", "blue"]


def test_vocab_contains_core_terms():
    for term in ["jeans", "tshirt", "kurta", "sneakers", "bikini", "denim"]:
        assert term in taxonomy.PRODUCT_VOCAB
