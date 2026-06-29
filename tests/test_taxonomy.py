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


def test_prefix_words_do_not_match_cap():
    # The classic bug: "cap" must not match cappuccino / captain / capris.
    assert taxonomy.derive_product_type("MCaffeine Cappuccino Body Wash") != "cap"
    assert taxonomy.derive_product_type("DailyObjects Captain Star iPhone Case") != "cap"
    assert taxonomy.derive_product_type("Black Panther Women Capris") != "cap"
    # ...but real caps (singular & plural) still classify as cap.
    assert taxonomy.derive_product_type("Roadster Unisex Printed Baseball Cap") == "cap"
    assert taxonomy.derive_product_type("Kook N Keech Unisex Caps") == "cap"


def test_word_pattern_whole_word():
    import re
    pat = taxonomy.word_pattern("cap")
    assert re.search(pat, "baseball cap")
    assert re.search(pat, "printed caps")
    assert not re.search(pat, "cappuccino body wash")
    assert not re.search(pat, "captain america")


def test_expand_synonyms():
    assert taxonomy.expand_synonyms(["denim"]) == ["jeans"]
    assert taxonomy.expand_synonyms(["tee"]) == ["tshirt"]
    # Dedupes and preserves unknown words.
    assert taxonomy.expand_synonyms(["denim", "jeans", "blue"]) == ["jeans", "blue"]


def test_vocab_contains_core_terms():
    for term in ["jeans", "tshirt", "kurta", "sneakers", "bikini", "denim"]:
        assert term in taxonomy.PRODUCT_VOCAB
