"""Tests for catalog enrichment (mining product_url + product_name)."""

import pandas as pd

import enrich


def test_parse_url():
    url = ("https://www.myntra.com/tshirts/roadster/"
           "roadster-men-navy-blue-typography-printed-cotton-t-shirt/20307178/buy")
    url_type, slug = enrich.parse_url(url)
    assert url_type == "tshirts"
    assert "navy-blue" in slug and "cotton" in slug


def test_parse_url_malformed():
    assert enrich.parse_url("not a url") == ("", "")


def _df(rows):
    df = pd.DataFrame(rows, columns=["product_name", "category", "product_url"])
    return enrich.enrich_dataframe(df)


def test_enrich_from_url_attributes():
    # product_name is sparse; the URL slug carries gender + color + material.
    out = _df([
        ("Printed Round Neck T-shirt", "Unisex",
         "https://www.myntra.com/tshirts/fido-dido/fido-dido-men-black-printed-round-neck-t-shirt/6781215/buy"),
    ])
    row = out.iloc[0]
    assert row["product_type"] == "tshirt"   # singularised from /tshirts/
    assert row["category"] == "Men"          # from slug, not the "Unisex" guess
    assert row["color"] == "black"
    assert row["pattern"] == "printed"


def test_enrich_material_and_fit():
    out = _df([
        ("Slim Fit Jeans", "Unisex",
         "https://www.myntra.com/jeans/roadster/roadster-men-blue-slim-fit-cotton-jeans/123/buy"),
    ])
    row = out.iloc[0]
    assert row["product_type"] == "jean"
    assert row["color"] == "blue"
    assert row["material"] == "cotton"
    assert row["fit"] == "slim"


def test_enrich_idempotent():
    out = _df([("Solid Top", "Women", "https://www.myntra.com/tops/x/x-women-red-solid-top/1/buy")])
    again = enrich.enrich_dataframe(out)
    assert list(again.columns) == list(out.columns)
    assert again.iloc[0]["color"] == "red"


def test_attribute_column_buckets():
    assert enrich.attribute_column("black") == "color"
    assert enrich.attribute_column("cotton") == "material"
    assert enrich.attribute_column("slim") == "fit"
    assert enrich.attribute_column("printed") == "pattern"
    assert enrich.attribute_column("roadster") is None
