"""End-to-end check of the Day 1 pipeline on fake Favorita-shaped data.

Run with:  pytest -q
"""

from __future__ import annotations

import json

import pandas as pd

from ml.data.prepare import prepare
from ml.graph.build import build_graph
from tests.fake_favorita import write_fake_raw

TEMPLATE_COLUMNS = {
    "locations.csv": ["location_id", "name", "type", "city", "region", "latitude", "longitude", "supplied_by"],
    "products.csv": ["product_id", "name", "category", "unit_cost", "lead_time_days"],
    "sales.csv": ["date", "location_id", "product_id", "units_sold", "on_promotion"],
    "inventory.csv": ["location_id", "product_id", "on_hand", "on_order"],
}


def run_pipeline(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "processed"
    write_fake_raw(raw, days=150)
    meta = prepare(raw, out, days=120)
    stats = build_graph(out)
    return out, meta, stats


def test_day1_pipeline(tmp_path):
    out, meta, stats = run_pipeline(tmp_path)

    # Template files exist with exactly the upload-template columns.
    for name, cols in TEMPLATE_COLUMNS.items():
        df = pd.read_csv(out / name)
        assert list(df.columns) == cols, name

    locations = pd.read_csv(out / "locations.csv")
    stores = locations[locations["type"] == "store"]
    warehouses = locations[locations["type"] == "warehouse"]
    assert len(stores) == 54 and len(warehouses) == 3

    # Largest original city became Hyderabad, with unique area names and plausible coordinates.
    assert (stores["city"] == "Hyderabad").sum() == 18
    assert stores["name"].is_unique
    assert stores["latitude"].between(15.8, 19.95).all()
    assert stores["longitude"].between(77.2, 81.4).all()

    # Every store is supplied by a real warehouse.
    assert set(stores["supplied_by"]) <= set(warehouses["location_id"])

    # Sales: last 120 days only, no all-zero series (books mostly dropped).
    sales = pd.read_csv(out / "sales.csv", parse_dates=["date"])
    assert sales["date"].nunique() == 120
    assert (sales.groupby(["location_id", "product_id"])["units_sold"].sum() > 0).all()

    # Inventory covers stores and warehouses, with a mix of stock health.
    inv = pd.read_csv(out / "inventory.csv")
    assert set(warehouses["location_id"]) <= set(inv["location_id"])
    assert (inv["on_hand"] >= 0).all()

    # Calendar: national holiday kept, local holiday and work day excluded.
    cal = pd.read_csv(out / "calendar.csv")
    assert cal["is_holiday"].sum() == 1
    assert cal["oil_price"].notna().all()

    meta_file = json.loads((out / "meta.json").read_text())
    assert meta_file["stores"] == 54 and meta_file["currency"] == "INR"

    # Graph: every store connected, edges reference valid nodes.
    edges = pd.read_csv(out / "graph" / "edges.csv")
    assert stats["nodes"] == 54
    assert stats["min_degree"] >= 1
    assert edges[["source", "target"]].max().max() < 54
    assert (edges["source"] < edges["target"]).all()  # stored once per pair


def test_inventory_is_reproducible(tmp_path):
    out_a, _, _ = run_pipeline(tmp_path / "a")
    out_b, _, _ = run_pipeline(tmp_path / "b")
    a = pd.read_csv(out_a / "inventory.csv")
    b = pd.read_csv(out_b / "inventory.csv")
    pd.testing.assert_frame_equal(a, b)
