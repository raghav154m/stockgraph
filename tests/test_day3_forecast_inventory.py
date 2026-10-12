"""Tests for the dataset loader, baseline model, inventory optimizer and pipeline.

Run with:  pytest -q
"""

from __future__ import annotations

import json
import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from ml.data.load import complete_panel
from ml.data.prepare import prepare
from ml.data.telangana import haversine_km
from ml.inventory.optimizer import _days_until_stockout, _demand_over, _rupees, transport_days
from ml.pipeline import run
from tests.fake_favorita import write_fake_raw


# ---------- loader ----------

def test_missing_day_is_marked_closed():
    dates = pd.date_range("2017-01-01", periods=5)
    sales = pd.DataFrame({
        "date": list(dates) * 2,
        "location_id": ["A"] * 5 + ["B"] * 5,
        "product_id": "p",
        "units_sold": [1.0] * 10,
    })
    sales = sales[sales["date"] != dates[2]]                           # missing for every series
    sales = sales[~((sales["location_id"] == "B") & (sales["date"] == dates[3]))]  # missing for one
    panel = complete_panel(sales)
    assert len(panel) == 10
    closed = panel[panel["is_closed"] == 1]
    assert set(closed["date"]) == {dates[2]}
    gap = panel[(panel["location_id"] == "B") & (panel["date"] == dates[3])].iloc[0]
    assert gap["units_sold"] == 0 and gap["is_closed"] == 0


# ---------- optimizer helpers ----------

def test_transport_days():
    assert transport_days(10) == 1
    assert transport_days(150) == 2
    assert transport_days(300) == 3


def test_demand_and_stockout_helpers():
    cum = np.cumsum(np.full(28, 10.0))           # 10 units a day
    assert _demand_over(cum, 3) == pytest.approx(30)
    assert _demand_over(cum, 2.5) == pytest.approx(25)
    assert _demand_over(cum, 35) == pytest.approx(350)  # extrapolated past day 28
    assert _days_until_stockout(cum, 45) == pytest.approx(4.5)
    assert _days_until_stockout(cum, 0) == 0
    assert _days_until_stockout(cum, 400) == pytest.approx(40)
    assert math.isinf(_days_until_stockout(np.zeros(28), 5))


def test_rupee_format():
    assert _rupees(950) == "₹950"
    assert _rupees(1234567) == "₹12,34,567"


# ---------- full pipeline on fake data ----------

@pytest.fixture(scope="module")
def results(tmp_path_factory):
    base = tmp_path_factory.mktemp("day3")
    write_fake_raw(base / "raw", days=200)
    prepare(base / "raw", base / "processed", days=180)
    summary = run(base / "processed", train_days=120, verbose=False)
    return base / "processed", summary


def test_pipeline_outputs(results):
    data, summary = results
    out = data / "results"
    for name in ("forecasts.csv", "backtest.csv", "metrics.csv", "inventory_status.csv",
                 "recommendations.csv", "summary.json"):
        assert (out / name).exists(), name
    assert json.loads((out / "summary.json").read_text())["as_of"] == summary["as_of"]


def test_forecasts(results):
    data, summary = results
    fc = pd.read_csv(data / "results" / "forecasts.csv", parse_dates=["date"])
    sales = pd.read_csv(data / "sales.csv")
    assert fc["date"].nunique() == 28
    assert fc["date"].min() == pd.Timestamp(summary["as_of"]) + pd.Timedelta(days=1)
    assert len(fc) == 28 * sales.groupby(["location_id", "product_id"]).ngroups
    assert (fc["p10"] <= fc["p50"] + 1e-9).all() and (fc["p50"] <= fc["p90"] + 1e-9).all()
    assert (fc["p10"] >= 0).all()


def test_model_is_scored_against_naive(results):
    data, summary = results
    metrics = pd.read_csv(data / "results" / "metrics.csv")
    assert set(metrics["model"]) == {"baseline", "naive"}
    assert 0 < summary["wape"]["baseline"] < 1


def test_safety_stock_formula(results):
    data, _ = results
    status = pd.read_csv(data / "results" / "inventory_status.csv")
    fc = pd.read_csv(data / "results" / "forecasts.csv")
    row = status[status["supplied_by"] != "SUPPLIER"].iloc[0]
    sigma = fc[(fc["location_id"] == row.location_id) & (fc["product_id"] == row.product_id)]["sigma"].iloc[0]
    expected = NormalDist().inv_cdf(0.95) * sigma * math.sqrt(row.lead_time_days + 7)
    assert row.safety_stock == pytest.approx(expected, abs=0.1)
    assert (status["reorder_point"] <= status["target_level"] + 1e-9).all()


def test_recommendations_are_sensible(results):
    data, _ = results
    recs = pd.read_csv(data / "results" / "recommendations.csv")
    status = pd.read_csv(data / "results" / "inventory_status.csv").set_index(["location_id", "product_id"])
    locations = pd.read_csv(data / "locations.csv").set_index("location_id")

    assert len(recs) > 0
    assert recs["rec_id"].is_unique
    assert (recs["quantity"] > 0).all()
    assert set(recs["priority"]) <= {"high", "medium", "low"}
    assert recs["reason"].str.len().gt(40).all()
    assert not recs["reason"].str.contains(r"\(s\)|\b1 days\b").any()

    valid = set(locations.index) | {"SUPPLIER"}
    assert set(recs["from_location"]) <= valid and set(recs["to_location"]) <= set(locations.index)

    transfers = recs[recs["type"] == "transfer"]
    for t in transfers.itertuples():
        assert status.loc[(t.from_location, t.product_id), "status"] == "overstock"
        assert status.loc[(t.to_location, t.product_id), "status"] in {"stockout_risk", "reorder_due"}
        a, b = locations.loc[t.from_location], locations.loc[t.to_location]
        assert haversine_km(a.latitude, a.longitude, b.latitude, b.longitude) <= 120

    # High priority = will run out before a normal delivery can arrive.
    risky = status[status["status"] == "stockout_risk"].index
    high = recs[(recs["priority"] == "high") & (recs["from_location"] != "SUPPLIER")]
    assert all((r.to_location, r.product_id) in risky for r in high.itertuples())
