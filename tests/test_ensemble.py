"""Blend tests. Uses stand-in ST-GNN predictions, so PyTorch is not needed.

Run with:  pytest -q
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ml.data.prepare import prepare
from ml.models.ensemble import blend
from ml.pipeline import run
from tests.fake_favorita import write_fake_raw


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    base = tmp_path_factory.mktemp("ensemble")
    write_fake_raw(base / "raw", days=200)
    data = base / "processed"
    prepare(base / "raw", data, days=180)
    run(data, train_days=120, verbose=False)
    out = data / "results"

    # Stand-in ST-GNN: the naive forecast for the holdout, a scaled baseline for the future.
    bt = pd.read_csv(out / "backtest.csv")
    bt["stgnn"] = bt["naive"]
    bt.to_csv(out / "backtest.csv", index=False)
    fs = pd.read_csv(out / "forecasts.csv")
    fs["p50"] = fs["p50"] * 1.2
    fs["model"] = "stgnn"
    fs.to_csv(out / "forecasts_stgnn.csv", index=False)
    metrics = pd.read_csv(out / "metrics.csv")
    naive_rows = metrics[metrics["model"] == "naive"].assign(model="stgnn")
    pd.concat([metrics, naive_rows]).to_csv(out / "metrics.csv", index=False)
    return data, out


def test_blend_scores_and_forecasts(results):
    data, out = results
    result = blend(data, verbose=False)

    bt = pd.read_csv(out / "backtest.csv")
    assert np.allclose(bt["ensemble"], 0.5 * bt["stgnn"] + 0.5 * bt["baseline"], atol=1e-3)  # file keeps 3 decimals
    assert set(result["holdout_wape"]) >= {"baseline", "stgnn", "ensemble", "naive"}
    assert result["best_model"] == min(("baseline", "stgnn", "ensemble"), key=result["holdout_wape"].get)

    fe = pd.read_csv(out / "forecasts_ensemble.csv")
    fb = pd.read_csv(out / "forecasts.csv")
    merged = fe.merge(fb, on=["location_id", "product_id", "date"], suffixes=("_e", "_b"))
    assert len(merged) == len(fb)
    assert np.allclose(merged["p50_e"], merged["p50_b"] * 1.1, atol=2e-3)  # 0.5 * 1.2x + 0.5 * 1x
    assert (fe["p10"] <= fe["p50"] + 1e-9).all() and (fe["p50"] <= fe["p90"] + 1e-9).all()


def test_blend_is_rerunnable_and_drives_recommendations(results):
    data, out = results
    blend(data, verbose=False)
    result = blend(data, model_for_recommendations="ensemble", verbose=False)
    metrics = pd.read_csv(out / "metrics.csv")
    assert (metrics[(metrics["model"] == "ensemble") & (metrics["location_id"] == "ALL")].shape[0]) == 1
    summary = json.loads((out / "summary.json").read_text())
    assert summary["forecast_model"] == "ensemble" == result["recommendation_model"]
    assert len(pd.read_csv(out / "recommendations.csv")) > 0
