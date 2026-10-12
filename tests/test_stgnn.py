"""ST-GNN tests. Data preparation needs only numpy; model tests are skipped without PyTorch.

Run with:  pytest -q
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ml.data.load import load_dataset
from ml.data.prepare import prepare
from ml.graph.build import build_graph
from ml.models.stgnn.data import HORIZON, LOOKBACK, build, normalised_adjacency, split_origins, windows
from tests.fake_favorita import write_fake_raw


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory):
    base = tmp_path_factory.mktemp("stgnn")
    write_fake_raw(base / "raw", days=260)
    prepare(base / "raw", base / "processed", days=240)
    build_graph(base / "processed")
    return base / "processed"


@pytest.fixture(scope="module")
def stdata(dataset_dir):
    ds = load_dataset(dataset_dir)
    return ds, build(ds, dataset_dir / "graph")


# ---------- data preparation (numpy only) ----------

def test_adjacency_is_symmetric_and_normalised():
    edges = pd.DataFrame({"source": [0, 1], "target": [1, 2], "weight": [1.0, 0.5]})
    a = normalised_adjacency(3, edges)
    assert np.allclose(a, a.T)
    assert a[0, 2] == 0 and a[0, 1] > 0
    assert np.all(np.linalg.eigvalsh(a) <= 1 + 1e-6)  # normalised: no amplification


def test_cube_matches_sales(stdata):
    ds, data = stdata
    n, p = len(data.stores), len(data.products)
    assert data.sales.shape == (len(data.dates), n, p)
    assert len(data.dates) == ds.sales["date"].nunique() + HORIZON
    assert data.mask.sum() == ds.sales.groupby(["location_id", "product_id"]).ngroups
    row = ds.sales.iloc[1234]
    i, j = data.stores.index(row.location_id), data.products.index(row.product_id)
    assert data.sales[data.index_of(row.date), i, j] == pytest.approx(row.units_sold, rel=1e-5)


def test_windows_align_with_dates(stdata):
    ds, data = stdata
    cutoff = data.index_of(ds.as_of) - HORIZON
    train, val = split_origins(cutoff)
    assert val.max() + HORIZON <= cutoff
    assert train.max() + HORIZON < val.min() + 1  # training targets end before validation targets start
    scale = data.scale(cutoff)
    w = windows(data, val, scale)
    assert w["x_sales"].shape == (len(val), LOOKBACK, len(data.stores), len(data.products))
    o = val[0]
    assert np.allclose(w["y"][0] * scale, data.sales[o + 1:o + 1 + HORIZON])
    assert np.allclose(w["x_sales"][0, -1] * scale, data.sales[o])


# ---------- model and training (need PyTorch) ----------

def test_model_shapes_and_weekday_index():
    torch = pytest.importorskip("torch")
    from ml.models.stgnn.model import STGNN, same_weekday_index

    idx = same_weekday_index()
    for k in range(1, HORIZON + 1):
        for j in idx[k - 1]:
            gap = k + (LOOKBACK - 1 - j)       # days between the lookback day and the target day
            assert gap % 7 == 0 and gap >= k

    model = STGNN(n_nodes=5, n_products=3, hidden=8)
    out = model(torch.rand(2, LOOKBACK, 5, 3), torch.rand(2, LOOKBACK, 9), torch.rand(2, HORIZON, 9), torch.eye(5))
    assert out.shape == (2, HORIZON, 5, 3)
    assert (out >= 0).all()


def test_training_end_to_end(dataset_dir):
    pytest.importorskip("torch")
    from ml.models.stgnn.train import train
    from ml.pipeline import run

    run(dataset_dir, train_days=120, verbose=False)
    report = train(dataset_dir, epochs=2, patience=2, hidden=16, verbose=False)
    out = dataset_dir / "results"

    assert {"stgnn", "baseline", "naive"} <= set(report["holdout_wape"])
    assert 0 < report["holdout_wape"]["stgnn"] < 2
    fc = pd.read_csv(out / "forecasts_stgnn.csv")
    assert fc["date"].nunique() == HORIZON and (fc["p10"] <= fc["p50"] + 1e-9).all()
    assert "stgnn" in pd.read_csv(out / "backtest.csv").columns
    assert "stgnn" in set(pd.read_csv(out / "metrics.csv")["model"])
    saved = json.loads((out / "stgnn_training.json").read_text())
    assert saved["best_epoch"] >= 1 and len(saved["history"]) >= 1
