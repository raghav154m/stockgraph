"""Backend tests: load pipeline results into SQLite and call every endpoint.

Uses the fake Favorita data, the real baseline pipeline, and a stand-in for the ST-GNN output
(so the blend and all three forecast files are exercised without PyTorch).

Run with:  pytest -q tests/test_backend.py
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from backend.config import Settings  # noqa: E402
from backend.db import init_db, make_engine  # noqa: E402
from backend.loader import load_dataset_into_db  # noqa: E402
from backend.main import create_app  # noqa: E402
from ml.data.prepare import prepare  # noqa: E402
from ml.graph.build import build_graph  # noqa: E402
from ml.models.ensemble import blend  # noqa: E402
from ml.pipeline import run  # noqa: E402
from tests.fake_favorita import write_fake_raw, write_fake_stgnn_outputs  # noqa: E402


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    base = tmp_path_factory.mktemp("backend")
    write_fake_raw(base / "raw", days=200)
    prepare(base / "raw", base / "processed", days=180)
    build_graph(base / "processed")
    run(base / "processed", train_days=120, verbose=False)
    write_fake_stgnn_outputs(base / "processed" / "results")
    blend(base / "processed", verbose=False)    # ensemble drives recommendations, as in the demo
    return base / "processed"


@pytest.fixture(scope="module")
def client(data_dir, tmp_path_factory):
    db = tmp_path_factory.mktemp("db") / "test.db"
    settings = Settings(db_url=f"sqlite:///{db}", data_dir=data_dir, autoload_demo=True,
                        cors_origins=["http://localhost:5173"])
    with TestClient(create_app(settings)) as c:   # startup loads the demo dataset
        yield c


# ---------- loader ----------

def test_loader_counts_match_files(data_dir, tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'load.db'}")
    init_db(engine)
    counts = load_dataset_into_db(engine, data_dir, "demo", history_days=30)
    results = data_dir / "results"
    n_series = len(pd.read_csv(data_dir / "inventory.csv"))
    assert counts["locations"] == len(pd.read_csv(data_dir / "locations.csv"))
    assert counts["recommendations"] == len(pd.read_csv(results / "recommendations.csv"))
    assert counts["inventory_status"] == n_series
    stores_x_products = len(pd.read_csv(results / "forecasts.csv"))
    assert counts["forecasts"] == 3 * stores_x_products          # baseline + stgnn + ensemble
    sales = pd.read_csv(data_dir / "sales.csv", parse_dates=["date"])
    assert counts["sales"] == (sales["date"] > sales["date"].max() - pd.Timedelta(days=30)).sum()
    assert counts["metrics_rolling"] == 8
    assert counts["graph_edges"] == len(pd.read_csv(data_dir / "graph" / "edges.csv"))

    again = load_dataset_into_db(engine, data_dir, "demo", history_days=30)   # reload replaces, no duplicates
    assert {k: v for k, v in again.items() if k != "seconds"} == {k: v for k, v in counts.items() if k != "seconds"}


# ---------- endpoints ----------

def test_health_and_datasets(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    datasets = client.get("/api/datasets").json()
    assert [d["id"] for d in datasets] == ["demo"]
    assert datasets[0]["forecast_model"] == "ensemble"


def test_overview(client, data_dir):
    body = client.get("/api/datasets/demo").json()
    summary = json.loads((data_dir / "results" / "summary.json").read_text())
    assert body["summary"] == summary
    assert body["stores"] == 54 and body["warehouses"] == 3
    assert body["products"] == len(pd.read_csv(data_dir / "products.csv"))
    assert "simulated" in body["disclosure"]
    assert client.get("/api/datasets/nope").status_code == 404


def test_locations(client):
    locs = client.get("/api/datasets/demo/locations").json()
    assert len(locs) == 57
    st = next(l for l in locs if l["location_id"] == "ST-01")
    assert set(st["status_counts"]) == {"stockout_risk", "reorder_due", "overstock", "healthy"}
    assert sum(st["status_counts"].values()) > 0
    assert st["worst_status"] in st["status_counts"]
    warehouses = client.get("/api/datasets/demo/locations", params={"type": "warehouse"}).json()
    assert len(warehouses) == 3 and all(w["supplied_by"] == "SUPPLIER" for w in warehouses)
    assert client.get("/api/datasets/demo/locations", params={"type": "shop"}).status_code == 422

    detail = client.get("/api/datasets/demo/locations/ST-01").json()
    assert detail["name"] == st["name"]
    order = ["stockout_risk", "reorder_due", "overstock", "healthy"]
    ranks = [order.index(i["status"]) for i in detail["inventory"]]
    assert ranks == sorted(ranks)                               # most urgent first
    assert {a["model"] for a in detail["accuracy"]} >= {"baseline", "ensemble"}
    assert client.get("/api/datasets/demo/locations/ST-99").status_code == 404


def test_products(client, data_dir):
    products = client.get("/api/datasets/demo/products").json()
    assert len(products) == len(pd.read_csv(data_dir / "products.csv"))
    status = pd.read_csv(data_dir / "results" / "inventory_status.csv")
    stores = status[status["supplied_by"] != "SUPPLIER"]
    p = products[0]
    expected = stores[stores["product_id"] == p["product_id"]]
    assert p["stores_at_risk"] == (expected["status"] == "stockout_risk").sum()
    assert p["store_on_hand"] == pytest.approx(expected["on_hand"].sum(), abs=0.1)


def test_inventory_filters(client, data_dir):
    status = pd.read_csv(data_dir / "results" / "inventory_status.csv")
    body = client.get("/api/datasets/demo/inventory", params={"limit": 5000}).json()
    assert body["total"] == len(status) == len(body["items"])
    risky = client.get("/api/datasets/demo/inventory", params={"status": "stockout_risk"}).json()
    assert risky["total"] == (status["status"] == "stockout_risk").sum()
    assert all(i["status"] == "stockout_risk" for i in risky["items"])
    wh = client.get("/api/datasets/demo/inventory", params={"location_type": "warehouse", "limit": 5000}).json()
    assert wh["total"] == (status["supplied_by"] == "SUPPLIER").sum()
    page = client.get("/api/datasets/demo/inventory", params={"limit": 10, "offset": 10}).json()
    assert len(page["items"]) == 10 and page["items"][0]["location_name"]


def test_forecast_for_one_store(client, data_dir):
    body = client.get("/api/datasets/demo/forecast",
                      params={"location_id": "ST-01", "product_id": "automotive", "history_days": 28}).json()
    assert set(body["forecasts"]) == {"baseline", "stgnn", "ensemble"}
    assert body["recommendation_model"] == "ensemble"
    ens = body["forecasts"]["ensemble"]
    assert len(ens) == 28 and all(d["p10"] <= d["p50"] <= d["p90"] for d in ens)
    assert len(body["history"]) == 28 and body["history"][-1]["date"] < ens[0]["date"]
    assert len(body["backtest"]) == 28 and body["backtest"][0]["stgnn"] is not None
    assert body["inventory"]["location_id"] == "ST-01"

    expected = pd.read_csv(data_dir / "results" / "forecasts_ensemble.csv")
    series = expected[(expected["location_id"] == "ST-01") & (expected["product_id"] == "automotive")]
    first = series.sort_values("date").iloc[0]
    assert ens[0]["p50"] == pytest.approx(first["p50"], abs=1e-3)

    only = client.get("/api/datasets/demo/forecast",
                      params={"location_id": "ST-01", "product_id": "automotive", "models": "baseline"}).json()
    assert set(only["forecasts"]) == {"baseline"}


def test_forecast_network_total(client, data_dir):
    body = client.get("/api/datasets/demo/forecast",
                      params={"location_id": "ALL", "product_id": "automotive", "models": "baseline"}).json()
    fc = pd.read_csv(data_dir / "results" / "forecasts.csv")
    day1 = fc[(fc["product_id"] == "automotive") & (fc["date"] == fc["date"].min())]
    total = body["forecasts"]["baseline"][0]
    assert total["p50"] == pytest.approx(day1["p50"].sum(), rel=1e-4)
    # summed band is narrower than adding up every store's band (errors partly cancel)
    assert total["p90"] - total["p50"] < day1["p90"].sum() - day1["p50"].sum()
    assert body["inventory"] is None


def test_forecast_errors(client):
    url = "/api/datasets/demo/forecast"
    assert client.get(url, params={"location_id": "WH-SHM", "product_id": "automotive"}).status_code == 400
    assert client.get(url, params={"location_id": "ST-01", "product_id": "nope"}).status_code == 404
    assert client.get(url, params={"location_id": "ST-99", "product_id": "automotive"}).status_code == 404


def test_recommendations(client, data_dir):
    recs = pd.read_csv(data_dir / "results" / "recommendations.csv")
    body = client.get("/api/datasets/demo/recommendations", params={"limit": 1000}).json()
    assert body["total"] == len(recs)
    assert body["total_value_inr"] == pytest.approx(recs["value_inr"].sum(), abs=1)
    order = {"high": 0, "medium": 1, "low": 2}
    ranks = [order[r["priority"]] for r in body["items"]]
    assert ranks == sorted(ranks)
    first = body["items"][0]
    assert first["product_name"] and first["to_name"] and first["reason"]

    reorders = client.get("/api/datasets/demo/recommendations", params={"type": "reorder"}).json()
    assert reorders["total"] == (recs["type"] == "reorder").sum()
    if (recs["from_location"] == "SUPPLIER").any():
        sup = next(r for r in client.get("/api/datasets/demo/recommendations",
                                         params={"type": "reorder", "limit": 1000}).json()["items"]
                   if r["from_location"] == "SUPPLIER")
        assert sup["from_name"] == "Supplier"

    loc = first["to_location"]
    mine = client.get("/api/datasets/demo/recommendations", params={"location_id": loc, "limit": 1000}).json()
    assert mine["total"] == ((recs["from_location"] == loc) | (recs["to_location"] == loc)).sum()

    one = client.get(f"/api/datasets/demo/recommendations/{first['rec_id']}").json()
    assert one == first
    assert client.get("/api/datasets/demo/recommendations/R9999").status_code == 404


def test_metrics_and_graph(client, data_dir):
    m = client.get("/api/datasets/demo/metrics").json()
    assert m["forecast_model"] == "ensemble"
    models = [r["model"] for r in m["overall"]]
    assert {"baseline", "naive", "stgnn", "ensemble"} <= set(models)
    wapes = [r["wape"] for r in m["overall"]]
    assert wapes == sorted(wapes)
    assert len(m["by_store"]) == 54 and "baseline" in m["by_store"][0]["wape"]
    assert len(m["rolling"]) == 8
    assert m["training"]["best_epoch"] == 3 and m["training"]["recommendation_model"] == "ensemble"

    g = client.get("/api/datasets/demo/graph").json()
    assert len(g["nodes"]) == 57
    assert len(g["edges"]) == len(pd.read_csv(data_dir / "graph" / "edges.csv"))
    ids = {n["location_id"] for n in g["nodes"]}
    assert all(e["source_id"] in ids and e["target_id"] in ids for e in g["edges"])


def test_cors_allows_the_frontend(client):
    r = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
