"""Load a dataset folder (template CSVs + results/ from the ML pipeline) into the database.

Reloading a dataset id replaces it completely, so this is safe to re-run after retraining.

Usage:
  python -m backend.loader                         # demo: data/processed -> data/stockgraph.db
  python -m backend.loader --data path/to/dataset --id my-upload --name "My shop"
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import Engine, delete, insert

from backend.config import Settings
from backend.db import init_db, make_engine
from backend.models import (DATA_TABLES, Backtest, Dataset, Forecast, GraphEdge, InventoryStatus, Location, Metric,
                            Product, Recommendation, RollingMetric, Sale)

HISTORY_DAYS = 182      # sales history kept for charts (the forecast is 28 days; this gives context)
CHUNK = 20_000          # rows per INSERT batch
FORECAST_FILES = {"baseline": "forecasts.csv", "stgnn": "forecasts_stgnn.csv", "ensemble": "forecasts_ensemble.csv"}


class ResultsMissing(FileNotFoundError):
    pass


def _records(df: pd.DataFrame, columns: list[str]) -> list[dict]:
    """DataFrame -> list of dicts for INSERT, with NaN/inf as NULL and numpy scalars as Python ones."""
    out = df[columns].replace([np.inf, -np.inf], np.nan).astype(object)
    out = out.where(out.notna(), None)
    return out.to_dict("records")


def _insert(conn, model, dataset_id: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    columns = [c.name for c in model.__table__.columns if c.name != "dataset_id"]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{model.__tablename__}: input is missing columns {missing}")
    rows = _records(df, columns)
    for row in rows:
        row["dataset_id"] = dataset_id
    for i in range(0, len(rows), CHUNK):
        conn.execute(insert(model), rows[i:i + CHUNK])
    return len(rows)


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def read_folder(data_dir: Path, history_days: int = HISTORY_DAYS) -> dict[str, pd.DataFrame | dict | None]:
    """Read everything the database needs from a dataset folder. Optional files come back empty/None."""
    results = data_dir / "results"
    for name in ("forecasts.csv", "backtest.csv", "metrics.csv", "inventory_status.csv",
                 "recommendations.csv", "summary.json"):
        if not (results / name).exists():
            raise ResultsMissing(f"{results / name} not found. Run `python -m ml.pipeline` first.")

    sales = pd.read_csv(data_dir / "sales.csv", dtype={"date": str})
    last = pd.Timestamp(sales["date"].max())
    first_kept = (last - pd.Timedelta(days=history_days - 1)).strftime("%Y-%m-%d")
    sales = sales[sales["date"] >= first_kept]
    if "on_promotion" not in sales.columns:
        sales = sales.assign(on_promotion=0)
    sales["on_promotion"] = sales["on_promotion"].fillna(0).astype(int)

    forecasts = []
    for model, name in FORECAST_FILES.items():
        if (results / name).exists():
            fc = pd.read_csv(results / name, dtype={"date": str})
            forecasts.append(fc.assign(model=model))   # file name decides the model label

    backtest = pd.read_csv(results / "backtest.csv", dtype={"date": str})
    for col in ("baseline", "naive", "stgnn", "ensemble"):
        if col not in backtest.columns:
            backtest[col] = np.nan

    rolling_path = results / "metrics_rolling.csv"
    edges_path = data_dir / "graph" / "edges.csv"
    return {
        "locations": pd.read_csv(data_dir / "locations.csv"),
        "products": pd.read_csv(data_dir / "products.csv"),
        "sales": sales,
        "forecasts": pd.concat(forecasts, ignore_index=True),
        "backtest": backtest,
        "metrics": pd.read_csv(results / "metrics.csv"),
        "rolling": pd.read_csv(rolling_path, dtype={"holdout_start": str}) if rolling_path.exists() else pd.DataFrame(),
        "status": pd.read_csv(results / "inventory_status.csv"),
        "recommendations": pd.read_csv(results / "recommendations.csv", dtype={"deadline": str}),
        "edges": pd.read_csv(edges_path) if edges_path.exists() else pd.DataFrame(),
        "summary": _read_json(results / "summary.json"),
        "meta": _read_json(data_dir / "meta.json") or {},
        "training": _read_json(results / "stgnn_training.json"),
    }


def load_dataset_into_db(engine: Engine, data_dir: Path, dataset_id: str = "demo", name: str | None = None,
                         history_days: int = HISTORY_DAYS) -> dict:
    """Replace `dataset_id` in the database with the contents of `data_dir`. Returns row counts."""
    t0 = time.time()
    d = read_folder(Path(data_dir), history_days)
    summary, meta = d["summary"], d["meta"]
    edges = d["edges"]

    counts: dict[str, int] = {}
    with engine.begin() as conn:            # one transaction: a failed load leaves the old data in place
        for model in DATA_TABLES:
            conn.execute(delete(model).where(model.dataset_id == dataset_id))
        conn.execute(delete(Dataset).where(Dataset.id == dataset_id))

        conn.execute(insert(Dataset), [{
            "id": dataset_id,
            "name": name or ("Demo: Telangana grocery chain" if dataset_id == "demo" else dataset_id),
            "source": meta.get("source", ""),
            "as_of": summary["as_of"],
            "forecast_model": summary.get("forecast_model", "baseline"),
            "loaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "summary_json": json.dumps(summary),
            "meta_json": json.dumps(meta),
            "training_json": json.dumps(d["training"]) if d["training"] else None,
        }])
        counts["locations"] = _insert(conn, Location, dataset_id, d["locations"])
        counts["products"] = _insert(conn, Product, dataset_id, d["products"])
        counts["sales"] = _insert(conn, Sale, dataset_id, d["sales"])
        counts["forecasts"] = _insert(conn, Forecast, dataset_id, d["forecasts"])
        counts["backtest"] = _insert(conn, Backtest, dataset_id, d["backtest"])
        counts["metrics"] = _insert(conn, Metric, dataset_id, d["metrics"])
        counts["metrics_rolling"] = _insert(conn, RollingMetric, dataset_id, d["rolling"])
        counts["inventory_status"] = _insert(conn, InventoryStatus, dataset_id, d["status"])
        counts["recommendations"] = _insert(conn, Recommendation, dataset_id, d["recommendations"])
        counts["graph_edges"] = _insert(conn, GraphEdge, dataset_id, edges)

    counts["seconds"] = round(time.time() - t0, 1)
    return counts


def main() -> None:
    settings = Settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=settings.data_dir)
    parser.add_argument("--db", default=settings.db_url, help="SQLAlchemy URL")
    parser.add_argument("--id", default="demo")
    parser.add_argument("--name", default=None)
    parser.add_argument("--history-days", type=int, default=HISTORY_DAYS)
    args = parser.parse_args()

    engine = make_engine(args.db)
    init_db(engine)
    counts = load_dataset_into_db(engine, args.data, args.id, args.name, args.history_days)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
