"""Run the full baseline pipeline on a dataset folder: forecasts -> stock status -> recommendations.

Used for the demo dataset now and for user uploads later (same template, same code).

Outputs, in <data>/results/:
  forecasts.csv          next 28 days per store x product: p50, p10, p90, sigma
  backtest.csv           last 28 days held out: actual vs baseline vs naive
  metrics.csv            MAE / WAPE / bias per model, overall and per store
  inventory_status.csv   policy numbers and status per store/warehouse x product
  recommendations.csv    transfers and reorders, prioritised, with plain-language reasons
  summary.json           headline numbers for the dashboard

Usage:
  python -m ml.pipeline                       # demo data in data/processed
  python -m ml.pipeline --service-level 0.98
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from ml.data.load import load_dataset
from ml.inventory.optimizer import Policy, recommend
from ml.models.baseline import backtest, forecast, series_sigma

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "processed"


def optimize(ds, fc: pd.DataFrame, out_dir: Path, metrics: pd.DataFrame, model: str,
             service_level: float = 0.95) -> dict:
    """Stock status + recommendations from `fc` (any model's forecasts); writes CSVs, returns summary."""
    status, recs = recommend(ds, fc, Policy(service_level=service_level))
    status.to_csv(out_dir / "inventory_status.csv", index=False)
    recs.to_csv(out_dir / "recommendations.csv", index=False)

    stores = status[status["supplied_by"] != "SUPPLIER"]
    unit_cost = ds.products.set_index("product_id")["unit_cost"]
    over = stores[stores["status"] == "overstock"]
    excess_value = ((over["on_hand"] - over["target_level"]).clip(lower=0) * over["product_id"].map(unit_cost)).sum()
    overall = metrics[metrics["location_id"] == "ALL"].set_index("model")["wape"]

    return {
        "as_of": ds.as_of.strftime("%Y-%m-%d"),
        "forecast_model": model,
        "service_level": service_level,
        "wape": {m: float(w) for m, w in overall.items()},
        "store_items": int(len(stores)),
        "status_counts": stores["status"].value_counts().to_dict(),
        "overstock_value_inr": round(float(excess_value), 0),
        "recommendations": recs["type"].value_counts().to_dict(),
        "high_priority": int((recs["priority"] == "high").sum()),
        "reorder_value_inr": round(float(recs.loc[recs["type"] == "reorder", "value_inr"].sum()), 0),
    }


def run(data_dir: Path = DEFAULT_DATA, out_dir: Path | None = None, service_level: float = 0.95,
        train_days: int = 400, seed: int = 42, verbose: bool = True) -> dict:
    out_dir = out_dir or data_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    t0 = time.time()
    ds = load_dataset(data_dir)
    log(f"Loaded {ds.sales[['location_id', 'product_id']].drop_duplicates().shape[0]} series, "
        f"{ds.sales['date'].nunique()} days (as of {ds.as_of:%Y-%m-%d})")

    log("Backtesting on the last 28 days ...")
    preds, metrics = backtest(ds, train_days, seed)
    overall = metrics[metrics["location_id"] == "ALL"].set_index("model")
    log(f"  WAPE  baseline {overall.at['baseline', 'wape']:.1%}   naive {overall.at['naive', 'wape']:.1%}")

    log("Forecasting the next 28 days ...")
    fc = forecast(ds, series_sigma(preds), train_days, seed)

    fc.assign(date=fc["date"].dt.strftime("%Y-%m-%d")).to_csv(out_dir / "forecasts.csv", index=False)
    preds.assign(date=preds["date"].dt.strftime("%Y-%m-%d")).round(3).to_csv(out_dir / "backtest.csv", index=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)

    log("Optimizing inventory ...")
    summary = optimize(ds, fc, out_dir, metrics, "baseline", service_level)
    summary["runtime_seconds"] = round(time.time() - t0, 1)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"Wrote results to {out_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--service-level", type=float, default=0.95)
    parser.add_argument("--train-days", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = run(args.data, args.out, args.service_level, args.train_days, args.seed)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
