"""Blend the baseline and ST-GNN forecasts, score the blend, and pick the model for recommendations.

The blend is a fixed 50/50 average of the two models' daily forecasts. The weight is not tuned
on the holdout (that would flatter the score); 50/50 is the standard choice when two models are
of similar accuracy but make different kinds of errors.

Reads (from results/): backtest.csv (with a `stgnn` column), metrics.csv, forecasts.csv,
forecasts_stgnn.csv. Writes: an `ensemble` column and rows, forecasts_ensemble.csv, and, when a
model other than the baseline is chosen, recomputed recommendations and summary.json.

Usage (no retraining, takes seconds):
  python -m ml.models.ensemble
  python -m ml.models.ensemble --model-for-recommendations stgnn
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.load import load_dataset
from ml.models.baseline import KEYS, Z80, error_metrics, series_sigma

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "processed"
MODELS = ("baseline", "stgnn", "ensemble")


def _forecast_file(model: str) -> str:
    return "forecasts.csv" if model == "baseline" else f"forecasts_{model}.csv"


def blend(data_dir: Path = DEFAULT_DATA, model_for_recommendations: str = "auto", weight: float = 0.5,
          verbose: bool = True) -> dict:
    out = data_dir / "results"
    for name in ("backtest.csv", "metrics.csv", "forecasts.csv", "forecasts_stgnn.csv"):
        if not (out / name).exists():
            raise FileNotFoundError(f"{out / name} not found. Run `python -m ml.pipeline` and "
                                    "`python -m ml.models.stgnn.train` first.")

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    ds = load_dataset(data_dir)
    closed = set(ds.sales.loc[ds.sales["is_closed"] == 1, "date"].unique())

    # ---- holdout score ----
    bt = pd.read_csv(out / "backtest.csv", parse_dates=["date"])
    if "stgnn" not in bt.columns or bt["stgnn"].isna().all():
        raise ValueError("backtest.csv has no ST-GNN predictions. Run `python -m ml.models.stgnn.train` first.")
    bt["ensemble"] = weight * bt["stgnn"] + (1 - weight) * bt["baseline"]
    scored = bt[~bt["date"].isin(closed)].dropna(subset=["stgnn"])

    metrics = pd.read_csv(out / "metrics.csv")
    metrics = metrics[metrics["model"] != "ensemble"]
    rows = [{"model": "ensemble", "location_id": "ALL", **error_metrics(scored["actual"], scored["ensemble"])}]
    rows += [{"model": "ensemble", "location_id": loc, **error_metrics(g["actual"], g["ensemble"])}
             for loc, g in scored.groupby("location_id")]
    metrics = pd.concat([metrics, pd.DataFrame(rows)], ignore_index=True)
    wape = metrics[metrics["location_id"] == "ALL"].set_index("model")["wape"].astype(float)

    # ---- blended forecast for the next 28 days ----
    fb = pd.read_csv(out / "forecasts.csv", parse_dates=["date"])
    fs = pd.read_csv(out / "forecasts_stgnn.csv", parse_dates=["date"])
    fc = fb[KEYS + ["date", "p50"]].merge(fs[KEYS + ["date", "p50"]], on=KEYS + ["date"], suffixes=("_b", "_s"))
    fc["p50"] = weight * fc["p50_s"] + (1 - weight) * fc["p50_b"]
    fc = fc.drop(columns=["p50_b", "p50_s"]).merge(series_sigma(scored, "ensemble"), on=KEYS, how="left")
    fc["sigma"] = fc["sigma"].fillna(fc["p50"] * 0.5)
    fc["p10"] = np.clip(fc["p50"] - Z80 * fc["sigma"], 0, None)
    fc["p90"] = fc["p50"] + Z80 * fc["sigma"]
    fc["model"] = "ensemble"
    fc = fc[KEYS + ["date", "model", "p50", "p10", "p90", "sigma"]].round({"p50": 3, "p10": 3, "p90": 3, "sigma": 3})

    fc.assign(date=fc["date"].dt.strftime("%Y-%m-%d")).to_csv(out / "forecasts_ensemble.csv", index=False)
    bt.assign(date=bt["date"].dt.strftime("%Y-%m-%d")).round(3).to_csv(out / "backtest.csv", index=False)
    metrics.to_csv(out / "metrics.csv", index=False)

    # ---- which model drives recommendations ----
    best = min(MODELS, key=lambda m: wape[m])
    chosen = best if model_for_recommendations == "auto" else model_for_recommendations
    log("Holdout WAPE  " + "   ".join(f"{m} {wape[m]:.1%}" for m in (*MODELS, "naive")))
    log(f"Recommendations use: {chosen}" + (" (lowest holdout error)" if model_for_recommendations == "auto" else ""))

    from ml.pipeline import optimize  # local import avoids a circular import at module load
    summary = optimize(ds, pd.read_csv(out / _forecast_file(chosen), parse_dates=["date"]), out, metrics, chosen)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    result = {
        "holdout_wape": {m: round(float(w), 4) for m, w in wape.items()},
        "ensemble_weight_stgnn": weight,
        "best_model": best,
        "recommendation_model": chosen,
        "note": "Models are compared on one 28-day holdout; the blend weight is fixed, not tuned on it.",
    }
    training = out / "stgnn_training.json"
    if training.exists():
        report = json.loads(training.read_text())
        report.update(result)
        training.write_text(json.dumps(report, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-for-recommendations", choices=["auto", *MODELS], default="auto")
    parser.add_argument("--weight", type=float, default=0.5, help="share of the ST-GNN in the blend")
    args = parser.parse_args()
    print(json.dumps(blend(args.data, args.model_for_recommendations, args.weight), indent=2))


if __name__ == "__main__":
    main()
