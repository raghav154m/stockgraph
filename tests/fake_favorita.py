"""Generate a small fake dataset with the same files and columns as Favorita.

Used by tests so the pipeline can be checked without downloading the real data.
The store/city layout mirrors Favorita: 54 stores in 22 cities, one city with 18 stores.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.catalog import FAMILIES

CITIES = (
    [("Quito", "Pichincha", 18), ("Guayaquil", "Guayas", 8), ("Cuenca", "Azuay", 3),
     ("Santo Domingo", "Santo Domingo", 3), ("Ambato", "Tungurahua", 2),
     ("Latacunga", "Cotopaxi", 2), ("Machala", "El Oro", 2)]
    + [(f"Town{i}", f"State{i}", 1) for i in range(16)]
)  # 23 cities, 54 stores


def write_fake_raw(raw_dir: Path, days: int = 150, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    raw_dir.mkdir(parents=True, exist_ok=True)

    stores, nbr = [], 1
    for city, state, n in CITIES:
        for _ in range(n):
            stores.append({"store_nbr": nbr, "city": city, "state": state,
                           "type": "ABCDE"[nbr % 5], "cluster": nbr % 17 + 1})
            nbr += 1
    stores = pd.DataFrame(stores)
    stores.to_csv(raw_dir / "stores.csv", index=False)

    dates = pd.date_range(end="2017-08-15", periods=days, freq="D")
    families = sorted(FAMILIES)
    weekly = 1 + 0.3 * np.sin(np.arange(days) * 2 * np.pi / 7)
    city_shock = {c: np.cumsum(rng.normal(0, 0.05, days)) for c, _, _ in CITIES}

    rows = []
    for _, s in stores.iterrows():
        store_level = rng.uniform(0.5, 2.0)
        shock = np.exp(city_shock[s.city] + rng.normal(0, 0.05, days))
        for f_i, fam in enumerate(families):
            if fam == "BOOKS" and s.store_nbr % 3:   # most stores never sell books
                sales = np.zeros(days)
            else:
                base = store_level * (5 + 3 * (f_i % 7))
                sales = np.maximum(0, base * weekly * shock + rng.normal(0, 1, days))
            promo = rng.binomial(3, 0.1, days)
            rows.append(pd.DataFrame({"date": dates, "store_nbr": s.store_nbr, "family": fam,
                                      "sales": sales.round(3), "onpromotion": promo}))
    train = pd.concat(rows, ignore_index=True)
    train.insert(0, "id", range(len(train)))
    train.to_csv(raw_dir / "train.csv", index=False)

    pd.DataFrame({
        "date": [dates[-20], dates[-15], dates[-14]],  # all inside the sliced window
        "type": ["Holiday", "Holiday", "Work Day"],
        "locale": ["National", "Local", "National"],
        "locale_name": ["Ecuador", "Quito", "Ecuador"],
        "description": ["Fake holiday", "Local fiesta", "Recovery day"],
        "transferred": [False, False, False],
    }).to_csv(raw_dir / "holidays_events.csv", index=False)

    oil_dates = dates[dates.dayofweek < 5]
    pd.DataFrame({"date": oil_dates, "dcoilwtico": 45 + rng.normal(0, 1, len(oil_dates)).cumsum()}) \
        .to_csv(raw_dir / "oil.csv", index=False)


def write_fake_stgnn_outputs(results: Path) -> None:
    """Write what `ml.models.stgnn.train` would: ST-GNN forecasts, backtest column, rolling metrics, report."""
    fc = pd.read_csv(results / "forecasts.csv")
    fc.assign(model="stgnn", p50=fc["p50"] * 1.05, p10=fc["p10"] * 1.05, p90=fc["p90"] * 1.05) \
        .to_csv(results / "forecasts_stgnn.csv", index=False)
    bt = pd.read_csv(results / "backtest.csv")
    bt.assign(stgnn=(bt["baseline"] * 1.02).round(3)).to_csv(results / "backtest.csv", index=False)
    metrics = pd.read_csv(results / "metrics.csv")
    base = metrics[metrics["model"] == "baseline"]
    pd.concat([metrics, base.assign(model="stgnn", wape=base["wape"] * 1.05)]) \
        .to_csv(results / "metrics.csv", index=False)
    rows = [{"holdout_start": h, "model": m, "mae": 2.0, "wape": w, "bias": 0.0}
            for h in ("2017-06-21", "ALL")
            for m, w in (("baseline", 0.139), ("stgnn", 0.148), ("ensemble", 0.140), ("naive", 0.162))]
    pd.DataFrame(rows).to_csv(results / "metrics_rolling.csv", index=False)
    (results / "stgnn_training.json").write_text(json.dumps({
        "settings": {"mode": "residual"}, "best_epoch": 3,
        "history": [{"epoch": 1, "loss": 0.5, "val_wape": 0.15}], "rolling_wape": {"stgnn": 0.148}}))
