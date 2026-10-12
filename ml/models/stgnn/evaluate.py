"""Rolling evaluation helpers for the ST-GNN (numpy/pandas only, no PyTorch).

  holdout_origins     the starting days of several back-to-back 28-day holdouts
  BaseForecasts       out-of-sample baseline forecasts for every origin, as [H, N, P] cubes
  naive_cube          seasonal-naive forecast from one origin, as a cube
  score               MAE / WAPE / bias over existing series on open days
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.load import Dataset
from ml.models.baseline import rolling_forecasts, seasonal_naive
from ml.models.stgnn.data import HORIZON, LOOKBACK, STData


def holdout_origins(last: int, count: int) -> list[int]:
    """Origins of `count` back-to-back, non-overlapping 28-day holdouts ending at `last`."""
    return [last - HORIZON * (j + 1) for j in reversed(range(count))]


def oof_start(last: int, history_days: int = 450) -> int:
    """First origin with an out-of-sample baseline forecast: about 15 months before the last day."""
    return max(LOOKBACK - 1, last - history_days)


def series_index(data: STData, keys: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    s_map = {s: i for i, s in enumerate(data.stores)}
    p_map = {p: i for i, p in enumerate(data.products)}
    return (keys["location_id"].map(s_map).to_numpy(), keys["product_id"].map(p_map).to_numpy())


class BaseForecasts:
    """Baseline forecasts for a range of origins: cube(origin) -> [HORIZON, N, P] in real units."""

    def __init__(self, origins: np.ndarray, cubes: np.ndarray):
        self.origins = origins
        self.cubes = cubes.astype(np.float32)
        self._row = {int(o): i for i, o in enumerate(origins)}

    def get(self, origins: np.ndarray) -> np.ndarray:
        missing = [int(o) for o in origins if int(o) not in self._row]
        if missing:
            raise KeyError(f"No baseline forecast for origins {missing[:5]}")
        return self.cubes[[self._row[int(o)] for o in origins]]

    @classmethod
    def build(cls, ds: Dataset, data: STData, boundaries: list[int], cache: Path | None = None,
              train_days: int = 400, seed: int = 42, log=lambda msg: None) -> "BaseForecasts":
        meta = {"boundaries": sorted(set(int(b) for b in boundaries)), "train_days": train_days, "seed": seed,
                "as_of": ds.as_of.strftime("%Y-%m-%d"), "series": int(data.mask.sum())}
        if cache is not None and cache.exists():
            saved = np.load(cache, allow_pickle=False)
            if json.loads(str(saved["meta"])) == meta:
                log("  using cached baseline forecasts")
                return cls(saved["origins"], saved["cubes"])

        origins, preds, keys, _ = rolling_forecasts(ds.sales, ds.calendar, meta["boundaries"],
                                                    train_days=train_days, seed=seed, log=log)
        si, pi = series_index(data, keys)
        ok = ~(pd.isna(si) | pd.isna(pi))
        cubes = np.zeros((len(origins), HORIZON, len(data.stores), len(data.products)), dtype=np.float32)
        # preds is [origin, series, day]; place each series at its (store, product) slot.
        cubes[:, :, si[ok].astype(int), pi[ok].astype(int)] = preds[:, ok, :].transpose(0, 2, 1)
        if cache is not None:
            np.savez_compressed(cache, origins=origins, cubes=cubes, meta=json.dumps(meta))
        return cls(origins, cubes)


def naive_cube(ds: Dataset, data: STData, origin: int) -> np.ndarray:
    df = seasonal_naive(ds.sales, data.dates[origin])
    si, pi = series_index(data, df)
    day = (df["date"] - data.dates[origin + 1]).dt.days.to_numpy()
    cube = np.zeros((HORIZON, len(data.stores), len(data.products)), dtype=np.float32)
    cube[day, si, pi] = df["p50"].to_numpy()
    return cube


def actual_cube(data: STData, origin: int) -> np.ndarray:
    return data.sales[origin + 1: origin + 1 + HORIZON]


def valid_mask(data: STData, origin: int) -> np.ndarray:
    """[HORIZON, N, P] True where the series exists and the day is not a store-closure day."""
    open_days = data.calendar[origin + 1: origin + 1 + HORIZON, 8] == 0
    return open_days[:, None, None] & (data.mask[None] > 0)


def score(pred: np.ndarray, actual: np.ndarray, valid: np.ndarray) -> dict:
    p, a = pred[valid].astype(np.float64), actual[valid].astype(np.float64)
    total = a.sum()
    return {
        "mae": round(float(np.abs(p - a).mean()), 3),
        "wape": round(float(np.abs(p - a).sum() / total), 4) if total > 0 else np.nan,
        "bias": round(float((p - a).sum() / total), 4) if total > 0 else np.nan,
    }
