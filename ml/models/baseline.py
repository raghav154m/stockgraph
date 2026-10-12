"""Baseline demand forecasts: a gradient-boosted model plus a seasonal-naive benchmark.

Gradient-boosted model ("baseline")
  One global model for every store x product series, forecasting each of the next 28 days
  directly. Every training row is a (series, target day, days-ahead k) example: its features
  only use sales up to k days before the target, exactly what is known when forecasting k
  days out. So day 1 of the forecast uses yesterday's sales, and day 28 uses sales from four
  weeks earlier, from one model. Poisson loss suits non-negative sales of very different sizes.

Seasonal naive ("naive")
  Each future day = the average of the same weekday over the last 4 weeks. Any real model
  has to beat this to be worth using.

Uncertainty: the spread of each series' errors on a 28-day holdout gives a daily sigma, used
for the p10/p90 band here and for safety stock in the inventory optimizer.

The sales panel must be complete (one row per series per day), which `load_dataset` ensures,
so the data can be held as a series x day matrix and every feature is a fast array lookup.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ml.data.load import Dataset

HORIZON = 28
Z80 = NormalDist().inv_cdf(0.9)  # p10 and p90 sit 1.28 sigma either side of the median
KEYS = ["location_id", "product_id"]
CATEGORICAL = ["location_code", "product_code"]


class Panel:
    """Sales as a (series x day) matrix plus per-day calendar arrays."""

    def __init__(self, sales: pd.DataFrame, calendar: pd.DataFrame, extra_days: int = 0):
        wide = sales.pivot_table(index=KEYS, columns="date", values="units_sold", aggfunc="sum")
        self.series = wide.index.to_frame(index=False)
        self.dates = pd.date_range(wide.columns.min(), periods=wide.shape[1] + extra_days, freq="D")
        self.y = np.full((len(self.series), len(self.dates)), np.nan)
        self.y[:, : wide.shape[1]] = wide.to_numpy(dtype=float)

        closed = sales.groupby("date")["is_closed"].max().reindex(self.dates, fill_value=0)
        holiday = calendar.set_index("date")["is_holiday"].reindex(self.dates, fill_value=0)
        self.closed = closed.to_numpy(dtype=int)
        self.holiday = holiday.fillna(0).to_numpy(dtype=int)
        self.dow = self.dates.dayofweek.to_numpy()
        self.dom = self.dates.day.to_numpy()
        self.month = self.dates.month.to_numpy()

        filled = np.nan_to_num(self.y)
        zero = np.zeros((len(self.series), 1))
        self.csum = np.hstack([zero, np.cumsum(filled, axis=1)])        # csum[:, t] = sum of y[:, :t]
        self.csq = np.hstack([zero, np.cumsum(filled ** 2, axis=1)])
        self.location_code = self.series["location_id"].astype("category").cat.codes.to_numpy()
        self.product_code = self.series["product_id"].astype("category").cat.codes.to_numpy()

    def features(self, s: np.ndarray, t: np.ndarray, k: np.ndarray) -> pd.DataFrame:
        """Features for target day index t of series s, forecast k days ahead (origin = t - k)."""
        o = t - k  # last day whose sales are known

        def window_mean(w: int) -> np.ndarray:
            ok = o - w + 1 >= 0
            total = self.csum[s, np.clip(o + 1, 0, None)] - self.csum[s, np.clip(o - w + 1, 0, None)]
            return np.where(ok, total / w, np.nan)

        def lag(d: np.ndarray) -> np.ndarray:
            idx = t - d
            return np.where(idx >= 0, self.y[s, np.clip(idx, 0, None)], np.nan)

        m0 = np.ceil(k / 7).astype(int)  # most recent same weekday that is already known
        mean28 = window_mean(28)
        sq28 = self.csq[s, np.clip(o + 1, 0, None)] - self.csq[s, np.clip(o - 27, 0, None)]
        var28 = np.where(o - 27 >= 0, sq28 / 28 - mean28 ** 2, np.nan)

        feats = {
            "days_ahead": k,
            "last_value": lag(k),
            "same_weekday_1": lag(7 * m0),
            "same_weekday_2": lag(7 * (m0 + 1)),
            "same_weekday_3": lag(7 * (m0 + 2)),
            "same_weekday_4": lag(7 * (m0 + 3)),
            "mean_7": window_mean(7),
            "mean_28": mean28,
            "mean_91": window_mean(91),
            "std_28": np.sqrt(np.clip(var28, 0, None)),
            "lag_364": lag(np.full_like(t, 364)),
            "day_of_week": self.dow[t],
            "day_of_month": self.dom[t],
            "month": self.month[t],
            "is_holiday": self.holiday[t],
            "is_closed": self.closed[t],
            "location_code": self.location_code[s],
            "product_code": self.product_code[s],
        }
        return pd.DataFrame(feats)


def _model(seed: int, categorical: list[str]) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.08,
        max_iter=300,
        max_leaf_nodes=63,
        min_samples_leaf=50,
        l2_regularization=1.0,
        categorical_features=categorical or None,
        early_stopping=False,
        random_state=seed,
    )


def _fit(panel: Panel, last: int, train_days: int, draws: int, seed: int):
    """Train on (series, day) rows with target day in (last - train_days, last]. Returns (model, features used)."""
    rng = np.random.default_rng(seed)
    first = max(last - train_days + 1, 28)
    s_idx, t_idx = np.meshgrid(np.arange(len(panel.series)), np.arange(first, last + 1), indexing="ij")
    s_idx, t_idx = np.tile(s_idx.ravel(), draws), np.tile(t_idx.ravel(), draws)
    k_idx = rng.integers(1, HORIZON + 1, size=len(s_idx))
    keep = panel.closed[t_idx] == 0  # closed days teach nothing about demand
    train_x = panel.features(s_idx[keep], t_idx[keep], k_idx[keep])
    train_y = panel.y[s_idx[keep], t_idx[keep]]

    used = [c for c in train_x.columns if train_x[c].nunique(dropna=True) > 1]
    model = _model(seed, [c for c in CATEGORICAL if c in used]).fit(train_x[used], train_y)
    return model, used


def _predict(panel: Panel, model, used: list[str], origins: np.ndarray) -> np.ndarray:
    """Forecasts for each origin (index of the last known day): [len(origins), n_series, HORIZON]."""
    n = len(panel.series)
    k = np.arange(1, HORIZON + 1)
    o_f = np.repeat(origins, n * HORIZON)
    s_f = np.tile(np.repeat(np.arange(n), HORIZON), len(origins))
    k_f = np.tile(k, n * len(origins))
    pred = model.predict(panel.features(s_f, o_f + k_f, k_f)[used])
    return np.clip(pred, 0, None).reshape(len(origins), n, HORIZON)


def fit_predict(sales: pd.DataFrame, calendar: pd.DataFrame, cutoff: pd.Timestamp,
                train_days: int = 400, draws: int = 2, seed: int = 42) -> pd.DataFrame:
    """Train on data up to `cutoff`, forecast the next 28 days. Returns KEYS, date, p50.

    `draws` = how many random days-ahead values each training (series, day) gets.
    """
    history = sales[sales["date"] <= cutoff]
    panel = Panel(history, calendar, extra_days=HORIZON)
    last = len(panel.dates) - HORIZON - 1  # cutoff index
    model, used = _fit(panel, last, train_days, draws, seed)
    pred = _predict(panel, model, used, np.array([last]))[0]          # [n_series, HORIZON]

    n = len(panel.series)
    out = panel.series.iloc[np.repeat(np.arange(n), HORIZON)].reset_index(drop=True)
    out["date"] = np.tile(panel.dates[last + 1: last + 1 + HORIZON], n)
    out["p50"] = pred.ravel()
    return out


def rolling_forecasts(sales: pd.DataFrame, calendar: pd.DataFrame, boundaries: list[int],
                      train_days: int = 400, draws: int = 2, seed: int = 42,
                      log=lambda msg: None) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DatetimeIndex]:
    """Out-of-sample baseline forecasts for every origin from boundaries[0] to boundaries[-1].

    The model is refit at each boundary (day index, 0 = first sales date) and forecasts every
    origin up to the next boundary. Each forecast only uses a model trained on targets on or
    before its own origin, so nothing it predicts was seen in training.

    Returns (origins, forecasts [n_origins, n_series, HORIZON], series keys, panel dates).
    """
    panel = Panel(sales, calendar, extra_days=HORIZON)
    bounds = sorted(set(boundaries))
    origins, preds = [], []
    for i, b in enumerate(bounds):
        nxt = bounds[i + 1] if i + 1 < len(bounds) else b + 1
        block = np.arange(b, nxt)
        log(f"  baseline refit {i + 1}/{len(bounds)} at {panel.dates[b]:%Y-%m-%d}, forecasting {len(block)} origin(s)")
        model, used = _fit(panel, b, train_days, draws, seed)
        origins.append(block)
        preds.append(_predict(panel, model, used, block))
    return np.concatenate(origins), np.concatenate(preds), panel.series, panel.dates


def seasonal_naive(sales: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Average of the same weekday over the 4 weeks before `cutoff`."""
    recent = sales[(sales["date"] > cutoff - pd.Timedelta(days=28)) & (sales["date"] <= cutoff)
                   & (sales["is_closed"] == 0)].copy()
    recent["day_of_week"] = recent["date"].dt.dayofweek
    profile = recent.groupby(KEYS + ["day_of_week"])["units_sold"].mean().rename("p50").reset_index()

    series = sales[KEYS].drop_duplicates()
    dates = pd.DataFrame({"date": pd.date_range(cutoff + pd.Timedelta(days=1), periods=HORIZON, freq="D")})
    future = series.merge(dates, how="cross")
    future["day_of_week"] = future["date"].dt.dayofweek
    out = future.merge(profile, on=KEYS + ["day_of_week"], how="left")
    out["p50"] = out["p50"].fillna(0.0)
    return out[KEYS + ["date", "p50"]]


def error_metrics(actual: pd.Series, pred: pd.Series) -> dict:
    err = pred - actual
    total = actual.sum()
    return {
        "mae": round(float(err.abs().mean()), 3),
        "wape": round(float(err.abs().sum() / total), 4) if total > 0 else np.nan,
        "bias": round(float(err.sum() / total), 4) if total > 0 else np.nan,
    }


def backtest(ds: Dataset, train_days: int = 400, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the last 28 days, forecast them, and score both models.

    Returns (predictions vs actuals, metrics by model overall and per location).
    """
    cutoff = ds.as_of - pd.Timedelta(days=HORIZON)
    actual = ds.sales[ds.sales["date"] > cutoff][KEYS + ["date", "units_sold", "is_closed"]]

    gb = fit_predict(ds.sales, ds.calendar, cutoff, train_days, seed=seed).rename(columns={"p50": "baseline"})
    naive = seasonal_naive(ds.sales, cutoff).rename(columns={"p50": "naive"})
    preds = actual.merge(gb, on=KEYS + ["date"]).merge(naive, on=KEYS + ["date"])
    preds = preds.rename(columns={"units_sold": "actual"})

    scored = preds[preds["is_closed"] == 0]
    rows = []
    for model in ("baseline", "naive"):
        rows.append({"model": model, "location_id": "ALL", **error_metrics(scored["actual"], scored[model])})
        for loc, g in scored.groupby("location_id"):
            rows.append({"model": model, "location_id": loc, **error_metrics(g["actual"], g[model])})
    return preds.drop(columns="is_closed"), pd.DataFrame(rows)


def series_sigma(preds: pd.DataFrame, model: str = "baseline") -> pd.DataFrame:
    """Daily forecast error (RMSE) per series from the holdout, with a floor.

    A series with a perfect holdout would otherwise get zero safety stock; the floor
    (half the product's typical relative error, applied to the series' own level) prevents that.
    """
    df = preds.copy()
    df["sq_err"] = (df[model] - df["actual"]) ** 2
    per = df.groupby(KEYS).agg(mse=("sq_err", "mean"), level=("actual", "mean")).reset_index()
    per["rmse"] = np.sqrt(per["mse"])
    rel = per["rmse"] / per["level"].replace(0, np.nan)
    per["product_rel"] = rel.groupby(per["product_id"]).transform("median").fillna(0.5)
    per["sigma"] = np.maximum(per["rmse"], 0.5 * per["product_rel"] * per["level"])
    return per[KEYS + ["sigma"]]


def forecast(ds: Dataset, sigma: pd.DataFrame, train_days: int = 400, seed: int = 42) -> pd.DataFrame:
    """Refit on all history and forecast the 28 days after `as_of` with p10/p90 bands."""
    out = fit_predict(ds.sales, ds.calendar, ds.as_of, train_days, seed=seed).merge(sigma, on=KEYS, how="left")
    out["sigma"] = out["sigma"].fillna(out["p50"] * 0.5)
    out["p10"] = np.clip(out["p50"] - Z80 * out["sigma"], 0, None)
    out["p90"] = out["p50"] + Z80 * out["sigma"]
    out["model"] = "baseline"
    for col in ("p50", "p10", "p90", "sigma"):
        out[col] = out[col].round(3)
    return out[KEYS + ["date", "model", "p50", "p10", "p90", "sigma"]]
