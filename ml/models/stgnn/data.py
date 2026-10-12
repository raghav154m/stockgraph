"""Turn a dataset and its store graph into arrays for the ST-GNN (numpy only, no PyTorch).

Shapes (T days, N stores, F product families):
  sales     [T, N, F]   units sold
  mask      [N, F]      1 where the store sells that product, 0 where the series doesn't exist
  scale     [N, F]      per-series average daily sales over the training period
  calendar  [T, C]      day-of-week one-hot (7), is_holiday, is_closed
  adjacency [N, N]      normalised: D^-1/2 (A + I) D^-1/2

The model sees sales divided by `scale`, so a big grocery series and a small books series
are on the same footing. Normalising the adjacency stops a highly connected store from
swamping its neighbours' signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.load import Dataset

LOOKBACK = 56
HORIZON = 28
CAL_FEATURES = 9


@dataclass
class STData:
    sales: np.ndarray
    mask: np.ndarray
    calendar: np.ndarray
    adjacency: np.ndarray
    dates: pd.DatetimeIndex
    stores: list[str]
    products: list[str]

    def scale(self, end: int, days: int = 365) -> np.ndarray:
        """Average daily sales per series over the `days` before index `end` (inclusive); 1 where unused."""
        start = max(0, end - days + 1)
        window = self.sales[start:end + 1]
        open_days = self.calendar[start:end + 1, 8] == 0
        avg = window[open_days].mean(axis=0) if open_days.any() else window.mean(axis=0)
        return np.where((avg > 0) & (self.mask > 0), avg, 1.0).astype(np.float32)

    def index_of(self, date: pd.Timestamp) -> int:
        return int(self.dates.get_loc(date))


def normalised_adjacency(n: int, edges: pd.DataFrame) -> np.ndarray:
    a = np.zeros((n, n), dtype=np.float64)
    for e in edges.itertuples():
        a[e.source, e.target] = a[e.target, e.source] = max(a[e.source, e.target], float(e.weight))
    a += np.eye(n)
    d = 1.0 / np.sqrt(a.sum(axis=1))
    return (a * d[:, None] * d[None, :]).astype(np.float32)


def calendar_features(dates: pd.DatetimeIndex, holidays: pd.Series, closed: pd.Series) -> np.ndarray:
    cal = np.zeros((len(dates), CAL_FEATURES), dtype=np.float32)
    cal[np.arange(len(dates)), dates.dayofweek] = 1.0
    cal[:, 7] = holidays.reindex(dates, fill_value=0).to_numpy(dtype=np.float32)
    cal[:, 8] = closed.reindex(dates, fill_value=0).to_numpy(dtype=np.float32)
    return cal


def build(ds: Dataset, graph_dir: str | Path, extra_days: int = HORIZON) -> STData:
    """Arrays for every day in the data plus `extra_days` future days (calendar only, sales 0)."""
    graph_dir = Path(graph_dir)
    nodes = pd.read_csv(graph_dir / "nodes.csv").sort_values("node_index")
    edges = pd.read_csv(graph_dir / "edges.csv")
    stores = nodes["location_id"].tolist()
    products = sorted(ds.sales["product_id"].unique())

    observed = pd.date_range(ds.sales["date"].min(), ds.sales["date"].max(), freq="D")
    dates = pd.date_range(observed[0], periods=len(observed) + extra_days, freq="D")

    cube = (ds.sales.pivot_table(index="date", columns=["location_id", "product_id"], values="units_sold", aggfunc="sum")
            .reindex(index=observed)
            .reindex(columns=pd.MultiIndex.from_product([stores, products]))
            .fillna(0.0))
    sales = np.zeros((len(dates), len(stores), len(products)), dtype=np.float32)
    sales[: len(observed)] = cube.to_numpy(dtype=np.float32).reshape(len(observed), len(stores), len(products))

    present = ds.sales[["location_id", "product_id"]].drop_duplicates()
    mask = np.zeros((len(stores), len(products)), dtype=np.float32)
    s_idx = {s: i for i, s in enumerate(stores)}
    p_idx = {p: i for i, p in enumerate(products)}
    for r in present.itertuples():
        if r.location_id in s_idx:
            mask[s_idx[r.location_id], p_idx[r.product_id]] = 1.0

    holidays = ds.calendar.set_index("date")["is_holiday"]
    closed = ds.sales.groupby("date")["is_closed"].max()
    calendar = calendar_features(dates, holidays, closed)
    adjacency = normalised_adjacency(len(stores), edges)
    return STData(sales, mask, calendar, adjacency, dates, stores, products)


def windows(data: STData, origins: np.ndarray, scale: np.ndarray) -> dict[str, np.ndarray]:
    """Model inputs for forecast origins (index of the last known day).

    x_sales   [B, LOOKBACK, N, F]   past sales / scale
    x_cal     [B, LOOKBACK, C]      past calendar
    f_cal     [B, HORIZON, C]       calendar of the days being forecast (known in advance)
    y         [B, HORIZON, N, F]    future sales / scale (zeros beyond the data)
    """
    past = origins[:, None] + np.arange(-LOOKBACK + 1, 1)[None, :]
    future = origins[:, None] + np.arange(1, HORIZON + 1)[None, :]
    if past.min() < 0 or future.max() >= len(data.dates):
        raise ValueError("Origin too close to the start or end of the data")
    s = scale[None, None]
    return {
        "x_sales": data.sales[past] / s,
        "x_cal": data.calendar[past],
        "f_cal": data.calendar[future],
        "y": data.sales[future] / s,
    }


def split_origins(last_known: int, val_windows: int = 8, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Training and validation forecast origins whose targets end on or before `last_known`.

    Validation = the most recent `val_windows` weekly origins; training = every earlier origin
    whose targets don't overlap the validation targets.
    """
    latest = last_known - HORIZON                         # last origin with a full 28-day target
    val = latest - 7 * np.arange(val_windows)[::-1]
    train_end = val.min() - HORIZON                       # training targets stop before validation starts
    train = np.arange(LOOKBACK - 1, train_end + 1, stride)
    if len(train) < 50:
        raise ValueError(f"Only {len(train)} training windows; need more history")
    return train, val
