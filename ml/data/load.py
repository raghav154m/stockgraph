"""Load a dataset in the four-file template into memory, ready for modelling.

The same loader serves the demo dataset and user uploads.

Missing days are filled in so every series has one row per day (models need evenly spaced
time steps). A date missing from every series at once (e.g. Christmas, when Favorita stores
are closed) is marked `is_closed = 1` with zero sales; any other gap is a day with no sales.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED = {
    "locations.csv": ["location_id", "name", "type", "city", "region", "latitude", "longitude", "supplied_by"],
    "products.csv": ["product_id", "name", "category", "unit_cost", "lead_time_days"],
    "sales.csv": ["date", "location_id", "product_id", "units_sold"],
    "inventory.csv": ["location_id", "product_id", "on_hand"],
}


@dataclass
class Dataset:
    locations: pd.DataFrame
    products: pd.DataFrame
    sales: pd.DataFrame       # complete daily panel: date, location_id, product_id, units_sold, on_promotion, is_closed
    inventory: pd.DataFrame   # location_id, product_id, on_hand, on_order
    calendar: pd.DataFrame    # date, is_holiday
    as_of: pd.Timestamp       # last date with sales; the "today" of the dataset

    @property
    def stores(self) -> pd.DataFrame:
        return self.locations[self.locations["type"] == "store"]

    @property
    def warehouses(self) -> pd.DataFrame:
        return self.locations[self.locations["type"] == "warehouse"]


def _read(path: Path, name: str) -> pd.DataFrame:
    if not (path / name).exists():
        raise FileNotFoundError(f"{name} not found in {path}")
    df = pd.read_csv(path / name)
    missing = [c for c in REQUIRED.get(name, []) if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    return df


def complete_panel(sales: pd.DataFrame) -> pd.DataFrame:
    """One row per series per day from the first to the last date in the data."""
    sales = sales.copy()
    sales["date"] = pd.to_datetime(sales["date"])
    if "on_promotion" not in sales.columns:
        sales["on_promotion"] = 0

    dates = pd.date_range(sales["date"].min(), sales["date"].max(), freq="D")
    observed_dates = set(sales["date"].unique())
    series = sales[["location_id", "product_id"]].drop_duplicates()

    grid = series.merge(pd.DataFrame({"date": dates}), how="cross")
    panel = grid.merge(sales, on=["location_id", "product_id", "date"], how="left")
    panel["units_sold"] = panel["units_sold"].fillna(0.0)
    panel["on_promotion"] = panel["on_promotion"].fillna(0).astype(int)
    panel["is_closed"] = (~panel["date"].isin(observed_dates)).astype(int)
    return panel.sort_values(["location_id", "product_id", "date"]).reset_index(drop=True)


def load_dataset(path: str | Path) -> Dataset:
    path = Path(path)
    locations = _read(path, "locations.csv")
    locations["supplied_by"] = locations["supplied_by"].fillna("")
    products = _read(path, "products.csv")
    inventory = _read(path, "inventory.csv")
    if "on_order" not in inventory.columns:
        inventory["on_order"] = 0.0
    inventory["on_order"] = inventory["on_order"].fillna(0.0)

    sales = complete_panel(_read(path, "sales.csv"))

    if (path / "calendar.csv").exists():
        calendar = pd.read_csv(path / "calendar.csv", parse_dates=["date"])[["date", "is_holiday"]]
    else:
        calendar = pd.DataFrame({"date": pd.date_range(sales["date"].min(), sales["date"].max()), "is_holiday": 0})

    return Dataset(locations, products, sales, inventory, calendar, as_of=sales["date"].max())
