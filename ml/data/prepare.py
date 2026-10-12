"""Turn raw Favorita files into StockGraph's four-file template, plus a calendar.

Steps:
  1. Keep the last N days of daily sales (default 730, about two years)
  2. Relabel stores to Telangana places and assign each store its nearest warehouse
  3. Write products (families) with simulated ₹ costs and lead times
  4. Simulate current inventory
  5. Build a calendar of national holidays and oil price for later model features

The output uses the same format as user uploads, so the demo and uploads share one pipeline.

Usage:
  python -m ml.data.prepare                 # defaults: data/raw -> data/processed, 730 days
  python -m ml.data.prepare --days 365
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ml.data.catalog import product_id, product_row
from ml.data.simulate import simulate_inventory
from ml.data.telangana import WAREHOUSES, assign_city_names, nearest_warehouse

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = ROOT / "data" / "raw"
DEFAULT_OUT = ROOT / "data" / "processed"


def load_raw(raw_dir: Path) -> dict[str, pd.DataFrame]:
    missing = [f for f in ("train.csv", "stores.csv") if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {raw_dir}. Run: python scripts/download_data.py"
        )
    raw = {
        "train": pd.read_csv(raw_dir / "train.csv", parse_dates=["date"]),
        "stores": pd.read_csv(raw_dir / "stores.csv"),
    }
    for name in ("holidays_events", "oil"):
        path = raw_dir / f"{name}.csv"
        raw[name] = pd.read_csv(path, parse_dates=["date"]) if path.exists() else None
    return raw


def build_locations(stores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (template locations incl. warehouses, mapping back to original store numbers)."""
    counts = stores.groupby("city").size().to_dict()
    city_map = assign_city_names(counts)

    rows, mapping = [], []
    for original_city, group in stores.sort_values("store_nbr").groupby("city", sort=False):
        city, sites = city_map[original_city]
        for (_, store), (site, lat, lon) in zip(group.iterrows(), sites):
            wh = nearest_warehouse(lat, lon)
            loc_id = f"ST-{int(store.store_nbr):02d}"
            rows.append({
                "location_id": loc_id,
                "name": city if site == city else f"{city} - {site}",
                "type": "store",
                "city": city,
                "region": wh["region"],
                "latitude": round(lat, 5),
                "longitude": round(lon, 5),
                "supplied_by": wh["location_id"],
            })
            mapping.append({
                "location_id": loc_id,
                "store_nbr": int(store.store_nbr),
                "original_city": original_city,
                "original_state": store.state,
                "store_type": store.type,
                "cluster": int(store.cluster),
            })

    for wh in WAREHOUSES:
        rows.append({**wh, "type": "warehouse", "supplied_by": ""})

    locations = pd.DataFrame(rows).sort_values(["type", "location_id"])  # stores first, then warehouses
    return locations.reset_index(drop=True), pd.DataFrame(mapping)


def build_sales(train: pd.DataFrame, days: int, store_map: pd.DataFrame) -> pd.DataFrame:
    end = train["date"].max()
    window = train[train["date"] > end - pd.Timedelta(days=days)].copy()

    loc_by_store = store_map.set_index("store_nbr")["location_id"]
    sales = pd.DataFrame({
        "date": window["date"],
        "location_id": window["store_nbr"].map(loc_by_store),
        "product_id": window["family"].map(product_id),
        "units_sold": window["sales"].round(2),
        "on_promotion": window["onpromotion"].fillna(0).astype(int),
    })

    # Drop store x product series with no sales in the whole window (e.g. a store that never sells books).
    totals = sales.groupby(["location_id", "product_id"])["units_sold"].transform("sum")
    sales = sales[totals > 0]
    return sales.sort_values(["location_id", "product_id", "date"]).reset_index(drop=True)


def build_calendar(raw: dict, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dates = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})

    hol = raw.get("holidays_events")
    if hol is not None:
        national = hol[
            (hol["locale"] == "National")
            & (hol["transferred"].astype(str).str.lower() != "true")
            & (hol["type"] != "Work Day")
        ]
        dates["is_holiday"] = dates["date"].isin(national["date"]).astype(int)
    else:
        dates["is_holiday"] = 0

    oil = raw.get("oil")
    if oil is not None:
        oil_series = oil.set_index("date")["dcoilwtico"]
        full = oil_series.reindex(pd.date_range(oil_series.index.min(), end, freq="D")).ffill().bfill()
        dates["oil_price"] = dates["date"].map(full).round(2)
    else:
        dates["oil_price"] = pd.NA

    dates["day_of_week"] = dates["date"].dt.dayofweek
    return dates


def prepare(raw_dir: Path = DEFAULT_RAW, out_dir: Path = DEFAULT_OUT, days: int = 730, seed: int = 42) -> dict:
    raw = load_raw(raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    locations, store_map = build_locations(raw["stores"])
    sales = build_sales(raw["train"], days, store_map)
    products = pd.DataFrame([product_row(f) for f in sorted(raw["train"]["family"].unique())])
    products = products[products["product_id"].isin(sales["product_id"].unique())]
    inventory = simulate_inventory(sales, locations, seed=seed)
    calendar = build_calendar(raw, sales["date"].min(), sales["date"].max())

    locations.to_csv(out_dir / "locations.csv", index=False)
    products.to_csv(out_dir / "products.csv", index=False)
    sales.assign(date=sales["date"].dt.strftime("%Y-%m-%d")).to_csv(out_dir / "sales.csv", index=False)
    inventory.to_csv(out_dir / "inventory.csv", index=False)
    calendar.assign(date=calendar["date"].dt.strftime("%Y-%m-%d")).to_csv(out_dir / "calendar.csv", index=False)
    store_map.to_csv(out_dir / "store_map.csv", index=False)

    meta = {
        "dataset": "demo",
        "source": "Favorita (Kaggle: store-sales-time-series-forecasting), relabelled to Telangana",
        "as_of": sales["date"].max().strftime("%Y-%m-%d"),
        "start": sales["date"].min().strftime("%Y-%m-%d"),
        "days": days,
        "stores": int((locations["type"] == "store").sum()),
        "warehouses": int((locations["type"] == "warehouse").sum()),
        "products": int(len(products)),
        "series": int(sales.groupby(["location_id", "product_id"]).ngroups),
        "sales_rows": int(len(sales)),
        "currency": "INR",
        "seed": seed,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    meta = prepare(args.raw, args.out, args.days, args.seed)
    print(f"Wrote demo dataset to {args.out}")
    for key, value in meta.items():
        print(f"  {key:<11} {value}")


if __name__ == "__main__":
    main()
