"""Simulate the inventory layer that Favorita doesn't have.

Each store x product gets a stock level expressed as "days of cover" (how many days of
recent average demand the stock would last). The mix is deliberately uneven so the demo
dashboard has realistic problems to show:

  ~15% at risk      1-4 days of cover   -> should trigger reorders or transfers
  ~65% healthy      7-21 days
  ~20% overstocked  35-60 days          -> candidates to transfer stock out

Warehouses hold enough to supply their stores for 20-40 days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

AT_RISK, HEALTHY, OVERSTOCK = (1, 4), (7, 21), (35, 60)
MIX = (0.15, 0.65, 0.20)
WAREHOUSE_COVER = (20, 40)


def simulate_inventory(
    sales: pd.DataFrame,
    locations: pd.DataFrame,
    window_days: int = 28,
    seed: int = 42,
) -> pd.DataFrame:
    """Return inventory rows: location_id, product_id, on_hand, on_order.

    Args:
        sales: template-format sales (date, location_id, product_id, units_sold, ...).
        locations: template-format locations, including warehouses with supplied_by links.
        window_days: how many recent days define "average daily demand".
        seed: random seed so the demo is identical on every run.
    """
    rng = np.random.default_rng(seed)
    end = sales["date"].max()
    recent = sales[sales["date"] > end - pd.Timedelta(days=window_days)]

    avg = (
        recent.groupby(["location_id", "product_id"])["units_sold"].sum() / window_days
    ).rename("avg_daily").reset_index()
    avg = avg[avg["avg_daily"] > 0].copy()  # a store that hasn't sold it recently doesn't stock it

    band = rng.choice(3, size=len(avg), p=MIX)
    lows = np.array([AT_RISK[0], HEALTHY[0], OVERSTOCK[0]])[band]
    highs = np.array([AT_RISK[1], HEALTHY[1], OVERSTOCK[1]])[band]
    cover = rng.uniform(lows, highs)
    avg["on_hand"] = np.round(avg["avg_daily"] * cover, 1)
    avg["on_order"] = 0.0

    # Warehouses: cover for the combined demand of the stores they supply.
    supplier = locations.set_index("location_id")["supplied_by"]
    avg["warehouse"] = avg["location_id"].map(supplier)
    wh = avg.groupby(["warehouse", "product_id"])["avg_daily"].sum().reset_index()
    wh_cover = rng.uniform(*WAREHOUSE_COVER, size=len(wh))
    wh = pd.DataFrame({
        "location_id": wh["warehouse"],
        "product_id": wh["product_id"],
        "on_hand": np.round(wh["avg_daily"] * wh_cover, 1),
        "on_order": 0.0,
    })

    stores = avg[["location_id", "product_id", "on_hand", "on_order"]]
    return (
        pd.concat([stores, wh], ignore_index=True)
        .sort_values(["location_id", "product_id"])
        .reset_index(drop=True)
    )
