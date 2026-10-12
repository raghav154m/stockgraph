"""Turn demand forecasts into stock status and recommendations.

Two-level network: stores are replenished from their regional warehouse; warehouses order
from suppliers. Standard, explainable inventory policy (periodic review, order-up-to):

  lead time L      stores: transport days from their warehouse (by distance)
                   warehouses: the product's supplier lead time
  review period R  how often orders are placed (7 days)
  safety stock     SS = z x sigma_daily x sqrt(L + R), z from the target service level
  reorder point    ROP = forecast demand over L + SS
  target level     S = forecast demand over L + R + SS
  reorder          if on_hand + on_order < ROP, order S - (on_hand + on_order)

Before ordering, a store that is short first tries a transfer from a nearby overstocked
store, which is faster and cheaper than waiting on the warehouse.

Status per store x product:
  stockout_risk  runs out before a delivery could arrive (red)
  reorder_due    below the reorder point (amber)
  overstock      more than `overstock_days` of cover (amber)
  healthy        (green)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

from ml.data.load import Dataset
from ml.data.telangana import haversine_km

KEYS = ["location_id", "product_id"]


@dataclass
class Policy:
    service_level: float = 0.95
    review_days: int = 7
    overstock_days: float = 35.0
    max_transfer_km: float = 120.0
    min_transfer_share: float = 0.2   # a transfer must cover at least 20% of the shortfall

    @property
    def z(self) -> float:
        return NormalDist().inv_cdf(self.service_level)


def transport_days(km: float) -> int:
    """Delivery time for a truck run of `km`: next day locally, longer for distant towns."""
    if km <= 60:
        return 1
    if km <= 200:
        return 2
    return 3


def _demand_over(cum: np.ndarray, days: float) -> float:
    """Forecast demand over a (possibly fractional) number of days from cumulative daily forecasts.

    Beyond the 28-day forecast, demand continues at the last week's average rate.
    """
    if days <= 0:
        return 0.0
    n = len(cum)
    if days <= n:
        whole = int(days)
        base = cum[whole - 1] if whole > 0 else 0.0
        frac = days - whole
        nxt = (cum[whole] - base) if whole < n else 0.0
        return float(base + frac * nxt)
    tail_rate = (cum[-1] - cum[-8]) / 7 if n >= 8 else cum[-1] / n
    return float(cum[-1] + (days - n) * tail_rate)


def _days_until_stockout(cum: np.ndarray, stock: float) -> float:
    """Days until cumulative forecast demand exceeds `stock` (extrapolated past 28 days)."""
    if stock <= 0:
        return 0.0
    n = len(cum)
    idx = int(np.searchsorted(cum, stock, side="right"))
    if idx < n:
        prev = cum[idx - 1] if idx > 0 else 0.0
        step = cum[idx] - prev
        return float(idx + ((stock - prev) / step if step > 0 else 0.0))
    tail_rate = (cum[-1] - cum[-8]) / 7 if n >= 8 else cum[-1] / max(n, 1)
    if tail_rate <= 0:
        return math.inf
    return float(n + (stock - cum[-1]) / tail_rate)


def _rupees(x: float) -> str:
    """Indian-style rupee formatting: 1234567 -> ₹12,34,567."""
    x = int(round(x))
    s = str(abs(x))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts) + "," + tail
    return ("-₹" if x < 0 else "₹") + s


def _units(x: float) -> str:
    n = int(round(x))
    return f"{n:,} unit" + ("" if n == 1 else "s")


def _days(n: float) -> str:
    n = int(round(n))
    return f"{n} day" + ("" if n == 1 else "s")


def stock_status(ds: Dataset, forecasts: pd.DataFrame, policy: Policy) -> pd.DataFrame:
    """One row per store x product with its policy numbers and status."""
    locs = ds.locations.set_index("location_id")
    stores = ds.stores.set_index("location_id")
    fc = forecasts.sort_values(KEYS + ["date"])
    sigma = fc.groupby(KEYS)["sigma"].first()

    inv = ds.inventory.set_index(KEYS)
    rows = []
    for (loc, prod), g in fc.groupby(KEYS, sort=False):
        if loc not in stores.index:
            continue
        wh = stores.at[loc, "supplied_by"]
        km = haversine_km(stores.at[loc, "latitude"], stores.at[loc, "longitude"],
                          locs.at[wh, "latitude"], locs.at[wh, "longitude"]) if wh in locs.index else 0.0
        lead = transport_days(km)
        cum = np.cumsum(g["p50"].to_numpy())
        on_hand = float(inv["on_hand"].get((loc, prod), 0.0))
        on_order = float(inv["on_order"].get((loc, prod), 0.0))
        ss = policy.z * float(sigma[(loc, prod)]) * math.sqrt(lead + policy.review_days)
        rop = _demand_over(cum, lead) + ss
        target = _demand_over(cum, lead + policy.review_days) + ss
        days_left = _days_until_stockout(cum, on_hand)
        avg_daily = float(cum[-1] / len(cum))

        if days_left < lead:
            status = "stockout_risk"
        elif on_hand + on_order < rop:
            status = "reorder_due"
        elif days_left > policy.overstock_days and on_hand > 1.5 * target:
            status = "overstock"
        else:
            status = "healthy"

        rows.append({
            "location_id": loc, "product_id": prod, "supplied_by": wh,
            "on_hand": round(on_hand, 1), "on_order": round(on_order, 1),
            "avg_daily_forecast": round(avg_daily, 2),
            "days_of_cover": round(days_left, 1) if math.isfinite(days_left) else None,
            "lead_time_days": lead,
            "safety_stock": round(ss, 1),
            "reorder_point": round(rop, 1),
            "target_level": round(target, 1),
            "status": status,
        })
    return pd.DataFrame(rows)


def recommend(ds: Dataset, forecasts: pd.DataFrame, policy: Policy | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (stock status per store x product incl. warehouses, recommendations)."""
    policy = policy or Policy()
    status = stock_status(ds, forecasts, policy)
    locs = ds.locations.set_index("location_id")
    products = ds.products.set_index("product_id")
    names = locs["name"].to_dict()
    as_of = ds.as_of
    recs: list[dict] = []

    def km(a: str, b: str) -> float:
        return haversine_km(locs.at[a, "latitude"], locs.at[a, "longitude"], locs.at[b, "latitude"], locs.at[b, "longitude"])

    def priority(days_left: float, lead: int) -> str:
        if days_left < lead:                         # runs out before a normal delivery can arrive
            return "high"
        if days_left < lead + policy.review_days:    # runs out before the next weekly review
            return "medium"
        return "low"

    def deadline(days_left: float, lead: int) -> str:
        slack = 0 if not math.isfinite(days_left) else max(0, int(days_left - lead))
        return (as_of + pd.Timedelta(days=min(slack, 28))).strftime("%Y-%m-%d")

    st = status.set_index(KEYS)
    st["days_left"] = st["days_of_cover"].astype(float).fillna(math.inf)  # no demand -> never runs out
    position = (st["on_hand"] + st["on_order"]).to_dict()
    surplus = {k: max(0.0, r.on_hand - r.target_level) for k, r in st.iterrows() if r.status == "overstock"}

    # 1. Transfers: short stores take stock from nearby overstocked stores, nearest first.
    short = st[st["status"].isin(["stockout_risk", "reorder_due"])].sort_values("days_left")
    for (loc, prod), r in short.iterrows():
        need = r.target_level - position[(loc, prod)]
        if need <= 0:
            continue
        donors = [(km(d, loc), d) for (d, p), left in surplus.items() if p == prod and left > 0 and d != loc]
        for dist, donor in sorted(donors):
            if dist > policy.max_transfer_km or transport_days(dist) > r.lead_time_days:
                break  # donors are sorted by distance, so every later one is too far or too slow
            qty = math.floor(min(need, surplus[(donor, prod)]))
            if qty <= 0 or qty < policy.min_transfer_share * need:
                continue
            surplus[(donor, prod)] -= qty
            position[(loc, prod)] += qty
            need -= qty
            unit_cost = float(products.at[prod, "unit_cost"])
            days_txt = "less than a day" if r.days_left < 1 else _days(r.days_left)
            recs.append({
                "type": "transfer", "product_id": prod, "from_location": donor, "to_location": loc,
                "quantity": qty, "deadline": as_of.strftime("%Y-%m-%d"),
                "priority": priority(r.days_left, r.lead_time_days),
                "value_inr": round(qty * unit_cost, 0),
                "reason": (f"{names[loc]} runs out of {products.at[prod, 'name']} in {days_txt}; "
                           f"{names[donor]} ({dist:.0f} km away) has {_units(surplus[(donor, prod)] + qty)} "
                           f"more than it needs. Moving {_units(qty)} arrives in "
                           f"{_days(transport_days(dist))} and frees {_rupees(qty * unit_cost)} of idle stock."),
            })
            if need <= 0:
                break

    # 2. Store reorders from the warehouse for whatever is still below the reorder point.
    shipped: dict[tuple[str, str], float] = {}
    for (loc, prod), r in st.iterrows():
        pos = position[(loc, prod)]
        if pos >= r.reorder_point:
            continue
        qty = math.ceil(r.target_level - pos)
        if qty <= 0:
            continue
        wh = r.supplied_by
        shipped[(wh, prod)] = shipped.get((wh, prod), 0.0) + qty
        unit_cost = float(products.at[prod, "unit_cost"])
        if not math.isfinite(r.days_left):
            days_txt = "beyond the forecast window"
        else:
            days_txt = "today" if r.days_left < 1 else f"in {_days(r.days_left)}"
        recs.append({
            "type": "reorder", "product_id": prod, "from_location": wh, "to_location": loc,
            "quantity": qty, "deadline": deadline(r.days_left, r.lead_time_days),
            "priority": priority(r.days_left, r.lead_time_days),
            "value_inr": round(qty * unit_cost, 0),
            "reason": (f"{names[loc]} has {_units(pos)} of {products.at[prod, 'name']} on hand or on the way, "
                       f"below its reorder point of {_units(r.reorder_point)}. At forecast demand it runs out {days_txt}; "
                       f"delivery from {names[wh]} takes {_days(r.lead_time_days)}."),
        })

    # 3. Warehouses: cover their stores' combined demand over the supplier lead time.
    fc = forecasts.copy()
    fc["warehouse"] = fc["location_id"].map(ds.stores.set_index("location_id")["supplied_by"])
    fc = fc.dropna(subset=["warehouse"])
    daily = fc.groupby(["warehouse", "product_id", "date"])["p50"].sum().reset_index()
    var = fc.groupby(["warehouse", "product_id", "location_id"])["sigma"].first() ** 2
    wh_sigma = np.sqrt(var.groupby(level=[0, 1]).sum())
    inv = ds.inventory.set_index(KEYS)

    wh_rows = []
    for (wh, prod), g in daily.groupby(["warehouse", "product_id"]):
        cum = np.cumsum(g.sort_values("date")["p50"].to_numpy())
        lead = int(products.at[prod, "lead_time_days"]) if prod in products.index else 7
        on_hand = float(inv["on_hand"].get((wh, prod), 0.0))
        on_order = float(inv["on_order"].get((wh, prod), 0.0))
        out = shipped.get((wh, prod), 0.0)
        ss = policy.z * float(wh_sigma[(wh, prod)]) * math.sqrt(lead + policy.review_days)
        rop = _demand_over(cum, lead) + ss
        target = _demand_over(cum, lead + policy.review_days) + ss
        # Stock that will leave for this week's store reorders is already spoken for.
        free = on_hand - out
        days_left = _days_until_stockout(cum, max(free, 0.0))
        pos = free + on_order
        if free < 0 or days_left < lead:
            st_wh = "stockout_risk"
        elif pos < rop:
            st_wh = "reorder_due"
        elif days_left > 2 * policy.overstock_days:
            st_wh = "overstock"
        else:
            st_wh = "healthy"
        wh_rows.append({
            "location_id": wh, "product_id": prod, "supplied_by": "SUPPLIER",
            "on_hand": round(on_hand, 1), "on_order": round(on_order, 1),
            "avg_daily_forecast": round(float(cum[-1] / len(cum)), 2),
            "days_of_cover": round(days_left, 1) if math.isfinite(days_left) else None,
            "lead_time_days": lead, "safety_stock": round(ss, 1),
            "reorder_point": round(rop, 1), "target_level": round(target, 1), "status": st_wh,
        })
        if pos < rop:
            qty = math.ceil(target - pos)
            unit_cost = float(products.at[prod, "unit_cost"])
            short_txt = (f" Store orders this week ({_units(out)}) exceed its stock by {_units(-free)}."
                         if free < 0 else "")
            recs.append({
                "type": "reorder", "product_id": prod, "from_location": "SUPPLIER", "to_location": wh,
                "quantity": qty, "deadline": deadline(days_left, lead),
                "priority": priority(days_left, lead), "value_inr": round(qty * unit_cost, 0),
                "reason": (f"{names[wh]} needs {_units(qty)} of {products.at[prod, 'name']} to keep its stores "
                           f"supplied for the next {_days(lead + policy.review_days)}; supplier lead time is "
                           f"{_days(lead)}.{short_txt}"),
            })

    all_status = pd.concat([status, pd.DataFrame(wh_rows)], ignore_index=True)
    rec_df = pd.DataFrame(recs, columns=["type", "product_id", "from_location", "to_location", "quantity",
                                         "deadline", "priority", "value_inr", "reason"])
    order = {"high": 0, "medium": 1, "low": 2}
    rec_df = rec_df.sort_values(["priority", "deadline", "value_inr"], key=lambda c: c.map(order) if c.name == "priority" else c,
                                ascending=[True, True, False]).reset_index(drop=True)
    rec_df.insert(0, "rec_id", [f"R{i + 1:04d}" for i in range(len(rec_df))])
    return all_status, rec_df
