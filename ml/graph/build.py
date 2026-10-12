"""Build the store graph the ST-GNN learns from.

Nodes are stores. Edges connect stores whose demand is likely to move together:

  same_city     stores in the same city (weight falls off with distance)
  correlation   each store's top-k most correlated stores by week-over-week sales change
  nearest       fallback so no store is left without a neighbour

Warehouses are not forecast nodes; the inventory optimizer uses them separately.

Usage:
  python -m ml.graph.build                    # reads data/processed, writes data/processed/graph
  python -m ml.graph.build --top-k 3 --min-corr 0.4
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.telangana import haversine_km

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "data" / "processed"
CITY_SCALE_KM = 10.0  # same-city weight = exp(-distance / 10 km)


def load_stores(data_dir: Path) -> pd.DataFrame:
    locations = pd.read_csv(data_dir / "locations.csv")
    stores = locations[locations["type"] == "store"].sort_values("location_id").reset_index(drop=True)
    stores.insert(0, "node_index", range(len(stores)))
    return stores


def sales_correlation(sales: pd.DataFrame, store_ids: list[str]) -> pd.DataFrame:
    """Correlation of week-over-week change in log total daily sales.

    Differencing by 7 days removes trend and the weekly cycle, which every store shares;
    what is left is the store-specific movement we want neighbours to share.
    Days with zero total sales (store closed or not yet open) are ignored.
    """
    totals = sales.pivot_table(index="date", columns="location_id", values="units_sold", aggfunc="sum")
    totals = totals.reindex(columns=store_ids)
    signal = np.log(totals.where(totals > 0)).diff(7)
    return signal.corr(min_periods=60)


def build_edges(stores: pd.DataFrame, corr: pd.DataFrame, top_k: int, min_corr: float) -> pd.DataFrame:
    idx = dict(zip(stores["location_id"], stores["node_index"]))
    coords = stores.set_index("location_id")[["latitude", "longitude"]]

    def dist(a: str, b: str) -> float:
        return haversine_km(*coords.loc[a], *coords.loc[b])

    edges: dict[tuple[str, str], dict] = {}

    def add(a: str, b: str, weight: float, kind: str) -> None:
        key = tuple(sorted((a, b)))
        if key[0] == key[1]:
            return
        if key not in edges or weight > edges[key]["weight"]:
            edges[key] = {"weight": round(float(weight), 4), "edge_type": kind}

    # 1. Same city
    for _, group in stores.groupby("city"):
        ids = group["location_id"].tolist()
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                add(a, b, math.exp(-dist(a, b) / CITY_SCALE_KM), "same_city")

    # 2. Correlated demand
    for a in stores["location_id"]:
        if a not in corr.columns:
            continue
        ranked = corr[a].drop(a, errors="ignore").dropna().sort_values(ascending=False)
        for b, value in ranked.head(top_k).items():
            if value >= min_corr:
                add(a, b, value, "correlation")

    # 3. No isolated stores: link to the nearest store
    connected = {s for key in edges for s in key}
    for a in stores["location_id"]:
        if a not in connected:
            b = min((s for s in stores["location_id"] if s != a), key=lambda s: dist(a, s))
            add(a, b, 0.5, "nearest")

    rows = []
    for (a, b), attrs in edges.items():
        rows.append({
            "source": idx[a], "target": idx[b],
            "source_id": a, "target_id": b,
            **attrs,
            "distance_km": round(dist(a, b), 1),
        })
    return pd.DataFrame(rows).sort_values(["source", "target"]).reset_index(drop=True)


def build_graph(data_dir: Path = DEFAULT_DIR, top_k: int = 3, min_corr: float = 0.4) -> dict:
    stores = load_stores(data_dir)
    sales = pd.read_csv(data_dir / "sales.csv", parse_dates=["date"])
    corr = sales_correlation(sales, stores["location_id"].tolist())
    edges = build_edges(stores, corr, top_k, min_corr)

    out = data_dir / "graph"
    out.mkdir(parents=True, exist_ok=True)
    stores[["node_index", "location_id", "name", "city", "region", "latitude", "longitude", "supplied_by"]].to_csv(
        out / "nodes.csv", index=False
    )
    edges.to_csv(out / "edges.csv", index=False)

    degree = pd.concat([edges["source"], edges["target"]]).value_counts().reindex(stores["node_index"], fill_value=0)
    return {
        "nodes": len(stores),
        "edges": len(edges),
        "by_type": edges["edge_type"].value_counts().to_dict(),
        "avg_degree": round(float(degree.mean()), 1),
        "min_degree": int(degree.min()),
        "max_degree": int(degree.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-corr", type=float, default=0.4)
    args = parser.parse_args()

    stats = build_graph(args.data, args.top_k, args.min_corr)
    print(f"Wrote graph to {args.data / 'graph'}")
    for key, value in stats.items():
        print(f"  {key:<11} {value}")


if __name__ == "__main__":
    main()
