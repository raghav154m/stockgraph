"""Read endpoints for the dashboard. Everything lives under /api/datasets/{dataset_id}/...

Lists are ordered the way a manager would want to read them: most urgent first.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterator
from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import case, func, or_, select
from sqlalchemy.orm import Session

from backend.models import (Backtest, Dataset, Forecast, GraphEdge, InventoryStatus, Location, Metric, Product,
                            Recommendation, RollingMetric, Sale)

router = APIRouter(prefix="/api")

Z80 = 1.2816  # p10/p90 are p50 -/+ 1.2816 sigma (normal 80% interval), same as the ML code
STATUSES = ("stockout_risk", "reorder_due", "overstock", "healthy")
PRIORITY_ORDER = case({"high": 0, "medium": 1, "low": 2}, value=Recommendation.priority, else_=3)
STATUS_ORDER = case({s: i for i, s in enumerate(STATUSES)}, value=InventoryStatus.status, else_=len(STATUSES))
DISCLOSURE = ("Demo data is adapted from the Favorita grocery sales dataset (Corporación Favorita, Ecuador). "
              "Store locations are relabelled to Telangana, prices are shown in ₹, and the warehouse and "
              "inventory layer is simulated. Sales patterns, including holidays, come from the original data.")


# ---------- dependencies ----------

def get_db(request: Request) -> Iterator[Session]:
    with request.app.state.session_factory() as session:
        yield session


def get_dataset(dataset_id: str, db: Session = Depends(get_db)) -> Dataset:
    ds = db.get(Dataset, dataset_id)
    if ds is None:
        raise HTTPException(404, f"Dataset '{dataset_id}' not found")
    return ds


def _row(obj, *exclude: str) -> dict:
    return {c.name: getattr(obj, c.name) for c in obj.__table__.columns
            if c.name not in ("dataset_id", *exclude)}


def _names(db: Session, model, key: str, dataset_id: str) -> dict[str, str]:
    col = getattr(model, key)
    return dict(db.execute(select(col, model.name).where(model.dataset_id == dataset_id)).all())


# ---------- datasets ----------

@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/datasets")
def list_datasets(db: Session = Depends(get_db)) -> list[dict]:
    rows = db.scalars(select(Dataset).order_by(Dataset.id)).all()
    return [{"id": d.id, "name": d.name, "as_of": d.as_of, "forecast_model": d.forecast_model,
             "loaded_at": d.loaded_at} for d in rows]


@router.get("/datasets/{dataset_id}")
def dataset_overview(ds: Dataset = Depends(get_dataset), db: Session = Depends(get_db)) -> dict:
    """Headline numbers for the dashboard's first screen."""
    counts = dict(db.execute(select(Location.type, func.count()).where(Location.dataset_id == ds.id)
                             .group_by(Location.type)).all())
    n_products = db.scalar(select(func.count()).select_from(Product).where(Product.dataset_id == ds.id))
    meta = json.loads(ds.meta_json or "{}")
    return {
        "id": ds.id,
        "name": ds.name,
        "source": ds.source,
        "as_of": ds.as_of,
        "forecast_model": ds.forecast_model,
        "loaded_at": ds.loaded_at,
        "stores": counts.get("store", 0),
        "warehouses": counts.get("warehouse", 0),
        "products": n_products,
        "summary": json.loads(ds.summary_json),
        "meta": meta,
        "disclosure": DISCLOSURE if ds.id == "demo" else None,
    }


# ---------- locations and products ----------

@router.get("/datasets/{dataset_id}/locations")
def list_locations(type: str | None = Query(None, pattern="^(store|warehouse)$"),
                   ds: Dataset = Depends(get_dataset), db: Session = Depends(get_db)) -> list[dict]:
    """Stores and warehouses with a count of items in each stock status (for the map)."""
    q = select(Location).where(Location.dataset_id == ds.id).order_by(Location.location_id)
    if type:
        q = q.where(Location.type == type)
    locations = db.scalars(q).all()

    status_counts: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(STATUSES, 0))
    for loc, status, n in db.execute(
            select(InventoryStatus.location_id, InventoryStatus.status, func.count())
            .where(InventoryStatus.dataset_id == ds.id)
            .group_by(InventoryStatus.location_id, InventoryStatus.status)).all():
        status_counts[loc][status] = n

    open_recs = dict(db.execute(
        select(Recommendation.to_location, func.count())
        .where(Recommendation.dataset_id == ds.id, Recommendation.priority == "high")
        .group_by(Recommendation.to_location)).all())

    out = []
    for loc in locations:
        counts = status_counts[loc.location_id]
        worst = next((s for s in STATUSES if counts.get(s)), "healthy")
        out.append({**_row(loc), "status_counts": counts, "worst_status": worst,
                    "high_priority_recommendations": open_recs.get(loc.location_id, 0)})
    return out


@router.get("/datasets/{dataset_id}/locations/{location_id}")
def location_detail(location_id: str, ds: Dataset = Depends(get_dataset), db: Session = Depends(get_db)) -> dict:
    loc = db.get(Location, (ds.id, location_id))
    if loc is None:
        raise HTTPException(404, f"Location '{location_id}' not found")
    products = _names(db, Product, "product_id", ds.id)
    items = db.scalars(select(InventoryStatus)
                       .where(InventoryStatus.dataset_id == ds.id, InventoryStatus.location_id == location_id)
                       .order_by(STATUS_ORDER, InventoryStatus.days_of_cover.is_(None),
                                 InventoryStatus.days_of_cover)).all()
    accuracy = db.execute(select(Metric.model, Metric.mae, Metric.wape, Metric.bias)
                          .where(Metric.dataset_id == ds.id, Metric.location_id == location_id)).all()
    return {
        **_row(loc),
        "inventory": [{**_row(i), "product_name": products.get(i.product_id, i.product_id)} for i in items],
        "accuracy": [{"model": m, "mae": mae, "wape": w, "bias": b} for m, mae, w, b in accuracy],
    }


@router.get("/datasets/{dataset_id}/products")
def list_products(ds: Dataset = Depends(get_dataset), db: Session = Depends(get_db)) -> list[dict]:
    """Product families with network-wide stock and how many stores are at risk."""
    totals = {pid: (on_hand, at_risk, overstock) for pid, on_hand, at_risk, overstock in db.execute(
        select(InventoryStatus.product_id,
               func.sum(InventoryStatus.on_hand),
               func.sum(case((InventoryStatus.status == "stockout_risk", 1), else_=0)),
               func.sum(case((InventoryStatus.status == "overstock", 1), else_=0)))
        .where(InventoryStatus.dataset_id == ds.id, InventoryStatus.supplied_by != "SUPPLIER")
        .group_by(InventoryStatus.product_id)).all()}
    products = db.scalars(select(Product).where(Product.dataset_id == ds.id)
                          .order_by(Product.category, Product.name)).all()
    out = []
    for p in products:
        on_hand, at_risk, over = totals.get(p.product_id, (0.0, 0, 0))
        out.append({**_row(p), "store_on_hand": round(on_hand or 0.0, 1),
                    "stores_at_risk": int(at_risk or 0), "stores_overstocked": int(over or 0)})
    return out


# ---------- inventory status ----------

@router.get("/datasets/{dataset_id}/inventory")
def inventory(location_id: str | None = None, product_id: str | None = None,
              status: str | None = Query(None, pattern="^(stockout_risk|reorder_due|overstock|healthy)$"),
              location_type: str | None = Query(None, pattern="^(store|warehouse)$"),
              limit: int = Query(100, ge=1, le=5000), offset: int = Query(0, ge=0),
              ds: Dataset = Depends(get_dataset), db: Session = Depends(get_db)) -> dict:
    """Stock position and policy numbers per location x product, most urgent first."""
    q = select(InventoryStatus).where(InventoryStatus.dataset_id == ds.id)
    if location_id:
        q = q.where(InventoryStatus.location_id == location_id)
    if product_id:
        q = q.where(InventoryStatus.product_id == product_id)
    if status:
        q = q.where(InventoryStatus.status == status)
    if location_type == "warehouse":
        q = q.where(InventoryStatus.supplied_by == "SUPPLIER")
    elif location_type == "store":
        q = q.where(InventoryStatus.supplied_by != "SUPPLIER")

    total = db.scalar(select(func.count()).select_from(q.subquery()))
    rows = db.scalars(q.order_by(STATUS_ORDER, InventoryStatus.days_of_cover.is_(None),
                                 InventoryStatus.days_of_cover, InventoryStatus.location_id,
                                 InventoryStatus.product_id).limit(limit).offset(offset)).all()
    locs, prods = _names(db, Location, "location_id", ds.id), _names(db, Product, "product_id", ds.id)
    return {
        "total": total,
        "items": [{**_row(r), "location_name": locs.get(r.location_id, r.location_id),
                   "product_name": prods.get(r.product_id, r.product_id)} for r in rows],
    }


# ---------- forecasts ----------

@router.get("/datasets/{dataset_id}/forecast")
def forecast(location_id: str, product_id: str,
             models: str | None = Query(None, description="comma-separated, e.g. ensemble,baseline; default all"),
             history_days: int = Query(56, ge=0, le=365),
             ds: Dataset = Depends(get_dataset), db: Session = Depends(get_db)) -> dict:
    """Recent sales, the 28-day forecast per model with an 80% band, and the backtest, for one product
    at one store. `location_id=ALL` sums all stores (the band assumes store errors are independent)."""
    product = db.get(Product, (ds.id, product_id))
    if product is None:
        raise HTTPException(404, f"Product '{product_id}' not found")
    network = location_id == "ALL"
    if not network:
        loc = db.get(Location, (ds.id, location_id))
        if loc is None:
            raise HTTPException(404, f"Location '{location_id}' not found")
        if loc.type != "store":
            raise HTTPException(400, "Forecasts are made per store; warehouses are supplied from their stores' "
                                     "forecasts. Use a store id or ALL.")
    wanted = [m.strip() for m in models.split(",") if m.strip()] if models else None

    def scope(model):
        conds = [model.dataset_id == ds.id, model.product_id == product_id]
        if not network:
            conds.append(model.location_id == location_id)
        return conds

    # history
    last_sale = db.scalar(select(func.max(Sale.date)).where(*scope(Sale)))
    history = []
    if last_sale and history_days:
        start = (date.fromisoformat(last_sale) - timedelta(days=history_days - 1)).isoformat()
        history = [{"date": d, "units_sold": round(u, 3), "on_promotion": int(p or 0)} for d, u, p in db.execute(
            select(Sale.date, func.sum(Sale.units_sold), func.max(Sale.on_promotion))
            .where(*scope(Sale), Sale.date >= start).group_by(Sale.date).order_by(Sale.date)).all()]

    # forecasts
    fq = (select(Forecast.model, Forecast.date, func.sum(Forecast.p50), func.sum(Forecast.sigma * Forecast.sigma),
                 func.sum(Forecast.p10), func.sum(Forecast.p90))
          .where(*scope(Forecast)).group_by(Forecast.model, Forecast.date).order_by(Forecast.model, Forecast.date))
    if wanted:
        fq = fq.where(Forecast.model.in_(wanted))
    forecasts: dict[str, list[dict]] = defaultdict(list)
    for model, day, p50, var, p10, p90 in db.execute(fq).all():
        if network:  # summed band: sqrt of summed variances, not the sum of the bands
            sd = math.sqrt(var or 0.0)
            p10, p90 = max(p50 - Z80 * sd, 0.0), p50 + Z80 * sd
        forecasts[model].append({"date": day, "p50": round(p50, 3), "p10": round(p10, 3), "p90": round(p90, 3)})

    # backtest
    cols = ["actual", "baseline", "naive", "stgnn", "ensemble"]
    bq = (select(Backtest.date, *[func.sum(getattr(Backtest, c)) for c in cols])
          .where(*scope(Backtest)).group_by(Backtest.date).order_by(Backtest.date))
    backtest = [{"date": r[0], **{c: (round(v, 3) if v is not None else None) for c, v in zip(cols, r[1:])}}
                for r in db.execute(bq).all()]

    status = None
    if not network:
        s = db.get(InventoryStatus, (ds.id, location_id, product_id))
        status = _row(s) if s else None

    return {
        "location_id": location_id,
        "product_id": product_id,
        "product_name": product.name,
        "as_of": ds.as_of,
        "recommendation_model": ds.forecast_model,
        "history": history,
        "forecasts": dict(forecasts),
        "backtest": backtest,
        "inventory": status,
    }


# ---------- recommendations ----------

@router.get("/datasets/{dataset_id}/recommendations")
def recommendations(type: str | None = Query(None, pattern="^(transfer|reorder)$"),
                    priority: str | None = Query(None, pattern="^(high|medium|low)$"),
                    location_id: str | None = Query(None, description="matches the source or the destination"),
                    product_id: str | None = None,
                    limit: int = Query(50, ge=1, le=1000), offset: int = Query(0, ge=0),
                    ds: Dataset = Depends(get_dataset), db: Session = Depends(get_db)) -> dict:
    """Transfers and reorders, most urgent and most valuable first."""
    q = select(Recommendation).where(Recommendation.dataset_id == ds.id)
    if type:
        q = q.where(Recommendation.type == type)
    if priority:
        q = q.where(Recommendation.priority == priority)
    if location_id:
        q = q.where(or_(Recommendation.from_location == location_id, Recommendation.to_location == location_id))
    if product_id:
        q = q.where(Recommendation.product_id == product_id)

    sub = q.subquery()
    total, value = db.execute(select(func.count(), func.coalesce(func.sum(sub.c.value_inr), 0.0))).one()
    rows = db.scalars(q.order_by(PRIORITY_ORDER, Recommendation.deadline, Recommendation.value_inr.desc(),
                                 Recommendation.rec_id).limit(limit).offset(offset)).all()
    return {"total": total, "total_value_inr": round(value, 0), "items": [_rec(db, ds.id, r) for r in rows]}


@router.get("/datasets/{dataset_id}/recommendations/{rec_id}")
def recommendation(rec_id: str, ds: Dataset = Depends(get_dataset), db: Session = Depends(get_db)) -> dict:
    r = db.get(Recommendation, (ds.id, rec_id))
    if r is None:
        raise HTTPException(404, f"Recommendation '{rec_id}' not found")
    return _rec(db, ds.id, r)


def _rec(db: Session, dataset_id: str, r: Recommendation) -> dict:
    locs = _cached_names(db, dataset_id)
    return {**_row(r),
            "product_name": locs["products"].get(r.product_id, r.product_id),
            "from_name": "Supplier" if r.from_location == "SUPPLIER" else locs["locations"].get(r.from_location,
                                                                                               r.from_location),
            "to_name": locs["locations"].get(r.to_location, r.to_location)}


def _cached_names(db: Session, dataset_id: str) -> dict[str, dict[str, str]]:
    cache = db.info.setdefault("names", {})
    if dataset_id not in cache:
        cache[dataset_id] = {"locations": _names(db, Location, "location_id", dataset_id),
                             "products": _names(db, Product, "product_id", dataset_id)}
    return cache[dataset_id]


# ---------- model accuracy ----------

@router.get("/datasets/{dataset_id}/metrics")
def metrics(ds: Dataset = Depends(get_dataset), db: Session = Depends(get_db)) -> dict:
    """How accurate each model is: latest holdout overall and per store, plus the rolling holdouts."""
    rows = db.scalars(select(Metric).where(Metric.dataset_id == ds.id)).all()
    locs = _names(db, Location, "location_id", ds.id)
    overall = sorted(({"model": m.model, "mae": m.mae, "wape": m.wape, "bias": m.bias}
                      for m in rows if m.location_id == "ALL"), key=lambda r: r["wape"])
    by_store: dict[str, dict] = {}
    for m in rows:
        if m.location_id != "ALL":
            entry = by_store.setdefault(m.location_id, {"location_id": m.location_id,
                                                        "name": locs.get(m.location_id, m.location_id), "wape": {}})
            entry["wape"][m.model] = m.wape
    rolling = [_row(r) for r in db.scalars(select(RollingMetric).where(RollingMetric.dataset_id == ds.id)
                                           .order_by(RollingMetric.holdout_start, RollingMetric.model)).all()]
    training = json.loads(ds.training_json) if ds.training_json else None
    if training:  # the per-epoch history is for the "how it was trained" chart; keep everything else small
        training = {k: training[k] for k in ("settings", "best_epoch", "history", "rolling_wape", "holdout_wape",
                                             "best_model", "recommendation_model", "selection_basis",
                                             "ensemble_weight_stgnn", "note") if k in training}
    return {"forecast_model": ds.forecast_model, "overall": overall,
            "by_store": sorted(by_store.values(), key=lambda r: r["location_id"]),
            "rolling": rolling, "training": training}


# ---------- store graph ----------

@router.get("/datasets/{dataset_id}/graph")
def graph(ds: Dataset = Depends(get_dataset), db: Session = Depends(get_db)) -> dict:
    """The store network the ST-GNN learns from (stores, warehouses and edges) for the map."""
    nodes = [_row(loc) for loc in
             db.scalars(select(Location).where(Location.dataset_id == ds.id).order_by(Location.location_id)).all()]
    edges = [_row(e) for e in db.scalars(select(GraphEdge).where(GraphEdge.dataset_id == ds.id)).all()]
    return {"nodes": nodes, "edges": edges}
