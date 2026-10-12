"""Database tables. One row in `datasets` per dataset (the demo now, uploads later); every other
table is keyed by `dataset_id` so datasets never mix.

Columns mirror the CSVs written by `ml.pipeline` (see the results contract in CLAUDE.md).
Dates are stored as ISO strings (YYYY-MM-DD), which sort correctly and need no conversion.
"""

from __future__ import annotations

from sqlalchemy import Float, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    source: Mapped[str] = mapped_column(Text, default="")
    as_of: Mapped[str] = mapped_column(String(10))
    forecast_model: Mapped[str] = mapped_column(String(32))     # model that drove the recommendations
    loaded_at: Mapped[str] = mapped_column(String(32))
    summary_json: Mapped[str] = mapped_column(Text)            # results/summary.json
    meta_json: Mapped[str] = mapped_column(Text, default="{}")  # meta.json (demo only)
    training_json: Mapped[str | None] = mapped_column(Text, nullable=True)  # stgnn_training.json, if any


class Location(Base):
    __tablename__ = "locations"
    dataset_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    location_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    type: Mapped[str] = mapped_column(String(16))          # store / warehouse
    city: Mapped[str] = mapped_column(String(100))
    region: Mapped[str] = mapped_column(String(100))
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    supplied_by: Mapped[str] = mapped_column(String(32))   # warehouse id, or SUPPLIER for warehouses


class Product(Base):
    __tablename__ = "products"
    dataset_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    product_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    category: Mapped[str] = mapped_column(String(100))
    unit_cost: Mapped[float] = mapped_column(Float)
    lead_time_days: Mapped[int] = mapped_column(Integer)


class Sale(Base):
    """Recent daily sales history (the chart's 'actual' line); the full history stays in the CSVs."""
    __tablename__ = "sales"
    dataset_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    location_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    product_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    date: Mapped[str] = mapped_column(String(10), primary_key=True)
    units_sold: Mapped[float] = mapped_column(Float)
    on_promotion: Mapped[int] = mapped_column(Integer, default=0)


class Forecast(Base):
    __tablename__ = "forecasts"
    dataset_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    location_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    product_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model: Mapped[str] = mapped_column(String(32), primary_key=True)   # baseline / stgnn / ensemble
    date: Mapped[str] = mapped_column(String(10), primary_key=True)
    p50: Mapped[float] = mapped_column(Float)
    p10: Mapped[float] = mapped_column(Float)
    p90: Mapped[float] = mapped_column(Float)
    sigma: Mapped[float] = mapped_column(Float)


class Backtest(Base):
    __tablename__ = "backtest"
    dataset_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    location_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    product_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    date: Mapped[str] = mapped_column(String(10), primary_key=True)
    actual: Mapped[float] = mapped_column(Float)
    baseline: Mapped[float | None] = mapped_column(Float, nullable=True)
    naive: Mapped[float | None] = mapped_column(Float, nullable=True)
    stgnn: Mapped[float | None] = mapped_column(Float, nullable=True)
    ensemble: Mapped[float | None] = mapped_column(Float, nullable=True)


class Metric(Base):
    """Latest-holdout error per model, overall (location_id = ALL) and per store."""
    __tablename__ = "metrics"
    dataset_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model: Mapped[str] = mapped_column(String(32), primary_key=True)
    location_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    mae: Mapped[float] = mapped_column(Float)
    wape: Mapped[float] = mapped_column(Float)
    bias: Mapped[float] = mapped_column(Float)


class RollingMetric(Base):
    """Error per model on each of the back-to-back holdouts (holdout_start = ALL is pooled)."""
    __tablename__ = "metrics_rolling"
    dataset_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    holdout_start: Mapped[str] = mapped_column(String(10), primary_key=True)
    model: Mapped[str] = mapped_column(String(32), primary_key=True)
    mae: Mapped[float] = mapped_column(Float)
    wape: Mapped[float] = mapped_column(Float)
    bias: Mapped[float] = mapped_column(Float)


class InventoryStatus(Base):
    __tablename__ = "inventory_status"
    dataset_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    location_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    product_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    supplied_by: Mapped[str] = mapped_column(String(32))
    on_hand: Mapped[float] = mapped_column(Float)
    on_order: Mapped[float] = mapped_column(Float)
    avg_daily_forecast: Mapped[float] = mapped_column(Float)
    days_of_cover: Mapped[float | None] = mapped_column(Float, nullable=True)  # None = no demand expected
    lead_time_days: Mapped[float] = mapped_column(Float)
    safety_stock: Mapped[float] = mapped_column(Float)
    reorder_point: Mapped[float] = mapped_column(Float)
    target_level: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16))   # stockout_risk / reorder_due / overstock / healthy


class Recommendation(Base):
    __tablename__ = "recommendations"
    dataset_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    rec_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    type: Mapped[str] = mapped_column(String(16))           # transfer / reorder
    product_id: Mapped[str] = mapped_column(String(64))
    from_location: Mapped[str] = mapped_column(String(32))  # store, warehouse or SUPPLIER
    to_location: Mapped[str] = mapped_column(String(32))
    quantity: Mapped[float] = mapped_column(Float)
    deadline: Mapped[str] = mapped_column(String(10))
    priority: Mapped[str] = mapped_column(String(8))        # high / medium / low
    value_inr: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(Text)


class GraphEdge(Base):
    __tablename__ = "graph_edges"
    dataset_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    target_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    edge_type: Mapped[str] = mapped_column(String(16), primary_key=True)   # same_city / correlation / nearest
    weight: Mapped[float] = mapped_column(Float)
    distance_km: Mapped[float] = mapped_column(Float)


# Lookups the dashboard makes often.
Index("ix_recs_priority", Recommendation.dataset_id, Recommendation.priority, Recommendation.value_inr)
Index("ix_status_status", InventoryStatus.dataset_id, InventoryStatus.status)

# Every table that holds per-dataset rows, in load order (used to delete and reload a dataset).
DATA_TABLES = [Location, Product, Sale, Forecast, Backtest, Metric, RollingMetric,
               InventoryStatus, Recommendation, GraphEdge]
