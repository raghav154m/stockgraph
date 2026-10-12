# StockGraph — context for AI assistants

Read this before writing code. The full plan lives in the "StockGraph — Project Plan" doc.

## What it is
A portfolio web app for inventory optimization across a network of stores and warehouses.
An ST-GNN forecasts demand per store x product family; a classic optimizer turns forecasts into
reorder and transfer recommendations. Target user: a non-technical inventory manager at a
mid-size multi-location retailer. Solo build, 2 weeks.

## Stack
- ML: Python 3.11+, pandas, scikit-learn HistGradientBoosting (baseline), PyTorch (ST-GNN: A3T-GCN implemented directly, dense adjacency, trains on CPU)
- Backend: FastAPI, SQLAlchemy, SQLite, background tasks for uploads
- Frontend: React + Vite + TypeScript, Tailwind, shadcn/ui, Recharts, Leaflet
- Hosting: Vercel (frontend), Render or Railway (backend)

## Layout
```
ml/data/       raw Favorita -> template CSVs (prepare.py), Telangana places, product catalogue, inventory simulation
ml/graph/      store graph (nodes.csv, edges.csv)
ml/data/load.py   loads any template dataset into a complete daily panel (fills closed days)
ml/models/     baseline.py: direct 28-day gradient-boosted model + seasonal naive
ml/models/stgnn/  data.py (numpy windows, adjacency), model.py (T-GCN cell + attention + decoder, residual mode),
                  evaluate.py (rolling holdouts, out-of-sample baseline cubes, scoring), train.py
ml/models/ensemble.py  50/50 blend of baseline + ST-GNN; the blend drives recommendations by default (no torch needed)
ml/inventory/  optimizer.py: lead times, safety stock, reorder points, status, transfers, reorders
ml/pipeline.py   one command: backtest -> forecast -> optimize -> data/processed/results/
backend/       FastAPI app: models.py (SQLAlchemy tables, keyed by dataset_id), loader.py (results/ -> SQLite,
               `python -m backend.loader`), api.py (read endpoints under /api), main.py (create_app, autoloads demo)
frontend/      React app: src/lib/api.ts (typed client), src/pages/ (Overview, MapPage, Recommendations, Forecast,
               Accuracy, About), src/components/ui/ (shadcn/ui, hand-added: registry unreachable from cloud sessions),
               status colours and labels in src/lib/status.ts, theme tokens in src/index.css
scripts/       one-off utilities (download_data.py, fake_demo.py: fake demo data for running the API/UI)
tests/         pytest; fake_favorita.py makes small test data
data/          gitignored: raw/ from Kaggle, processed/ outputs
```

## Data contract (the upload template)
Demo data and user uploads use the same four files:
- locations.csv: location_id, name, type (store/warehouse), city, region, latitude, longitude, supplied_by
- products.csv: product_id, name, category, unit_cost (INR), lead_time_days
- sales.csv: date, location_id, product_id, units_sold, on_promotion
- inventory.csv: location_id, product_id, on_hand, on_order
Demo extras: calendar.csv (is_holiday, oil_price, day_of_week), store_map.csv, meta.json.
IDs: stores `ST-01`..`ST-54`, warehouses `WH-SHM`, `WH-MDC`, `WH-WGL`, products are slugs like `grocery-i`.

## Results contract (written by ml.pipeline, loaded into SQLite later)
- forecasts.csv: location_id, product_id, date, model, p50, p10, p90, sigma (28 days ahead)
- backtest.csv: location_id, product_id, date, actual, baseline, naive (last 28 days held out)
- metrics.csv: model, location_id (ALL or store), mae, wape, bias
- inventory_status.csv: location_id, product_id, supplied_by, on_hand, on_order, avg_daily_forecast,
  days_of_cover, lead_time_days, safety_stock, reorder_point, target_level,
  status (stockout_risk / reorder_due / overstock / healthy)
- recommendations.csv: rec_id, type (transfer / reorder), product_id, from_location (store, warehouse or SUPPLIER),
  to_location, quantity, deadline, priority (high / medium / low), value_inr, reason
- forecasts_stgnn.csv: same columns as forecasts.csv, model = stgnn (from ml.models.stgnn.train)
- backtest.csv gains a `stgnn` column and metrics.csv `stgnn` rows after ST-GNN training
- forecasts_ensemble.csv: 50/50 blend; backtest.csv `ensemble` column, metrics.csv `ensemble` rows
- metrics_rolling.csv: holdout_start (date or ALL), model, mae, wape, bias over 4 back-to-back 28-day holdouts
- stgnn_training.json: settings, per-epoch loss and validation WAPE, rolling and latest-holdout WAPE, chosen model
- summary.json: headline numbers for the dashboard; `forecast_model` says which model drove recommendations

## Backend settings (env vars)
STOCKGRAPH_DB_URL (default sqlite:///data/stockgraph.db), STOCKGRAPH_DATA_DIR (default data/processed),
STOCKGRAPH_AUTOLOAD_DEMO (default 1), STOCKGRAPH_CORS_ORIGINS (default the Vite dev server).

## Conventions
- Small focused modules; pure functions that take/return DataFrames; CLI via `python -m ml.<pkg>.<module>`
- Deterministic: every random step takes a `seed` (default 42)
- Money in INR (₹); plain language in anything user-facing
- Add or update a pytest test for every pipeline change; tests must run without the real dataset
- Never commit data, models, `.env` or `kaggle.json`
- Commit style: `feat:`, `fix:`, `data:`, `docs:`, `test:`

## Decisions (don't change without asking)
- Forecast level: product family. Uploads use the baseline model only.
- Demo recommendations use the ST-GNN + gradient boosting blend (project is built around the ST-GNN;
  4-holdout WAPE: baseline 13.9%, blend 14.0%, ST-GNN 14.8%, naive 16.2%).
- No login in v1. Datasets are identified by id; demo is one dataset.
- Simulated inventory layer and Telangana relabelling must be disclosed in the README and About page.

## Status
- [x] Day 1: data prep, relabelling, simulated inventory, store graph
- [x] Days 3–4: baseline + inventory optimizer
- [x] Days 5–7: ST-GNN (verify on real data)
- [x] Day 8: backend read API (tests/test_backend.py; needs fastapi, sqlalchemy, httpx)
- [x] Days 9–11: frontend (dev: `npm run dev` in frontend/, proxies /api to :8000)

## Frontend conventions
- Status colours mean stock status only, always with icon + label (StatusBadge, StatusTape); models keep fixed colours
  (MODEL_COLOR) whatever is shown
- ₹ with Indian grouping and lakh/crore (src/lib/format.ts); plain words, sentence case; no hard-coded result numbers
  except the 4-holdout WAPE on the About page
- Filters live in the URL query so views can be linked; `npm run lint` and `npx tsc -b` must pass
