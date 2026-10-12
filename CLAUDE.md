# StockGraph — context for AI assistants

Read this before writing code. The full plan lives in the "StockGraph — Project Plan" doc.

## What it is
A portfolio web app for inventory optimization across a network of stores and warehouses.
An ST-GNN forecasts demand per store x product family; a classic optimizer turns forecasts into
reorder and transfer recommendations. Target user: a non-technical inventory manager at a
mid-size multi-location retailer. Solo build, 2 weeks.

## Stack
- ML: Python 3.11+, pandas, LightGBM (baseline), PyTorch Geometric Temporal (ST-GNN, trained on Colab/Kaggle)
- Backend: FastAPI, SQLAlchemy, SQLite, background tasks for uploads
- Frontend: React + Vite + TypeScript, Tailwind, shadcn/ui, Recharts, Leaflet
- Hosting: Vercel (frontend), Render or Railway (backend)

## Layout
```
ml/data/       raw Favorita -> template CSVs (prepare.py), Telangana places, product catalogue, inventory simulation
ml/graph/      store graph (nodes.csv, edges.csv)
ml/models/     baseline and ST-GNN (day 3+)
ml/inventory/  safety stock, reorder points, transfers (day 3+)
backend/       FastAPI app (day 8+)
frontend/      React app (day 8+)
scripts/       one-off utilities (download_data.py)
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

## Conventions
- Small focused modules; pure functions that take/return DataFrames; CLI via `python -m ml.<pkg>.<module>`
- Deterministic: every random step takes a `seed` (default 42)
- Money in INR (₹); plain language in anything user-facing
- Add or update a pytest test for every pipeline change; tests must run without the real dataset
- Never commit data, models, `.env` or `kaggle.json`
- Commit style: `feat:`, `fix:`, `data:`, `docs:`, `test:`

## Decisions (don't change without asking)
- Forecast level: product family. Uploads use the baseline model only.
- No login in v1. Datasets are identified by id; demo is one dataset.
- Simulated inventory layer and Telangana relabelling must be disclosed in the README and About page.

## Status
- [x] Day 1: data prep, relabelling, simulated inventory, store graph
- [ ] Days 3–4: baseline + inventory optimizer
