# StockGraph

**AI-driven inventory optimization for multi-location retail.** StockGraph forecasts demand across a network of stores and warehouses with a Spatio-Temporal Graph Neural Network (ST-GNN), then tells inventory managers what to reorder, where to move stock, and when.

> Work in progress — a 2-week solo portfolio build.

## Status

- [x] Day 1: data preparation, Telangana relabelling, simulated inventory layer, store graph
- [ ] Days 3–4: baseline model and inventory optimizer
- [ ] Days 5–7: ST-GNN
- [ ] Days 8–11: FastAPI backend and React dashboard
- [ ] Days 12–13: upload flow and deployment
- [ ] Day 14: polish, screenshots, demo video

## Quick start (data pipeline)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Download Favorita from Kaggle (needs a Kaggle API token, see below)
python scripts/download_data.py

# 2. Slice, relabel to Telangana, simulate inventory, write the template CSVs
python -m ml.data.prepare

# 3. Build the store graph
python -m ml.graph.build
```

Outputs land in `data/processed/` in the same four-file format users upload:
`locations.csv`, `products.csv`, `sales.csv`, `inventory.csv`, plus `graph/nodes.csv` and `graph/edges.csv`.

### Kaggle setup

1. Create a free account at kaggle.com and open the
   [Store Sales – Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) competition.
2. Click **Join Competition** and accept the rules (required before downloading).
3. Go to **Settings → API → Create New Token**. Save `kaggle.json` to `~/.kaggle/kaggle.json`
   (Windows: `C:\Users\<you>\.kaggle\kaggle.json`).

The raw data is not committed to this repo: it is large and Kaggle's terms don't allow redistribution.

## Data transparency

Demo data is adapted from the Favorita grocery sales dataset (Corporación Favorita, Ecuador). Store
locations are relabelled to Telangana, prices are shown in ₹, and the warehouse and inventory layer is
simulated. Sales patterns, including holidays, come from the original Ecuadorian data.

## Tech stack

Python · pandas · LightGBM · PyTorch Geometric Temporal · FastAPI · SQLite · React · Vite · TypeScript · Tailwind · shadcn/ui · Recharts · Leaflet
