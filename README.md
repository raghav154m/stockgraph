# StockGraph

**AI-driven inventory optimization for multi-location retail.** StockGraph forecasts demand across a network of stores and warehouses with a Spatio-Temporal Graph Neural Network (ST-GNN), then tells inventory managers what to reorder, where to move stock, and when.

> Work in progress — a 2-week solo portfolio build.

## Status

- [x] Day 1: data preparation, Telangana relabelling, simulated inventory layer, store graph
- [x] Days 3–4: baseline model and inventory optimizer
- [x] Days 5–7: ST-GNN
- [ ] Days 8–11: FastAPI backend and React dashboard
- [ ] Days 12–13: upload flow and deployment
- [ ] Day 14: polish, screenshots, demo video

## Quick start (data pipeline)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Download Favorita from Kaggle (needs a Kaggle API token, see below)
python scripts/download_data.py

# 2. Slice, relabel to Telangana, simulate inventory, write the template CSVs
python -m ml.data.prepare

# 3. Build the store graph
python -m ml.graph.build

# 4. Forecast, optimize inventory, write recommendations (about 1–3 minutes)
python -m ml.pipeline

# 5. Train the ST-GNN, blend it with the baseline, use the most accurate for recommendations
python -m ml.models.stgnn.train
```

Step 4 writes `data/processed/results/`: 28-day forecasts with uncertainty bands, a backtest against a
seasonal-naive benchmark, stock status per store and warehouse, and prioritised reorder and transfer
recommendations with plain-language reasons.

Outputs land in `data/processed/` in the same four-file format users upload:
`locations.csv`, `products.csv`, `sales.csv`, `inventory.csv`, plus `graph/nodes.csv` and `graph/edges.csv`.

### Kaggle setup

1. Create a free account at kaggle.com and open the
   [Store Sales – Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) competition.
2. Click **Join Competition** and accept the rules (required before downloading).
3. Go to **Settings → API → Generate New Token**, copy the token, and save it:
   ```bash
   mkdir -p ~/.kaggle && pbpaste > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
   ```
   (Requires `kaggle>=1.8`. A legacy `~/.kaggle/kaggle.json` also works.)

The raw data is not committed to this repo: it is large and Kaggle's terms don't allow redistribution.

## How the ST-GNN works

Each store is a node in a graph; edges link stores in the same city and stores whose sales move
together. The model is an A3T-GCN, written directly in PyTorch:

1. **Encoder:** a graph-convolutional GRU reads the last 56 days. Each day, every store updates its
   state from its own sales, the calendar, and its neighbours' states.
2. **Attention** over those 56 days picks out the ones that matter most.
3. **Decoder:** for each of the next 28 days, it starts from the gradient-boosted baseline's
   forecast and learns a correction from the store's state, last year's sales, the known calendar
   and the network context. Untrained, it reproduces the baseline exactly; training only has to
   learn what the store network adds. The baseline forecasts it learns from are out-of-sample
   (refit every 8 weeks), so it never sees a forecast that already knew the answer.

All models (ST-GNN, baseline, a fixed 50/50 blend, seasonal naive) are scored on four
back-to-back 28-day holdouts. Whichever has the lowest pooled error drives the recommendations.

## Data transparency

Demo data is adapted from the Favorita grocery sales dataset (Corporación Favorita, Ecuador). Store
locations are relabelled to Telangana, prices are shown in ₹, and the warehouse and inventory layer is
simulated. Sales patterns, including holidays, come from the original Ecuadorian data.

## Tech stack

Python · pandas · scikit-learn · PyTorch · PyTorch Geometric Temporal · FastAPI · SQLite · React · Vite · TypeScript · Tailwind · shadcn/ui · Recharts · Leaflet
