"""Train the ST-GNN, compare it with the baseline, and (if it wins) use it for recommendations.

Run the baseline pipeline first; this script adds to its results:

  python -m ml.pipeline              # baseline: writes data/processed/results/
  python -m ml.models.stgnn.train    # ST-GNN: trains on CPU in a few minutes

Steps:
  1. Backtest: train on everything up to 28 days before the last date (early stopping on the
     8 most recent weekly windows), forecast the held-out 28 days, score it like the baseline.
  2. Final model: retrain on all data for the same number of epochs, forecast the next 28 days.
  3. If its holdout WAPE beats the baseline, re-run the optimizer with ST-GNN forecasts.

Adds to results/: forecasts_stgnn.csv, a `stgnn` column in backtest.csv, `stgnn` rows in
metrics.csv, stgnn_training.json (loss curve, settings), stgnn.pt (weights, gitignored).
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ml.data.load import load_dataset
from ml.models.baseline import KEYS, Z80, error_metrics, series_sigma
from ml.models.stgnn.data import HORIZON, LOOKBACK, STData, build, split_origins, windows
from ml.models.stgnn.model import STGNN

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT / "data" / "processed"


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def to_tensors(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)).to(device) for k, v in batch.items()}


class Trainer:
    def __init__(self, data: STData, scale: np.ndarray, device: torch.device, hidden: int, lr: float,
                 batch_size: int, seed: int):
        torch.manual_seed(seed)
        self.data, self.scale, self.device, self.batch_size = data, scale, device, batch_size
        self.rng = np.random.default_rng(seed)
        n, p = data.mask.shape
        self.model = STGNN(n, p, hidden=hidden).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.adj = torch.from_numpy(data.adjacency).to(device)
        # Loss weight per series ~ its size, so the loss behaves like a Poisson deviance on real units.
        w = data.mask * scale
        self.weight = torch.from_numpy((w / w[data.mask > 0].mean()).astype(np.float32)).to(device)
        self.mask = torch.from_numpy(data.mask).to(device)
        self.scale_t = torch.from_numpy(scale).to(device)

    def predict(self, origins: np.ndarray) -> np.ndarray:
        """Forecasts in real units, [B, HORIZON, N, P]."""
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(origins), self.batch_size):
                b = to_tensors(windows(self.data, origins[i:i + self.batch_size], self.scale), self.device)
                pred = self.model(b["x_sales"], b["x_cal"], b["f_cal"], self.adj) * self.scale_t
                out.append(pred.cpu().numpy())
        return np.concatenate(out)

    def wape(self, origins: np.ndarray) -> float:
        pred = self.predict(origins)
        b = windows(self.data, origins, self.scale)
        actual = b["y"] * self.scale
        open_days = (b["f_cal"][..., 8] == 0)[:, :, None, None]
        m = (self.data.mask[None, None] > 0) & open_days
        return float(np.abs(pred - actual)[np.broadcast_to(m, pred.shape)].sum()
                     / actual[np.broadcast_to(m, actual.shape)].sum())

    def epoch(self, origins: np.ndarray) -> float:
        self.model.train()
        order = self.rng.permutation(origins)
        total, count = 0.0, 0
        for i in range(0, len(order), self.batch_size):
            b = to_tensors(windows(self.data, order[i:i + self.batch_size], self.scale), self.device)
            pred = self.model(b["x_sales"], b["x_cal"], b["f_cal"], self.adj)
            open_days = (1 - b["f_cal"][..., 8])[:, :, None, None]
            w = self.weight[None, None] * open_days
            loss = (w * (pred - b["y"]) ** 2).sum() / w.sum().clamp(min=1e-6)
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            total += float(loss) * len(b["y"])
            count += len(b["y"])
        return total / count


def to_frame(data: STData, pred: np.ndarray, origin: int, column: str) -> pd.DataFrame:
    """[HORIZON, N, P] forecast from `origin` -> long rows for series that exist."""
    s, p = np.nonzero(data.mask)
    dates = data.dates[origin + 1: origin + 1 + HORIZON]
    return pd.DataFrame({
        "location_id": np.repeat(np.array(data.stores)[s], HORIZON),
        "product_id": np.repeat(np.array(data.products)[p], HORIZON),
        "date": np.tile(dates, len(s)),
        column: pred[:, s, p].T.ravel(),
    })


def train(data_dir: Path = DEFAULT_DATA, epochs: int = 40, patience: int = 6, hidden: int = 64,
          lr: float = 1e-3, batch_size: int = 32, seed: int = 42, device: str = "auto",
          use_for_recommendations: str = "auto", verbose: bool = True) -> dict:
    t0 = time.time()
    out_dir = data_dir / "results"
    if not (out_dir / "backtest.csv").exists():
        raise FileNotFoundError(f"{out_dir / 'backtest.csv'} not found. Run `python -m ml.pipeline` first.")

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    dev = pick_device(device)
    ds = load_dataset(data_dir)
    data = build(ds, data_dir / "graph")
    last = data.index_of(ds.as_of)
    cutoff = last - HORIZON
    log(f"{len(data.stores)} stores x {len(data.products)} product families, {last + 1} days, device {dev}")

    # ---- 1. Backtest model with early stopping ----
    scale = data.scale(cutoff)
    train_o, val_o = split_origins(cutoff)
    log(f"Backtest training: {len(train_o)} windows, validating on {len(val_o)}")
    trainer = Trainer(data, scale, dev, hidden, lr, batch_size, seed)
    history, best, best_state, best_epoch, stale = [], np.inf, None, 0, 0
    for ep in range(1, epochs + 1):
        loss = trainer.epoch(train_o)
        val = trainer.wape(val_o)
        history.append({"epoch": ep, "train_loss": round(loss, 5), "val_wape": round(val, 5)})
        log(f"  epoch {ep:>2}  loss {loss:.4f}  validation WAPE {val:.1%}")
        if val < best - 1e-4:
            best, best_state, best_epoch, stale = val, copy.deepcopy(trainer.model.state_dict()), ep, 0
        else:
            stale += 1
            if stale >= patience:
                log(f"  stopping: no improvement for {patience} epochs (best epoch {best_epoch})")
                break
    trainer.model.load_state_dict(best_state)

    holdout = to_frame(data, trainer.predict(np.array([cutoff]))[0], cutoff, "stgnn")
    bt = pd.read_csv(out_dir / "backtest.csv", parse_dates=["date"]).drop(columns="stgnn", errors="ignore")
    bt = bt.merge(holdout, on=KEYS + ["date"], how="left")
    closed_dates = set(data.dates[data.calendar[:, 8] == 1])
    scored = bt[~bt["date"].isin(closed_dates)]

    metrics = pd.read_csv(out_dir / "metrics.csv")
    metrics = metrics[metrics["model"] != "stgnn"]
    rows = [{"model": "stgnn", "location_id": "ALL", **error_metrics(scored["actual"], scored["stgnn"])}]
    rows += [{"model": "stgnn", "location_id": loc, **error_metrics(g["actual"], g["stgnn"])}
             for loc, g in scored.groupby("location_id")]
    metrics = pd.concat([metrics, pd.DataFrame(rows)], ignore_index=True)
    wape = metrics[metrics["location_id"] == "ALL"].set_index("model")["wape"]
    log(f"Holdout WAPE  stgnn {wape['stgnn']:.1%}   baseline {wape['baseline']:.1%}   naive {wape['naive']:.1%}")

    # ---- 2. Final model on all data for the chosen number of epochs ----
    scale_f = data.scale(last)
    all_o = np.arange(LOOKBACK - 1, last - HORIZON + 1)
    log(f"Final training on {len(all_o)} windows for {best_epoch} epochs ...")
    final = Trainer(data, scale_f, dev, hidden, lr, batch_size, seed)
    for _ in range(best_epoch):
        final.epoch(all_o)
    fc = to_frame(data, final.predict(np.array([last]))[0], last, "p50")
    fc = fc.merge(series_sigma(bt.dropna(subset=["stgnn"]), "stgnn"), on=KEYS, how="left")
    fc["sigma"] = fc["sigma"].fillna(fc["p50"] * 0.5)
    fc["p10"] = np.clip(fc["p50"] - Z80 * fc["sigma"], 0, None)
    fc["p90"] = fc["p50"] + Z80 * fc["sigma"]
    fc["model"] = "stgnn"
    fc = fc[KEYS + ["date", "model", "p50", "p10", "p90", "sigma"]].round({"p50": 3, "p10": 3, "p90": 3, "sigma": 3})

    # ---- write ----
    fc.assign(date=fc["date"].dt.strftime("%Y-%m-%d")).to_csv(out_dir / "forecasts_stgnn.csv", index=False)
    bt.assign(date=bt["date"].dt.strftime("%Y-%m-%d")).round(3).to_csv(out_dir / "backtest.csv", index=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    torch.save(final.model.state_dict(), out_dir / "stgnn.pt")

    wins = bool(wape["stgnn"] < wape["baseline"])
    use = use_for_recommendations == "always" or (use_for_recommendations == "auto" and wins)
    report = {
        "settings": {"lookback": LOOKBACK, "horizon": HORIZON, "hidden": hidden, "lr": lr,
                     "batch_size": batch_size, "seed": seed, "max_epochs": epochs, "patience": patience},
        "best_epoch": best_epoch,
        "history": history,
        "holdout_wape": {m: float(w) for m, w in wape.items()},
        "beats_baseline": wins,
        "used_for_recommendations": use,
        "train_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "stgnn_training.json").write_text(json.dumps(report, indent=2))

    if use:
        from ml.pipeline import optimize  # local import: pipeline doesn't need torch
        log("Re-running the inventory optimizer with ST-GNN forecasts ...")
        summary = optimize(ds, fc, out_dir, metrics, "stgnn")
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    else:
        log("Keeping baseline forecasts for recommendations (ST-GNN did not beat it).")
    log(f"Done in {time.time() - t0:.0f}s. Results in {out_dir}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    parser.add_argument("--use-for-recommendations", choices=["auto", "always", "never"], default="auto")
    args = parser.parse_args()
    report = train(args.data, args.epochs, args.patience, args.hidden, args.lr, args.batch_size,
                   args.seed, args.device, args.use_for_recommendations)
    print(json.dumps({k: report[k] for k in ("best_epoch", "holdout_wape", "beats_baseline",
                                             "used_for_recommendations", "train_seconds")}, indent=2))


if __name__ == "__main__":
    main()
