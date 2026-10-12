"""Train the ST-GNN, compare it with the baseline on several holdouts, pick the model for recommendations.

Run the baseline pipeline first; this script adds to its results:

  python -m ml.pipeline              # baseline: writes data/processed/results/
  python -m ml.models.stgnn.train    # ST-GNN: about 15 minutes on a laptop CPU

Steps:
  1. Out-of-sample baseline forecasts: the baseline is refit at each holdout and (in residual
     mode) every 8 weeks before them, so the ST-GNN never learns from a baseline forecast that
     had already seen the answer. Cached in baseline_oof.npz, so re-runs skip this step.
  2. Pick the number of epochs by early stopping on the 8 weekly windows before the first holdout.
  3. Rolling evaluation: for each of `holdouts` back-to-back 28-day holdouts, retrain on every
     window before it, forecast it, and score ST-GNN, baseline, 50/50 blend and naive.
  4. Final model on all data, forecasting the next 28 days.
  5. Blend with the baseline (ml.models.ensemble) and re-run the optimizer with whichever model
     has the lowest error pooled over all holdouts.

Residual mode (default): the ST-GNN receives the baseline forecast and starts out as an exact
copy of it, learning only corrections. `--standalone` trains it without the baseline input.

Adds to results/: forecasts_stgnn.csv, forecasts_ensemble.csv, metrics_rolling.csv, `stgnn` and
`ensemble` columns/rows in backtest.csv and metrics.csv (latest holdout), stgnn_training.json,
baseline_oof.npz and stgnn.pt (both gitignored data).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ml.data.load import load_dataset
from ml.models.baseline import KEYS, Z80, error_metrics, series_sigma
from ml.models.ensemble import blend
from ml.models.stgnn.data import HORIZON, LOOKBACK, STData, build, split_origins, windows
from ml.models.stgnn.evaluate import (BaseForecasts, actual_cube, holdout_origins, naive_cube, oof_start, score,
                                      valid_mask)
from ml.models.stgnn.model import STGNN

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT / "data" / "processed"
REFIT_EVERY = 56  # days between baseline refits for the out-of-sample forecasts


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def to_tensors(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)).to(device) for k, v in batch.items()}


class Trainer:
    def __init__(self, data: STData, scale: np.ndarray, base: BaseForecasts | None, device: torch.device,
                 hidden: int, lr: float, batch_size: int, seed: int):
        torch.manual_seed(seed)
        self.data, self.scale, self.base, self.device, self.batch_size = data, scale, base, device, batch_size
        self.rng = np.random.default_rng(seed)
        n, p = data.mask.shape
        self.model = STGNN(n, p, hidden=hidden, residual=base is not None).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.adj = torch.from_numpy(data.adjacency).to(device)
        # Loss weight per series ~ its size, so the loss behaves like a Poisson deviance on real units.
        w = data.mask * scale
        self.weight = torch.from_numpy((w / w[data.mask > 0].mean()).astype(np.float32)).to(device)
        self.scale_t = torch.from_numpy(scale).to(device)

    def _batch(self, origins: np.ndarray) -> dict[str, torch.Tensor]:
        b = windows(self.data, origins, self.scale)
        if self.base is not None:
            b["f_base"] = self.base.get(origins) / self.scale[None, None]
        return to_tensors(b, self.device)

    def _forward(self, b: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(b["x_sales"], b["x_cal"], b["f_cal"], b["f_ly"], b["f_ly_ok"], self.adj, b.get("f_base"))

    def predict(self, origins: np.ndarray) -> np.ndarray:
        """Forecasts in real units, [B, HORIZON, N, P]."""
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(origins), self.batch_size):
                b = self._batch(origins[i:i + self.batch_size])
                out.append((self._forward(b) * self.scale_t).cpu().numpy())
        return np.concatenate(out)

    def wape(self, origins: np.ndarray) -> float:
        pred = self.predict(origins)
        actual = np.stack([actual_cube(self.data, o) for o in origins])
        valid = np.stack([valid_mask(self.data, o) for o in origins])
        return score(pred, actual, valid)["wape"]

    def epoch(self, origins: np.ndarray) -> float:
        self.model.train()
        order = self.rng.permutation(origins)
        total, count = 0.0, 0
        for i in range(0, len(order), self.batch_size):
            b = self._batch(order[i:i + self.batch_size])
            pred = self._forward(b)
            open_days = (1 - b["f_cal"][..., 8])[:, :, None, None]
            w = self.weight[None, None] * open_days
            loss = (w * (pred - b["y"]) ** 2).sum() / w.sum().clamp(min=1e-6)
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            total += loss.item() * len(b["y"])
            count += len(b["y"])
        return total / count

    def fit(self, origins: np.ndarray, epochs: int) -> None:
        for _ in range(epochs):
            self.epoch(origins)


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
          model_for_recommendations: str = "auto", holdouts: int = 4, residual: bool = True,
          verbose: bool = True) -> dict:
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
    holdout_o = holdout_origins(last, holdouts)
    start = oof_start(last) if residual else LOOKBACK - 1
    log(f"{len(data.stores)} stores x {len(data.products)} product families, {last + 1} days, device {dev}, "
        f"ST-GNN {'residual on the baseline' if residual else 'standalone'}, {holdouts} holdout(s)")

    # ---- 1. Out-of-sample baseline forecasts ----
    log("Baseline forecasts (cached after the first run) ...")
    bounds = holdout_o + [last] + (list(range(start, holdout_o[0], REFIT_EVERY)) if residual else [])
    oof = BaseForecasts.build(ds, data, bounds, cache=out_dir / "baseline_oof.npz", seed=seed, log=log)
    base = oof if residual else None  # standalone mode only uses it for scoring

    # ---- 2. Early stopping before the first holdout ----
    train_o, val_o = split_origins(holdout_o[0], start)
    log(f"Choosing epochs: {len(train_o)} training windows, {len(val_o)} validation windows")
    trainer = Trainer(data, data.scale(holdout_o[0]), base, dev, hidden, lr, batch_size, seed)
    history, best, best_epoch, stale = [], np.inf, 1, 0
    if residual:
        start_wape = trainer.wape(val_o)  # an untrained residual model is exactly the baseline
        history.append({"epoch": 0, "train_loss": None, "val_wape": round(start_wape, 5)})
        log(f"  epoch  0  (= baseline)    validation WAPE {start_wape:.1%}")
    for ep in range(1, epochs + 1):
        loss = trainer.epoch(train_o)
        val = trainer.wape(val_o)
        history.append({"epoch": ep, "train_loss": round(loss, 5), "val_wape": round(val, 5)})
        log(f"  epoch {ep:>2}  loss {loss:.4f}  validation WAPE {val:.1%}")
        if val < best - 1e-4:
            best, best_epoch, stale = val, ep, 0
        else:
            stale += 1
            if stale >= patience:
                log(f"  stopping: no improvement for {patience} epochs (best epoch {best_epoch})")
                break

    # ---- 3. Rolling evaluation over the holdouts ----
    rows, pooled = [], {m: ([], []) for m in ("stgnn", "baseline", "ensemble", "naive")}
    latest_pred = None
    for i, h in enumerate(holdout_o):
        log(f"Holdout {i + 1}/{holdouts}: {data.dates[h + 1]:%Y-%m-%d} to {data.dates[h + HORIZON]:%Y-%m-%d}, "
            f"training {best_epoch} epochs ...")
        t = Trainer(data, data.scale(h), base, dev, hidden, lr, batch_size, seed)
        t.fit(np.arange(start, h - HORIZON + 1), best_epoch)
        stgnn = t.predict(np.array([h]))[0]
        baseline = oof.get(np.array([h]))[0]
        preds = {"stgnn": stgnn, "baseline": baseline, "ensemble": 0.5 * (stgnn + baseline),
                 "naive": naive_cube(ds, data, h)}
        actual, valid = actual_cube(data, h), valid_mask(data, h)
        for m, pr in preds.items():
            rows.append({"holdout_start": f"{data.dates[h + 1]:%Y-%m-%d}", "model": m, **score(pr, actual, valid)})
            pooled[m][0].append(pr[valid])
            pooled[m][1].append(actual[valid])
        log("  WAPE  " + "   ".join(f"{r['model']} {r['wape']:.1%}" for r in rows[-4:]))
        latest_pred = stgnn

    for m, (p_list, a_list) in pooled.items():
        p_all, a_all = np.concatenate(p_list), np.concatenate(a_list)
        rows.append({"holdout_start": "ALL", "model": m, **score(p_all, a_all, np.ones(len(p_all), dtype=bool))})
    rolling = pd.DataFrame(rows)
    rolling.to_csv(out_dir / "metrics_rolling.csv", index=False)
    overall = rolling[rolling["holdout_start"] == "ALL"].set_index("model")["wape"]
    log("Pooled over all holdouts  " + "   ".join(f"{m} {w:.1%}" for m, w in overall.items()))

    # Latest holdout -> backtest.csv / metrics.csv, the per-store files the app reads.
    cutoff = holdout_o[-1]
    holdout = to_frame(data, latest_pred, cutoff, "stgnn")
    bt = pd.read_csv(out_dir / "backtest.csv", parse_dates=["date"]).drop(columns=["stgnn", "ensemble"], errors="ignore")
    bt = bt.merge(holdout, on=KEYS + ["date"], how="left")
    closed_dates = set(data.dates[data.calendar[:, 8] == 1])
    scored = bt[~bt["date"].isin(closed_dates)]
    metrics = pd.read_csv(out_dir / "metrics.csv")
    metrics = metrics[~metrics["model"].isin(["stgnn", "ensemble"])]
    new = [{"model": "stgnn", "location_id": "ALL", **error_metrics(scored["actual"], scored["stgnn"])}]
    new += [{"model": "stgnn", "location_id": loc, **error_metrics(g["actual"], g["stgnn"])}
            for loc, g in scored.groupby("location_id")]
    metrics = pd.concat([metrics, pd.DataFrame(new)], ignore_index=True)

    # ---- 4. Final model on all data ----
    final_o = np.arange(start, last - HORIZON + 1)
    log(f"Final training on {len(final_o)} windows for {best_epoch} epochs ...")
    final = Trainer(data, data.scale(last), base, dev, hidden, lr, batch_size, seed)
    final.fit(final_o, best_epoch)
    fc = to_frame(data, final.predict(np.array([last]))[0], last, "p50")
    fc = fc.merge(series_sigma(scored.dropna(subset=["stgnn"]), "stgnn"), on=KEYS, how="left")
    fc["sigma"] = fc["sigma"].fillna(fc["p50"] * 0.5)
    fc["p10"] = np.clip(fc["p50"] - Z80 * fc["sigma"], 0, None)
    fc["p90"] = fc["p50"] + Z80 * fc["sigma"]
    fc["model"] = "stgnn"
    fc = fc[KEYS + ["date", "model", "p50", "p10", "p90", "sigma"]].round({"p50": 3, "p10": 3, "p90": 3, "sigma": 3})

    fc.assign(date=fc["date"].dt.strftime("%Y-%m-%d")).to_csv(out_dir / "forecasts_stgnn.csv", index=False)
    bt.assign(date=bt["date"].dt.strftime("%Y-%m-%d")).round(3).to_csv(out_dir / "backtest.csv", index=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    torch.save(final.model.state_dict(), out_dir / "stgnn.pt")

    report = {
        "settings": {"mode": "residual" if residual else "standalone", "lookback": LOOKBACK, "horizon": HORIZON,
                     "hidden": hidden, "lr": lr, "batch_size": batch_size, "seed": seed,
                     "max_epochs": epochs, "patience": patience, "holdouts": holdouts},
        "best_epoch": best_epoch,
        "history": history,
        "rolling_wape": {m: float(w) for m, w in overall.items()},
        "train_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "stgnn_training.json").write_text(json.dumps(report, indent=2))

    # ---- 5. Blend, choose, optimize ----
    report.update(blend(data_dir, model_for_recommendations, verbose=verbose))
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
    parser.add_argument("--holdouts", type=int, default=4, help="back-to-back 28-day holdouts to score on")
    parser.add_argument("--standalone", action="store_true", help="train without the baseline input")
    parser.add_argument("--model-for-recommendations", choices=["auto", "baseline", "stgnn", "ensemble"],
                        default="auto", help="auto = lowest error pooled over the holdouts")
    args = parser.parse_args()
    report = train(args.data, args.epochs, args.patience, args.hidden, args.lr, args.batch_size, args.seed,
                   args.device, args.model_for_recommendations, args.holdouts, not args.standalone)
    print(json.dumps({k: report[k] for k in ("best_epoch", "rolling_wape", "holdout_wape", "best_model",
                                             "recommendation_model", "train_seconds")}, indent=2))


if __name__ == "__main__":
    main()
