"""Download the Favorita dataset from Kaggle into data/raw/.

Before running:
  1. Join the competition and accept its rules:
     https://www.kaggle.com/competitions/store-sales-time-series-forecasting
  2. On kaggle.com go to Settings -> API -> Generate New Token, then save the token:
       mkdir -p ~/.kaggle && pbpaste > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
     (or set KAGGLE_API_TOKEN; a legacy ~/.kaggle/kaggle.json also works)

Usage:
  python scripts/download_data.py
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

COMPETITION = "store-sales-time-series-forecasting"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
EXPECTED = ["train.csv", "stores.csv", "holidays_events.csv", "oil.csv", "transactions.csv"]


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if all((RAW_DIR / f).exists() for f in EXPECTED):
        print(f"All files already present in {RAW_DIR}")
        return 0

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("The 'kaggle' package is missing. Run: pip install -r requirements.txt")
        return 1

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # noqa: BLE001 - show a clear message for any auth failure
        print(f"Kaggle authentication failed: {exc}")
        print("Generate a token at kaggle.com -> Settings -> API -> Generate New Token,")
        print("save it to ~/.kaggle/access_token, and make sure kaggle>=1.8 is installed.")
        return 1

    print(f"Downloading {COMPETITION} to {RAW_DIR} ...")
    try:
        api.competition_download_files(COMPETITION, path=str(RAW_DIR), quiet=False)
    except Exception as exc:  # noqa: BLE001
        print(f"Download failed: {exc}")
        print("Make sure you clicked 'Join Competition' and accepted the rules on Kaggle.")
        return 1

    for zip_path in RAW_DIR.glob("*.zip"):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW_DIR)
        zip_path.unlink()

    missing = [f for f in EXPECTED if not (RAW_DIR / f).exists()]
    if missing:
        print(f"Download finished but these files are missing: {missing}")
        return 1

    print("Done. Files:")
    for f in EXPECTED:
        size_mb = (RAW_DIR / f).stat().st_size / 1e6
        print(f"  {f:<22} {size_mb:8.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
