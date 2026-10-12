"""Settings, read from environment variables so the same code runs locally and on Render/Railway.

  STOCKGRAPH_DB_URL         SQLAlchemy URL (default: sqlite:///data/stockgraph.db)
  STOCKGRAPH_DATA_DIR       demo dataset folder with results/ (default: data/processed)
  STOCKGRAPH_AUTOLOAD_DEMO  load the demo into the database on startup if it isn't there (default: 1)
  STOCKGRAPH_CORS_ORIGINS   comma-separated frontend origins (default: the Vite dev server)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _origins() -> list[str]:
    raw = os.getenv("STOCKGRAPH_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    return [o.strip() for o in raw.split(",") if o.strip()]


@dataclass
class Settings:
    db_url: str = field(default_factory=lambda: os.getenv(
        "STOCKGRAPH_DB_URL", f"sqlite:///{ROOT / 'data' / 'stockgraph.db'}"))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv(
        "STOCKGRAPH_DATA_DIR", str(ROOT / "data" / "processed"))))
    autoload_demo: bool = field(default_factory=lambda: os.getenv("STOCKGRAPH_AUTOLOAD_DEMO", "1") == "1")
    cors_origins: list[str] = field(default_factory=_origins)
