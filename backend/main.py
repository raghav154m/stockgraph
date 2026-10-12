"""FastAPI app.

Run locally:
  python -m backend.loader                  # once, after the ML pipeline (or let startup do it)
  uvicorn backend.main:app --reload         # http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import router
from backend.config import Settings
from backend.db import init_db, make_engine
from backend.loader import ResultsMissing, load_dataset_into_db
from backend.models import Dataset

log = logging.getLogger("stockgraph")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = make_engine(settings.db_url)
        app.state.engine = engine
        app.state.session_factory = init_db(engine)
        if settings.autoload_demo:
            with app.state.session_factory() as db:
                loaded = db.get(Dataset, "demo") is not None
            if not loaded:
                try:
                    counts = load_dataset_into_db(engine, Path(settings.data_dir), "demo")
                    log.info("Loaded demo dataset: %s", counts)
                except (ResultsMissing, FileNotFoundError) as exc:
                    log.warning("Demo dataset not loaded: %s", exc)
        yield
        engine.dispose()

    app = FastAPI(title="StockGraph API", version="0.1.0", lifespan=lifespan,
                  description="Demand forecasts, stock status and reorder/transfer recommendations "
                              "for a network of stores and warehouses.")
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"],
                       allow_headers=["*"])
    app.include_router(router)
    return app


app = create_app()
