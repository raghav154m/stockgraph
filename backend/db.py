"""Engine and session setup."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from backend.models import Base


def make_engine(db_url: str) -> Engine:
    if db_url.startswith("sqlite:///"):
        Path(db_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(db_url, connect_args={"check_same_thread": False})

        @event.listens_for(engine, "connect")
        def _pragmas(conn, _record) -> None:  # faster bulk loads, safe for a read-mostly app
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()

        return engine
    return create_engine(db_url)


def init_db(engine: Engine) -> sessionmaker[Session]:
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    with factory() as session:
        yield session
