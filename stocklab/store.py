"""Parquet/sqlite persistence helpers."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import config as C

DB_PATH = C.DATA_DIR / "stocklab.db"


# ------------------------------------------------------------------ sqlite
_SCHEMA_READY = False
_SCHEMA_SQL = [
    """
    CREATE TABLE IF NOT EXISTS runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job TEXT NOT NULL,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        status TEXT NOT NULL DEFAULT 'running',
        detail TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ingest_state (
        symbol TEXT PRIMARY KEY,
        last_date TEXT,
        rows INTEGER,
        updated_at TEXT,
        error TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS kv (
        key TEXT PRIMARY KEY,
        value TEXT,
        updated_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sentiment (
        headline_hash TEXT PRIMARY KEY,
        headline TEXT,
        score REAL,
        model TEXT,
        updated_at TEXT
    )
    """,
]


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    global _SCHEMA_READY
    if not _SCHEMA_READY:
        _create_schema(conn)
        _SCHEMA_READY = True
    return conn


def _create_schema(conn: sqlite3.Connection) -> None:
    for sql in _SCHEMA_SQL:
        conn.execute(sql)
    conn.commit()


def init_db() -> None:
    _conn().close()


def start_run(job: str) -> int:
    init_db()
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO runs (job, started_at) VALUES (?, ?)",
            (job, datetime.now().isoformat(timespec="seconds")),
        )
        return int(cur.lastrowid)


def finish_run(run_id: int, status: str, detail: str = "") -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE runs SET finished_at=?, status=?, detail=? WHERE id=?",
            (datetime.now().isoformat(timespec="seconds"), status, detail, run_id),
        )


def recent_runs(limit: int = 20) -> pd.DataFrame:
    init_db()
    with _conn() as conn:
        return pd.read_sql_query(
            "SELECT * FROM runs ORDER BY id DESC LIMIT ?", conn, params=(limit,)
        )


def set_state(symbol: str, last_date, rows: int, error: str = "") -> None:
    with _conn() as conn:
        conn.execute(
            """
            INSERT INTO ingest_state (symbol, last_date, rows, updated_at, error)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                last_date=excluded.last_date,
                rows=excluded.rows,
                updated_at=excluded.updated_at,
                error=excluded.error
            """,
            (
                symbol,
                str(last_date) if last_date is not None else None,
                rows,
                datetime.now().isoformat(timespec="seconds"),
                error,
            ),
        )


def get_state(symbol: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT symbol, last_date, rows, updated_at, error FROM ingest_state WHERE symbol=?",
            (symbol,),
        ).fetchone()
    if not row:
        return None
    return dict(zip(["symbol", "last_date", "rows", "updated_at", "error"], row))


def kv_set(key: str, value) -> None:
    with _conn() as conn:
        conn.execute(
            """
            INSERT INTO kv (key, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """,
            (key, json.dumps(value), datetime.now().isoformat(timespec="seconds")),
        )


def kv_get(key: str, default=None):
    with _conn() as conn:
        row = conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
    if not row:
        return default
    try:
        return json.loads(row[0])
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------------ parquet
def price_path(symbol: str) -> Path:
    safe = symbol.replace("^", "IDX_").replace("/", "_").replace("&", "_")
    return C.PRICE_DIR / f"{safe}.parquet"


def save_prices(symbol: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    df.to_parquet(price_path(symbol), compression="snappy")
    set_state(symbol, df.index.max().date() if hasattr(df.index.max(), "date") else df.index.max(), len(df))


def load_prices(symbol: str) -> pd.DataFrame:
    p = price_path(symbol)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def available_symbols() -> list[str]:
    out = []
    for p in C.PRICE_DIR.glob("*.parquet"):
        name = p.stem
        if name.startswith("IDX_"):
            out.append("^" + name[4:])
        else:
            out.append(name)
    return sorted(out)


def table(name: str) -> Path:
    return C.DATA_DIR / f"{name}.parquet"


def save_table(name: str, df: pd.DataFrame) -> None:
    df.to_parquet(table(name), compression="snappy")


def load_table(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    p = table(name)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p, columns=columns)


# ------------------------------------------------------------------ sentiment cache
def hash_headline(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.strip().lower().encode("utf-8", "ignore")).hexdigest()


def sentiment_lookup(texts: list[str]) -> dict[str, float]:
    """Return cached scores keyed by headline hash."""
    if not texts:
        return {}
    hashes = [hash_headline(t) for t in texts]
    out: dict[str, float] = {}
    with _conn() as conn:
        for chunk_start in range(0, len(hashes), 500):
            chunk = hashes[chunk_start:chunk_start + 500]
            marks = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT headline_hash, score FROM sentiment WHERE headline_hash IN ({marks})",
                chunk,
            ).fetchall()
            out.update({r[0]: r[1] for r in rows})
    return out


def sentiment_store(pairs: list[tuple[str, float]], model: str) -> None:
    if not pairs:
        return
    now = datetime.now().isoformat(timespec="seconds")
    rows = [(hash_headline(t), t[:500], float(s), model, now) for t, s in pairs]
    with _conn() as conn:
        conn.executemany(
            """
            INSERT INTO sentiment (headline_hash, headline, score, model, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(headline_hash) DO UPDATE SET score=excluded.score, updated_at=excluded.updated_at
            """,
            rows,
        )

