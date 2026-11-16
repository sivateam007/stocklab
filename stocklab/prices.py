"""Daily OHLCV ingest via Yahoo Finance, with incremental updates and caching."""

from __future__ import annotations

import time
import warnings
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from . import config as C
from . import store

warnings.filterwarnings("ignore")

STD_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise columns/index and drop obviously broken rows."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=STD_COLS)

    if isinstance(df.columns, pd.MultiIndex):
        if df.columns.nlevels == 2:
            df.columns = df.columns.get_level_values(-1)
        else:
            df.columns = df.columns.get_level_values(0)

    for col in STD_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[STD_COLS].copy()

    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = pd.DatetimeIndex(idx).normalize()
    df.index.name = "date"

    for col in STD_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[~df.index.duplicated(keep="last")].sort_index()
    # A usable bar needs a close and volume; missing Adj Close falls back to Close.
    df = df.dropna(subset=["Close"])
    df["Volume"] = df["Volume"].fillna(0.0)
    df["Adj Close"] = df["Adj Close"].fillna(df["Close"])
    bad = df[["Open", "High", "Low"]].isna().any(axis=1)
    if bad.any():
        df.loc[bad, "Open"] = df.loc[bad, "Close"]
        df.loc[bad, "High"] = df.loc[bad, ["High", "Close"]].max(axis=1)
        df.loc[bad, "Low"] = df.loc[bad, ["Low", "Close"]].min(axis=1)
    # Guard against zero/negative prices which poison every indicator.
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    return df


def _download_one(yahoo: str, period: str | None, start: str | None) -> pd.DataFrame:
    try:
        t = yf.Ticker(yahoo)
        if start:
            raw = t.history(start=start, interval="1d", auto_adjust=False, actions=False)
        else:
            raw = t.history(period=period or C.PRICE_PERIOD, interval="1d",
                            auto_adjust=False, actions=False)
        return _clean(raw)
    except Exception:
        return pd.DataFrame(columns=STD_COLS)


def _download_batch(yahoos: list[str], period: str | None, start: str | None) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if not yahoos:
        return out
    try:
        if start:
            raw = yf.download(
                yahoos, start=start, interval="1d", group_by="ticker",
                auto_adjust=False, actions=False, threads=True, progress=False,
            )
        else:
            raw = yf.download(
                yahoos, period=period or C.PRICE_PERIOD, interval="1d", group_by="ticker",
                auto_adjust=False, actions=False, threads=True, progress=False,
            )
    except Exception:
        raw = pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        for y in yahoos:
            if y in raw.columns.get_level_values(0):
                out[y] = _clean(raw[y])
    elif len(yahoos) == 1 and not raw.empty:
        out[yahoos[0]] = _clean(raw)
    return out


def update_symbols(symbols: list[str], force: bool = False, log=print) -> dict:
    """Download or incrementally refresh daily bars for `symbols`."""
    yahoo_map = {s: (s if s.startswith("^") else f"{s}.NS") for s in symbols}
    results = {"ok": 0, "failed": [], "rows": 0}

    pending: dict[str, str | None] = {}
    for sym in symbols:
        existing = store.load_prices(sym)
        if force or existing.empty:
            pending[sym] = None
        elif (date.today() - existing.index.max().date()).days >= 1:
            pending[sym] = (existing.index.max().date() - timedelta(days=7)).isoformat()
        else:
            continue

    if not pending:
        log("Nothing to update - all series current.")
        return results

    log(f"Updating {len(pending)} symbols...")
    items = list(pending.items())
    for i in range(0, len(items), C.BATCH_SIZE):
        chunk = items[i:i + C.BATCH_SIZE]
        yahoos = [yahoo_map[s] for s, _ in chunk]
        starts = {s: st for s, st in chunk}
        distinct_starts = {st for st in starts.values()}
        if len(distinct_starts) > 1:
            groups: dict[str | None, list[str]] = {}
            for s, st in chunk:
                groups.setdefault(st, []).append(s)
        else:
            groups = {chunk[0][1]: [s for s, _ in chunk]}

        for start_val, group_syms in groups.items():
            got = _download_batch([yahoo_map[s] for s in group_syms], C.PRICE_PERIOD, start_val)
            for sym in group_syms:
                y = yahoo_map[sym]
                new = got.get(y, pd.DataFrame())
                tries = 0
                while (new.empty or len(new) < 2) and tries < C.RETRIES:
                    time.sleep(1.0 + tries)
                    new = _download_one(y, C.PRICE_PERIOD, start_val)
                    tries += 1
                if new.empty:
                    store.set_state(sym, None, 0, "no data returned")
                    results["failed"].append(sym)
                    continue
                if start_val:
                    old = store.load_prices(sym)
                    if not old.empty:
                        merged = pd.concat([old, new])
                        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                        new = merged
                store.save_prices(sym, new)
                results["ok"] += 1
                results["rows"] += len(new)
        log(f"  {min(i + C.BATCH_SIZE, len(items))}/{len(items)} symbols")
        time.sleep(0.6)

    return results


def load_panel(symbols: list[str] | None = None, start: str | None = None) -> pd.DataFrame:
    """Long-format panel: date, symbol, open, high, low, close, volume, adj_close."""
    symbols = symbols or store.available_symbols()
    frames = []
    for sym in symbols:
        df = store.load_prices(sym)
        if df.empty:
            continue
        d = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
        })
        d["symbol"] = sym
        frames.append(d.reset_index())
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    if start:
        panel = panel[panel["date"] >= pd.Timestamp(start)]
    return panel.sort_values(["symbol", "date"]).reset_index(drop=True)


def build_wide(symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Wide frames keyed by field: open/high/low/close/volume, columns=symbols."""
    panel = load_panel(symbols)
    if panel.empty:
        return {}
    out = {}
    for field, src in [("open", "open"), ("high", "high"), ("low", "low"),
                       ("close", "close"), ("volume", "volume"), ("adj_close", "adj_close")]:
        out[field] = panel.pivot(index="date", columns="symbol", values=src)
    return out


def adjusted(df: pd.DataFrame) -> pd.DataFrame:
    """Scale OHLC by the split/bonus adjustment factor, keep raw volume."""
    if df.empty:
        return df
    factor = (df["Adj Close"] / df["Close"]).replace([float("inf"), -float("inf")], 1.0).fillna(1.0)
    out = df.copy()
    for col in ("Open", "High", "Low", "Close"):
        out[col] = out[col] * factor
    out = out.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume", "Adj Close": "adj_close",
    })
    return out[["open", "high", "low", "close", "volume", "adj_close"]]
