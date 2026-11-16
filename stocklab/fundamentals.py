"""Fundamental data and sector-relative quality / growth / valuation scoring."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

from . import store

INFO_FIELDS = [
    "longName", "sector", "industry", "marketCap",
    "trailingPE", "forwardPE", "priceToBook", "trailingPegRatio", "pegRatio",
    "returnOnEquity", "returnOnAssets", "debtToEquity", "currentRatio",
    "profitMargins", "operatingMargins", "grossMargins",
    "revenueGrowth", "earningsGrowth", "earningsQuarterlyGrowth",
    "trailingEps", "forwardEps", "bookValue", "revenuePerShare",
    "freeCashflow", "operatingCashflow", "totalRevenue", "totalDebt", "totalCash",
    "ebitda", "enterpriseToEbitda", "enterpriseToRevenue",
    "heldPercentInsiders", "heldPercentInstitutions",
    "dividendYield", "beta", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
    "averageVolume", "recommendationKey", "numberOfAnalystOpinions", "targetMeanPrice",
]


def _pct_or_none(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return v


def _fetch_one(symbol: str) -> dict:
    row: dict = {"symbol": symbol}
    try:
        info = yf.Ticker(f"{symbol}.NS").info or {}
    except Exception as exc:  # noqa: BLE001
        return {"symbol": symbol, "error": f"{type(exc).__name__}"}
    for f in INFO_FIELDS:
        row[f] = info.get(f)
    return row


def fetch_fundamentals(symbols: list[str] | None = None, force: bool = False,
                       workers: int = 6, log=print) -> pd.DataFrame:
    symbols = symbols or [s for s in store.available_symbols() if not s.startswith("^")]
    cached = store.load_table("fundamentals") if not force else pd.DataFrame()

    have = set()
    frames = []
    if not cached.empty:
        frames.append(cached)
        have = set(cached["symbol"])

    todo = [s for s in symbols if s not in have]
    if not todo:
        log("Fundamentals already cached.")
        return cached

    log(f"Fetching fundamentals for {len(todo)} symbols...")
    rows: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, s): s for s in todo}
        for fut in as_completed(futures):
            rows.append(fut.result())
            done += 1
            if done % 50 == 0:
                log(f"  {done}/{len(todo)}")
                if rows:
                    part = pd.DataFrame(rows)
                    frames.append(part)
                    rows = []
            time.sleep(0.01)

    if rows:
        frames.append(pd.DataFrame(rows))

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return df
    df = df.drop_duplicates("symbol", keep="last")

    # ---- normalisation
    num_cols = [c for c in INFO_FIELDS if c not in ("longName", "sector", "industry", "recommendationKey")]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "debtToEquity" in df.columns:
        d2e = df["debtToEquity"].astype(float)
        # yfinance reports this as a percentage for Indian tickers
        df["debtToEquity"] = np.where(d2e.abs() > 5, d2e / 100.0, d2e)
    if "returnOnEquity" in df.columns:
        roe = df["returnOnEquity"].astype(float)
        df["returnOnEquity"] = np.where(roe.abs() <= 1.5, roe * 100, roe)
    if "returnOnAssets" in df.columns:
        roa = df["returnOnAssets"].astype(float)
        df["returnOnAssets"] = np.where(roa.abs() <= 1.5, roa * 100, roa)
    for c in ("profitMargins", "operatingMargins", "grossMargins", "revenueGrowth",
              "earningsGrowth", "earningsQuarterlyGrowth",
              "heldPercentInsiders", "heldPercentInstitutions"):
        if c in df.columns:
            df[c + "_pct"] = df[c].astype(float) * 100.0

    if {"freeCashflow", "marketCap"}.issubset(df.columns):
        df["fcf_yield"] = df["freeCashflow"] / df["marketCap"].replace(0, np.nan) * 100.0
    if {"trailingEps", "bookValue"}.issubset(df.columns):
        df["eps_book"] = df["trailingEps"] / df["bookValue"].replace(0, np.nan)
    if {"totalDebt", "ebitda"}.issubset(df.columns):
        df["debt_ebitda"] = df["totalDebt"] / df["ebitda"].replace(0, np.nan)

    df["fund_updated_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
    store.save_table("fundamentals", df)
    return df


# ------------------------------------------------------------------ scoring
def _inv_rank(s: pd.Series) -> pd.Series:
    """Lower is better -> higher score."""
    return (1.0 - s.rank(pct=True)) * 100.0


def _fwd_rank(s: pd.Series) -> pd.Series:
    """Higher is better -> higher score."""
    return s.rank(pct=True) * 100.0


def _avg(parts: list[tuple[pd.Series, float]]) -> pd.Series:
    num, den = 0.0, 0.0
    for s, w in parts:
        if s.isna().all():
            continue
        num = num + s.fillna(s.median()) * w
        den += w
    if den == 0:
        return pd.Series(np.nan, index=parts[0][0].index)
    return num / den


def score_fundamentals(fund: pd.DataFrame | None = None) -> pd.DataFrame:
    """Sector-relative fundamental scores, 0-100."""
    if fund is None:
        fund = store.load_table("fundamentals")
    if fund.empty:
        return fund

    df = fund.copy()
    # Prefer our own NSE-derived sector map so grouping is consistent
    # (yfinance uses a different taxonomy, e.g. "Financial Services").
    uni = store.load_table("universe")
    if not uni.empty and "symbol" in uni.columns:
        smap = dict(zip(uni["symbol"], uni["sector"]))
        df["_sector"] = df["symbol"].map(smap).fillna(df.get("sector", "Other"))
    else:
        df["_sector"] = df.get("sector", "Other")
    df["_sector"] = df["_sector"].fillna("Other")
    out = []
    for sector, grp in df.groupby("_sector", dropna=False):
        g = grp.copy()

        def col(name: str) -> pd.Series:
            if name in g.columns:
                return pd.to_numeric(g[name], errors="coerce")
            return pd.Series(np.nan, index=g.index)

        quality = _avg([
            (_fwd_rank(col("returnOnEquity")), 0.35),
            (_fwd_rank(col("operatingMargins_pct")), 0.25),
            (_inv_rank(col("debtToEquity")), 0.25),
            (_fwd_rank(col("currentRatio")), 0.15),
        ])
        growth = _avg([
            (_fwd_rank(col("revenueGrowth_pct")), 0.40),
            (_fwd_rank(col("earningsGrowth_pct")), 0.40),
            (_fwd_rank(col("earningsQuarterlyGrowth_pct")), 0.20),
        ])
        valuation = _avg([
            (_inv_rank(col("trailingPE")), 0.35),
            (_inv_rank(col("priceToBook")), 0.20),
            (_inv_rank(col("trailingPegRatio")), 0.25),
            (_fwd_rank(col("fcf_yield")), 0.20),
        ])
        g["s_quality"] = quality
        g["s_growth"] = growth
        g["s_valuation"] = valuation
        out.append(g)

    res = pd.concat(out, ignore_index=True)
    res["s_fundamental"] = (
        res["s_quality"] * 0.40 + res["s_growth"] * 0.35 + res["s_valuation"] * 0.25
    ).clip(0, 100)
    # Need at least a couple of real inputs, otherwise the score is noise.
    res["fund_ok"] = (
        res[["returnOnEquity", "revenueGrowth", "trailingPE"]].notna().sum(axis=1) >= 2
    )
    res["s_fundamental"] = res["s_fundamental"].where(res["fund_ok"])
    return res
