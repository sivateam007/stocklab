"""Results (earnings) engine: dates, estimates, surprises and post-result drift.

Post-earnings drift is one of the few short-horizon anomalies that survives
out-of-sample: stocks that gap up on a positive surprise tend to keep drifting
for two to six weeks. That window is exactly the STCG horizon, so it is worth
scoring explicitly.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

from . import store


def _ticker(symbol: str) -> yf.Ticker:
    return yf.Ticker(f"{symbol}.NS")


def _calendar_one(symbol: str) -> dict:
    out = {"symbol": symbol}
    try:
        cal = _ticker(symbol).calendar or {}
    except Exception as exc:  # noqa: BLE001
        out["error"] = type(exc).__name__
        return out

    def first(key):
        v = cal.get(key)
        if isinstance(v, (list, tuple)):
            return v[0] if v else None
        return v

    ed = first("Earnings Date")
    out["earnings_date"] = pd.to_datetime(ed, errors="coerce") if ed else pd.NaT
    out["eps_est"] = first("Earnings Average")
    out["eps_low"] = first("Earnings Low")
    out["eps_high"] = first("Earnings High")
    out["rev_est"] = first("Revenue Average")
    out["ex_div_date"] = pd.to_datetime(first("Ex-Dividend Date"), errors="coerce")
    return out


def _surprises_one(symbol: str) -> list[dict]:
    try:
        df = _ticker(symbol).earnings_dates
    except Exception:
        return []
    if df is None or len(df) == 0:
        return []
    df = df.reset_index()
    date_col = df.columns[0]
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "symbol": symbol,
            "report_date": pd.to_datetime(r[date_col], errors="coerce"),
            "eps_actual": r.get("Reported EPS"),
            "eps_est": r.get("EPS Estimate"),
            "surprise_pct": r.get("Surprise(%)"),
        })
    return rows


def fetch_calendar(symbols: list[str] | None = None, workers: int = 6,
                   log=print) -> pd.DataFrame:
    symbols = symbols or [s for s in store.available_symbols() if not s.startswith("^")]
    cached = store.load_table("earnings_calendar")
    have = set(cached["symbol"]) if not cached.empty else set()
    todo = [s for s in symbols if s not in have]
    if not todo:
        log("Earnings calendar already cached.")
        return cached

    log(f"Fetching earnings calendar for {len(todo)} symbols...")
    rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_calendar_one, s) for s in todo]
        for fut in as_completed(futures):
            rows.append(fut.result())
            done += 1
            if done % 100 == 0:
                log(f"  {done}/{len(todo)}")
    new = pd.DataFrame(rows)
    out = pd.concat([cached, new], ignore_index=True) if not cached.empty else new
    out = out.drop_duplicates("symbol", keep="last")
    store.save_table("earnings_calendar", out)
    return out


def fetch_surprises(symbols: list[str], log=print) -> pd.DataFrame:
    cached = store.load_table("earnings_surprises")
    have = set(cached["symbol"]) if not cached.empty else set()
    todo = [s for s in symbols if s not in have]
    if not todo:
        return cached
    log(f"Fetching past surprises for {len(todo)} symbols...")
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for fut in as_completed([pool.submit(_surprises_one, s) for s in todo]):
            rows.extend(fut.result())
    new = pd.DataFrame(rows)
    if new.empty:
        return cached
    out = pd.concat([cached, new], ignore_index=True) if not cached.empty else new
    out = out.dropna(subset=["report_date"]).drop_duplicates(["symbol", "report_date"], keep="last")
    store.save_table("earnings_surprises", out)
    return out


def results_state(symbols: list[str] | None = None, asof: date | None = None,
                  stale_after: int = 60) -> pd.DataFrame:
    """Per-symbol results flags used by the scorer.

    Columns: symbol, days_to_result, results_watch, surprise_pct,
             days_since_result, reaction_pct, drift_score, s_results

    Data reality check (this matters for trusting the output):
    - The earnings *calendar* (Yahoo `Ticker.calendar`) is fresh and gives one
      date per symbol.  If it is <= asof it is the **last** report; if it is
      after asof it is the **upcoming** report (days_to_result, results_watch,
      the 0-5 day screen penalty).
    - Yahoo's EPS *surprise history* (yfinance `earnings_dates` ->
      `/v1/finance/visualization`) is stale for `.NS` tickers: the newest row
      can be >1 year old, so it can no longer anchor "days since result".
      `surprise_pct` is therefore only reported when it is genuinely fresh
      (report_date within `stale_after` days); otherwise NaN rather than a
      stale number pretending to be current.
    - The durable signal is the **price reaction** around the last report
      (last close on/before the report -> first close after it, adjusted),
      computed from our own price store for any symbol with a known past
      report date.  s_results blends reaction and (only-if-fresh) surprise,
      decayed by age, and stays NaN when neither can be measured yet.
    """
    asof = pd.Timestamp(asof or date.today()).normalize()
    cal = store.load_table("earnings_calendar")
    sur = store.load_table("earnings_surprises")
    if symbols is not None:
        if not cal.empty:
            cal = cal[cal["symbol"].isin(symbols)]
        if not sur.empty:
            sur = sur[sur["symbol"].isin(symbols)]

    cols = ["symbol", "days_to_result", "results_watch", "surprise_pct",
            "days_since_result", "reaction_pct", "drift_score", "s_results"]
    base = pd.DataFrame({"symbol": symbols}) if symbols is not None else pd.DataFrame()
    if not cal.empty:
        c = cal[["symbol", "earnings_date"]].copy()
        c["earnings_date"] = pd.to_datetime(c["earnings_date"], errors="coerce")
        c["days_to_result"] = (c["earnings_date"] - asof).dt.days
        base = c[["symbol", "earnings_date", "days_to_result"]]
    if base.empty or "days_to_result" not in base:
        return pd.DataFrame(columns=cols)

    base["results_watch"] = base["days_to_result"].between(0, 7)

    # ---- latest surprise row per symbol (kept only to date/blend, may be stale)
    last_sur = pd.DataFrame(columns=["symbol", "report_date", "surprise_pct"])
    if not sur.empty:
        s0 = sur.copy()
        s0["report_date"] = pd.to_datetime(s0["report_date"], errors="coerce",
                                           utc=True).dt.tz_localize(None)
        s0 = s0.dropna(subset=["report_date"])
        s0 = s0[s0["report_date"] <= asof]
        if not s0.empty:
            last_sur = s0.sort_values(["symbol", "report_date"]).groupby("symbol").tail(1)
    sur_map = last_sur.set_index("symbol") if len(last_sur) else None

    # ---- last report date: a past calendar date beats stale surprise history
    last_dates = []
    for _, r in base.iterrows():
        d = None
        if pd.notna(r.get("earnings_date")) and r["earnings_date"] <= asof:
            d = r["earnings_date"]
        elif sur_map is not None and r["symbol"] in sur_map.index:
            sd = sur_map.loc[r["symbol"], "report_date"]
            if pd.notna(sd):
                d = pd.Timestamp(sd)
        last_dates.append(d)
    base["last_report_date"] = last_dates
    base["days_since_result"] = [
        (asof - d).days if d is not None else np.nan for d in last_dates]

    # ---- surprise_pct: only when genuinely fresh (never ship a stale number)
    def _fresh_surprise(sym):
        if sur_map is None or sym not in sur_map.index:
            return np.nan
        row = sur_map.loc[sym]
        sp = row.get("surprise_pct", np.nan)
        rd = row.get("report_date", pd.NaT)
        if pd.isna(sp) or pd.isna(rd):
            return np.nan
        if (asof - pd.Timestamp(rd)).days > stale_after:
            return np.nan
        return float(sp)

    base["surprise_pct"] = [_fresh_surprise(s) for s in base["symbol"]]

    # ---- price reaction on the first session after the result
    #      (only worth loading prices for reports young enough to score)
    reactions = {}
    young = base[base["days_since_result"].notna()
                 & (base["days_since_result"] <= stale_after)]
    for _, r in young.iterrows():
        px = store.load_prices(r["symbol"])
        if px.empty:
            continue
        adj = px["Adj Close"] / px["Close"]
        close = (px["Close"] * adj).dropna()
        rd = r["last_report_date"]
        after = close[close.index > rd]
        before = close[close.index <= rd]
        if len(after) >= 1 and len(before) >= 1:
            reactions[r["symbol"]] = (after.iloc[0] / before.iloc[-1] - 1.0) * 100.0
    base["reaction_pct"] = base["symbol"].map(reactions)

    # ---- drift score: reaction + (only-if-fresh) surprise, decayed by age
    sur_pct = pd.to_numeric(base["surprise_pct"], errors="coerce")
    react = pd.to_numeric(base["reaction_pct"], errors="coerce")
    age = pd.to_numeric(base["days_since_result"], errors="coerce")
    decay = np.clip(1.0 - age / 45.0, 0.0, 1.0)

    s_surprise = np.clip(sur_pct / 10.0, -1.0, 1.0)
    s_react = np.clip(react / 5.0, -1.0, 1.0)
    # Renormalise over the components we actually have: with no fresh surprise
    # the reaction carries the whole signal instead of being 45% of nothing.
    num = 0.55 * s_surprise.fillna(0) * sur_pct.notna() \
        + 0.45 * s_react.fillna(0) * react.notna()
    den = 0.55 * sur_pct.notna() + 0.45 * react.notna()
    raw = (num / den.replace(0, np.nan)) * decay
    base["drift_score"] = raw.round(3)
    base["s_results"] = ((raw + 1.0) * 50.0).clip(0, 100).round(2)
    base.loc[den == 0, "s_results"] = np.nan   # nothing measurable -> no vote
    base.loc[age > stale_after, "s_results"] = np.nan
    return base[cols]
