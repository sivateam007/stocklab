"""Bullish-technical + volume scan (daily chart only, no news/fundamentals).

Question: which stocks show a *positive technical structure with active buyers*
on the latest daily bar?

Data caveat that shaped this scan: the newest bar (2026-11-16) was captured
**mid-session** -- market-wide volume is 0.32x the 20-day average, so any
same-day `vol_ratio` is garbage. Volume gates therefore use the last five
*complete* sessions; price/trend gates use the latest bar.

Gates (all must hold):
  1. liquid universe (>= Rs5 Cr avg daily turnover)
  2. EMA stack aligned      EMA20 > EMA50 > EMA200
  3. above 200-DMA
  4. Supertrend bullish     st_dir == +1
  5. trending, not chop     ADX14 >= 20
  6. bullish but not blown  50 <= RSI14 <= 75
  7. accumulation           OBV 20d slope > 0
  8. volume behind the move last 5 complete sessions: avg volume >= 1.15x
                            the 20d average AND price higher over that window

Trigger (at least one): MACD histogram rising, or a new 20-day closing high,
or closing within 1% of the 20-day high.

Rank = volume expansion + trend quality + breakout proximity + momentum.

Run:  python experiments/bullish_scan.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from stocklab import store

MIN_ADX = 20.0
RSI_LO, RSI_HI = 50.0, 75.0
MIN_TURNOVER_CR = 5.0          # ignore illiquid names even if flagged liquid
VOL_EXPANSION = 1.15           # last 5 complete sessions vs 20d average


def _volume_context(feat: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Volume evidence from the last five *complete* sessions.

    Returns (per-symbol DataFrame, last complete session date).
    """
    dates = np.sort(feat["date"].unique())
    last = pd.Timestamp(dates[-1])
    # the newest bar is partial if its market-wide volume collapsed
    prior = feat[feat["date"] < last]
    if prior.empty:
        return pd.DataFrame(), last
    complete = np.sort(prior["date"].unique())[-5:]
    base = np.sort(prior["date"].unique())[-25:-5]      # 20 sessions before that

    v = feat.pivot_table(index="symbol", columns="date", values="volume")
    recent = v[[d for d in complete if d in v.columns]].mean(axis=1)
    baseline = v[[d for d in base if d in v.columns]].mean(axis=1)
    px = feat.pivot_table(index="symbol", columns="date", values="close")
    cols_c = [d for d in complete if d in px.columns]
    ret5 = px[cols_c[-1]] / px[[d for d in base if d in px.columns][-1]] - 1.0 \
        if cols_c and any(d in px.columns for d in base) else None

    out = pd.DataFrame({
        "vol5_ratio": recent / baseline.replace(0, np.nan),
        "ret5_pct": (ret5 * 100) if isinstance(ret5, pd.Series) else np.nan,
        "vol5_date": cols_c[-1] if cols_c else pd.NaT,
    })
    return out, pd.Timestamp(complete[-1]) if len(complete) else last


def scan() -> pd.DataFrame:
    f = store.load_table("features_latest")
    f = f[f["liquid"].fillna(False)].copy()

    for c in ("close", "ema_stack", "above_ema200", "st_dir", "adx14", "rsi14",
              "obv_slope", "macd_rising", "new_high20", "dist_high20",
              "dist_high52", "di_spread", "turnover_sma20", "roc20", "atr_pct",
              "s_technical"):
        if c in f.columns:
            f[c] = pd.to_numeric(f[c], errors="coerce")

    vol, vol_date = _volume_context(store.load_table(
        "features", columns=["date", "symbol", "volume", "close"]))
    f = f.merge(vol, on="symbol", how="left")

    gates = {
        "ema_stack": f["ema_stack"] == 1,
        "above_ema200": f["above_ema200"] == 1,
        "st_up": f["st_dir"] == 1,
        "trending": f["adx14"] >= MIN_ADX,
        "rsi_ok": f["rsi14"].between(RSI_LO, RSI_HI),
        "accumulation": f["obv_slope"] > 0,
        "liquid_enough": f["turnover_sma20"] >= MIN_TURNOVER_CR,
        "volume_expansion": f["vol5_ratio"] >= VOL_EXPANSION,
        "price_up_5d": f["ret5_pct"] > 0,
    }
    mask = pd.concat(gates.values(), axis=1).all(axis=1)
    out = f[mask].copy()

    trigger = (out["macd_rising"] == 1) | (out["new_high20"] == 1) \
        | (out["dist_high20"] >= -1.0)
    out = out[trigger].copy()
    if out.empty:
        return out

    # ---- ranking (all technical)
    def pct(s: pd.Series) -> pd.Series:
        return s.rank(pct=True)

    out["score"] = (
        0.30 * pct(out["vol5_ratio"])                    # buyer intensity, 5 sessions
        + 0.20 * pct(out["di_spread"])                    # +DI beating -DI
        + 0.15 * pct(out["adx14"])                        # trend strength
        + 0.15 * pct(out["dist_high20"].clip(-20, 0))     # breakout proximity
        + 0.10 * pct(out["obv_slope"])                    # 20d accumulation
        + 0.10 * pct(out["roc20"])                        # 1-month momentum
    ) * 100

    out["signals"] = out.apply(
        lambda r: ", ".join([x for x in (
            "MACD rising" if r["macd_rising"] == 1 else "",
            "20d high" if r["new_high20"] == 1 else "",
            "at 20d high" if r["dist_high20"] >= -1.0 and r["new_high20"] != 1 else "",
            "Supertrend up" if r["st_dir"] == 1 else "",
        ) if x]), axis=1)

    out.attrs["vol_date"] = vol_date
    cols = ["symbol", "sector", "close", "score", "vol5_ratio", "ret5_pct",
            "turnover_sma20", "adx14", "di_spread", "rsi14", "roc20",
            "dist_high20", "dist_high52", "atr_pct", "s_technical", "signals"]
    cols = [c for c in cols if c in out.columns]
    res = out[cols].sort_values("score", ascending=False)
    res.attrs["vol_date"] = vol_date
    return res


def main() -> None:
    res = scan()
    print("=" * 100)
    if res.empty:
        print("No stock passes every gate today.")
        return
    vd = res.attrs.get("vol_date")
    print(f"BULLISH TECHNICAL + VOLUME SCAN  ({len(res)} stocks pass all gates)")
    print(f"volume gates use the last 5 complete sessions through "
          f"{pd.Timestamp(vd).date() if vd is not None else '?'} "
          f"(latest bar is partial)")
    print("=" * 100)
    show = res.head(25).copy()
    for c in show.columns[2:]:
        if c != "signals":
            show[c] = pd.to_numeric(show[c], errors="coerce").round(2)
    print(show.to_string(index=False))

    print("\nVolume leaders (5-session buyer intensity):")
    v = res.sort_values("vol5_ratio", ascending=False).head(10).copy()
    for c in ("close", "vol5_ratio", "ret5_pct", "turnover_sma20", "adx14",
              "rsi14", "dist_high20"):
        v[c] = pd.to_numeric(v[c], errors="coerce").round(2)
    print(v[["symbol", "sector", "close", "vol5_ratio", "ret5_pct",
             "turnover_sma20", "adx14", "rsi14", "dist_high20", "signals"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
