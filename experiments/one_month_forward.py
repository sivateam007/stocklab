"""Conditional forward-return study for a ~1-month (20 session) hold.

Question: historically, when the screener's technical composite put a stock in
the top N on a given day, what did the NEXT 20 sessions actually do?

This is descriptive (no costs, no stops), same caveat family as the IC study:
survivorship-biased universe, point-in-time sector map is fine but the
features themselves are from the current price store.  Use it as a sense of
base rates, not as a promise.

Run:  python experiments/one_month_forward.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from stocklab import config as C, store

HOLD = 20          # sessions ~ one month
STEP = 5           # sample every 5th session (limits overlap)
START = "2023-01-01"
TOP_NS = (5, 10)


def load_panel(symbols: list[str]) -> pd.DataFrame:
    cols = {}
    for sym in symbols:
        px = store.load_prices(sym)
        if px.empty or "Adj Close" not in px.columns:
            continue
        s = pd.to_numeric(px["Adj Close"], errors="coerce")
        s.index = pd.to_datetime(s.index)
        cols[sym] = s
    return pd.DataFrame(cols).sort_index()


def main() -> None:
    feat = store.load_table("features")
    feat = feat[["date", "symbol", "s_technical", "liquid"]].copy()
    feat["date"] = pd.to_datetime(feat["date"])
    feat = feat[feat["date"] >= START]
    feat = feat[feat["liquid"].fillna(False) & feat["s_technical"].notna()]
    if feat.empty:
        print("no features")
        return

    syms = sorted(feat["symbol"].unique())
    bench = None
    bpx = store.load_prices(C.BENCHMARK)
    if not bpx.empty and "Adj Close" in bpx.columns:
        bench = pd.to_numeric(bpx["Adj Close"], errors="coerce")
        bench.index = pd.to_datetime(bench.index)
        bench = bench.sort_index()
    px = load_panel(syms)
    px = px.dropna(axis=1, how="all")
    feat = feat[feat["symbol"].isin(px.columns)]

    dates = np.sort(feat["date"].unique())
    sample = dates[::STEP]

    # forward returns on the price panel
    fwd = px.shift(-HOLD) / px - 1.0
    b_fwd = (bench.shift(-HOLD) / bench - 1.0) if bench is not None else None

    rows = []
    for d in sample:
        day = feat[feat["date"] == d].nlargest(max(TOP_NS), "s_technical")
        fr = fwd.loc[d] if d in fwd.index else None
        if fr is None:
            continue
        br = float(b_fwd.loc[d]) if b_fwd is not None and d in b_fwd.index else np.nan
        for n in TOP_NS:
            picks = day.head(n)
            if picks.empty:
                continue
            rets = fr.reindex(picks["symbol"].values).dropna()
            if rets.empty:
                continue
            rows.append({
                "date": d, "top": n, "picks": len(rets),
                "mean": rets.mean() * 100, "med": rets.median() * 100,
                "win": (rets > 0).mean() * 100,
                "bench": br * 100 if not np.isnan(br) else np.nan,
                "hit_any": (rets > br).mean() * 100 if not np.isnan(br) else np.nan,
            })
    r = pd.DataFrame(rows)
    if r.empty:
        print("no samples")
        return

    mid = pd.Timestamp("2024-05-21")
    print(f"Conditional forward {HOLD}d returns (top-N by s_technical, "
          f"every {STEP}th session, from {START})")
    print("=" * 78)
    for n in TOP_NS:
        sub = r[r["top"] == n]
        excess = sub["mean"] - sub["bench"]
        print(f"\nTOP {n}  ({len(sub)} rebalances, {int(sub['picks'].sum())} picks)")
        print(f"  avg pick  : {sub['mean'].mean():+.2f}%   median {sub['med'].mean():+.2f}%"
              f"   win rate {sub['win'].mean():.0f}%")
        print(f"  Nifty     : {sub['bench'].mean():+.2f}%")
        print(f"  excess    : {excess.mean():+.2f}%   pick beats index "
              f"{sub['hit_any'].mean():.0f}% of the time")
        for label, mask in (("H1 (to 2024-05-21)", sub["date"] < mid),
                            ("H2 (after)", sub["date"] >= mid)):
            s2 = sub[mask]
            if s2.empty:
                continue
            print(f"  {label:<20} avg {s2['mean'].mean():+.2f}%  "
                  f"bench {s2['bench'].mean():+.2f}%  win {s2['win'].mean():.0f}%")
        tail = sub.tail(12)
        print("  last 12 rebalances: "
              + " ".join(f"{x['date'].strftime('%y-%m-%d')}:{x['mean']:+.1f}"
                         for _, x in tail.iterrows()))


if __name__ == "__main__":
    main()
