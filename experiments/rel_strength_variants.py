"""Which relative-strength definition (if any) earns its block?

rs_N = roc_N - index_return is rank-identical to roc_N (the index return is
date-constant), so the old s_rel_strength block was a duplicate of momentum.
Candidates that actually re-rank across names:
  sector_rs_N : roc_N minus the stock's sector median roc_N (same date)
  rs_N        : kept as control -- identical ranks to roc_N by construction
Run at both horizons (20d = typical hold, 40d = default hold).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from stocklab import backtest, store
from stocklab.diagnostics import _rank_ic

STEP = 5

feat = store.load_table("features")
if str(feat["symbol"].dtype) == "category":
    feat = feat.assign(symbol=feat["symbol"].astype(str))

for n in (60, 120):
    med = feat.groupby(["date", "sector"])[f"roc{n}"].transform("median")
    feat[f"sector_rs_{n}"] = feat[f"roc{n}"] - med

prep = backtest.prepare(feat)
close, dates = prep["close"], prep["dates"]
tradable = prep["elig"]
half = len(dates) // 2

CANDIDATES = ["roc60", "sector_rs_60", "roc120", "sector_rs_120", "rs_120"]

for hz in (20, 40):
    fwd = np.full_like(close, np.nan)
    fwd[:-hz] = close[hz:] / close[:-hz] - 1.0
    print(f"\n=== horizon {hz} (mid {dates[half].date()}) ===")
    rows = []
    for name in CANDIDATES:
        mat = (
            feat[["date", "symbol", name]]
            .pivot(index="date", columns="symbol", values=name)
            .reindex(index=dates, columns=prep["syms"])
            .to_numpy(dtype=float)
        )
        mat = np.where(tradable, mat, np.nan)
        ics, ics_is, ics_oos = [], [], []
        for i in range(0, len(dates) - hz, STEP):
            ic = _rank_ic(mat[i], fwd[i])
            if not np.isfinite(ic):
                continue
            ics.append(ic)
            (ics_is if i < half else ics_oos).append(ic)

        def stat(v):
            a = np.asarray(v)
            if len(a) < 10:
                return np.nan, np.nan
            t = a.mean() / (a.std(ddof=1) / np.sqrt(len(a))) if a.std() > 0 else np.nan
            return float(a.mean()), float(t)

        m, t = stat(ics)
        m1, t1 = stat(ics_is)
        m2, t2 = stat(ics_oos)
        rows.append({
            "factor": name, "ic": round(m, 4), "t": round(t, 2),
            "h1": round(m1, 4), "t_h1": round(t1, 2),
            "h2": round(m2, 4), "t_h2": round(t2, 2),
            "stable": bool(np.sign(m1) == np.sign(m2)),
        })
    print(pd.DataFrame(rows).to_string(index=False))

# rank correlation between sector_rs variants and plain momentum (mid-sample)
sample = dates[half]
print(f"\nrank correlations on {sample.date()}:")
pairs = [("roc60", "sector_rs_60"), ("roc120", "sector_rs_120")]
for a, b in pairs:
    sa = feat[feat["date"] == sample].set_index("symbol")[a]
    sb = feat[feat["date"] == sample].set_index("symbol")[b]
    both = pd.concat([sa.rename(a), sb.rename(b)], axis=1).dropna()
    if len(both) > 20:
        rho = both[a].rank().corr(both[b].rank())
        print(f"  {a} vs {b}: rho={rho:.3f} over {len(both)} stocks")
