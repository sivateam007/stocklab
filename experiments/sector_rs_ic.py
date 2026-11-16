"""One-off: is sector-relative 60d momentum actually predictive?

The old s_rel_strength used rs_N = roc_N - index_return. Because the index
return is constant across a date, cross-sectional ranks of rs_N are identical
to ranks of roc_N -- the block added no information beyond price momentum.
This tests a real relative-strength measure: roc60 minus the stock's sector
median roc60 on the same date, which does re-rank across sectors.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from stocklab import backtest, store
from stocklab.diagnostics import _rank_ic

HORIZON = 20
STEP = 5

feat = store.load_table("features")
if str(feat["symbol"].dtype) == "category":
    feat = feat.assign(symbol=feat["symbol"].astype(str))

# sector median momentum per date
feat["sector_med_roc60"] = feat.groupby(["date", "sector"])["roc60"].transform("median")
feat["sector_rs_60"] = feat["roc60"] - feat["sector_med_roc60"]
# control: plain roc60 for direct comparison
# (already in KEY_FACTORS, re-run here so both see the identical universe/mask)

prep = backtest.prepare(feat)
close, dates = prep["close"], prep["dates"]
tradable = prep["elig"]

fwd = np.full_like(close, np.nan)
fwd[:-HORIZON] = close[HORIZON:] / close[:-HORIZON] - 1.0
half = len(dates) // 2

print(f"dates={len(dates)} horizon={HORIZON} mid={dates[half].date()}")
print(f"tradable share={tradable.mean():.1%}\n")

rows = []
for name in ("sector_rs_60", "roc60", "rs_60"):
    mat = (
        feat[["date", "symbol", name]]
        .pivot(index="date", columns="symbol", values=name)
        .reindex(index=dates, columns=prep["syms"])
        .to_numpy(dtype=float)
    )
    mat = np.where(tradable, mat, np.nan)
    ics, ics_is, ics_oos = [], [], []
    for i in range(0, len(dates) - HORIZON, STEP):
        ic = _rank_ic(mat[i], fwd[i])
        if not np.isfinite(ic):
            continue
        ics.append(ic)
        (ics_is if i < half else ics_oos).append(ic)

    def stat(v):
        a = np.asarray(v)
        if len(a) < 10:
            return np.nan, np.nan, len(a)
        t = a.mean() / (a.std(ddof=1) / np.sqrt(len(a))) if a.std() > 0 else np.nan
        return float(a.mean()), float(t), len(a)

    m, t, n = stat(ics)
    m1, t1, _ = stat(ics_is)
    m2, t2, _ = stat(ics_oos)
    rows.append({
        "factor": name, "ic": round(m, 4), "t": round(t, 2), "n": n,
        "ic_h1": round(m1, 4), "t_h1": round(t1, 2),
        "ic_h2": round(m2, 4), "t_h2": round(t2, 2),
        "sign_stable": bool(np.sign(m1) == np.sign(m2) and m1 != 0 and m2 != 0),
    })
    print(f"  {name}: IC {m:+.4f} (t {t:+.2f})  h1 {m1:+.4f} (t {t1:+.2f})"
          f"  h2 {m2:+.4f} (t {t2:+.2f})  n={n}")

print()
print(pd.DataFrame(rows).to_string(index=False))

# How different are the two rankings really? (Spearman of cross-sectional ranks)
sample = dates[len(dates) // 2]
a = feat[feat["date"] == sample].set_index("symbol")["roc60"]
b = feat[feat["date"] == sample].set_index("symbol")["sector_rs_60"]
both = pd.concat([a.rename("roc60"), b.rename("sector_rs")], axis=1).dropna()
if len(both) > 20:
    rho = both["roc60"].rank().corr(both["sector_rs"].rank())
    print(f"\nRank correlation roc60 vs sector_rs_60 on {sample.date()}: "
          f"{rho:.3f} over {len(both)} stocks")
