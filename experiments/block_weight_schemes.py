"""Choose technical block weights without in-sample cheating.

Schemes are derived from FIRST-HALF block ICs only (or fixed a priori), then
judged by second-half composite IC at both horizons. Full-sample-IC weights
are included only as an upper bound reference, not as the pick.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from stocklab import backtest, diagnostics, store

BLOCKS = ["s_trend", "s_momentum", "s_breakout", "s_volume", "s_rel_strength"]
STEP = 5

feat = store.load_table("features")
prep = backtest.prepare(feat)
close, dates = prep["close"], prep["dates"]
tradable = prep["elig"]
half = len(dates) // 2


def block_ics(name: str, hz: int) -> tuple[float, float, float]:
    fwd = np.full_like(close, np.nan)
    fwd[:-hz] = close[hz:] / close[:-hz] - 1.0
    mat = (
        feat[["date", "symbol", name]]
        .pivot(index="date", columns="symbol", values=name)
        .reindex(index=dates, columns=prep["syms"])
        .to_numpy(dtype=float)
    )
    mat = np.where(tradable, mat, np.nan)
    ics = []
    for i in range(0, len(dates) - hz, STEP):
        ic = diagnostics._rank_ic(mat[i], fwd[i])
        if np.isfinite(ic):
            ics.append((i, ic))
    h1 = np.mean([v for i, v in ics if i < half])
    h2 = np.mean([v for i, v in ics if i >= half])
    full = np.mean([v for _, v in ics])
    return float(full), float(h1), float(h2)


# first-half ICs per block per horizon -> derive weights on h1 only
h1_ic = {b: block_ics(b, 20)[1] + block_ics(b, 40)[1] for b in BLOCKS}
s = sum(abs(v) for v in h1_ic.values())
ic_prop_h1 = {b: 0.60 * abs(v) / s for b, v in h1_ic.items()}

full_ic = {b: block_ics(b, 20)[0] + block_ics(b, 40)[0] for b in BLOCKS}
s2 = sum(abs(v) for v in full_ic.values())
ic_prop_full = {b: 0.60 * abs(v) / s2 for b, v in full_ic.items()}

SCHEMES = {
    "current": {"s_trend": 0.13, "s_momentum": 0.15, "s_breakout": 0.18,
                "s_volume": 0.09, "s_rel_strength": 0.05},
    "h1-ic-proportional": ic_prop_h1,
    "full-ic-proportional (reference)": ic_prop_full,
    "equal": {b: 0.12 for b in BLOCKS},
    "tilt-to-volume": {"s_trend": 0.11, "s_momentum": 0.12, "s_breakout": 0.12,
                       "s_volume": 0.18, "s_rel_strength": 0.07},
}


def composite_ic(weights: dict[str, float], hz: int) -> tuple[float, float]:
    fwd = np.full_like(close, np.nan)
    fwd[:-hz] = close[hz:] / close[:-hz] - 1.0
    total = np.zeros_like(close)
    for b, w in weights.items():
        mat = (
            feat[["date", "symbol", b]]
            .pivot(index="date", columns="symbol", values=b)
            .reindex(index=dates, columns=prep["syms"])
            .to_numpy(dtype=float)
        )
        total += np.where(tradable, mat, np.nan) * w
    ics_h1, ics_h2 = [], []
    for i in range(0, len(dates) - hz, STEP):
        row = total[i].copy()
        row[~tradable[i]] = np.nan
        ic = diagnostics._rank_ic(row, fwd[i])
        if not np.isfinite(ic):
            continue
        (ics_h1 if i < half else ics_h2).append(ic)
    return float(np.mean(ics_h1)), float(np.mean(ics_h2))


print("first-half-derived weights:", {k: round(v, 3) for k, v in ic_prop_h1.items()})
print()
for hz in (20, 40):
    print(f"=== horizon {hz} (judge on second half) ===")
    rows = []
    for name, w in SCHEMES.items():
        h1, h2 = composite_ic(w, hz)
        rows.append({"scheme": name, "ic_h1": round(h1, 4), "ic_h2": round(h2, 4)})
    print(pd.DataFrame(rows).to_string(index=False))
    print()
