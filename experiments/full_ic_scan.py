"""Scan IC for every numeric feature column, 20d and 40d horizons.

The hand-picked KEY_FACTORS list may miss stable signals. This sweeps all
numeric columns and flags those whose IC sign survives both halves of history
(the rework rule: only stable-sign factors may vote in a score block).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from stocklab import diagnostics, store

EXCLUDE = {
    "date", "symbol", "sector", "liquid",
    # level/size variables whose cross-sectional rank is not a tradeable signal
    "close", "volume", "sma50", "sma200", "atr14", "macd_hist",
    # block scores rebuilt separately
    "s_trend", "s_momentum", "s_breakout", "s_volume", "s_rel_strength",
    "s_technical",
}


def classify(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    h1, h2 = out["ic_first_half"], out["ic_second_half"]
    stable = out["ic"].notna() & h1.notna() & h2.notna() & (np.sign(h1) == np.sign(h2))
    out["stable_sign"] = stable
    out["votes"] = np.where(
        ~stable, "NO (sign flipped)",
        np.where(out["ic"] > 0, "yes (+)", "yes (inverted)"),
    )
    return out


def main() -> None:
    feat = store.load_table("features")
    num = [c for c in feat.columns
           if c not in EXCLUDE and pd.api.types.is_numeric_dtype(feat[c])]
    print(f"scanning {len(num)} numeric columns")

    for hz in (20, 40):
        print(f"\n=== horizon {hz} ===")
        raw = diagnostics.factor_ic(horizon=hz, step=5, feat=feat, factors=num,
                                    liquid_only=True, log=lambda s: None)
        tab = classify(raw)
        stable = tab[tab["stable_sign"]].sort_values("ic", key=abs, ascending=False)
        print(stable[["factor", "ic", "t_stat", "ic_first_half", "ic_second_half",
                      "votes"]].to_string(index=False))
        tab.to_parquet(f"data/full_ic_h{hz}.parquet", index=False)


if __name__ == "__main__":
    main()
