"""Parameter robustness study.

A high in-sample Sharpe means nothing. This runs every parameter set on the
first half of the history and again on the second half, then ranks by the
*worse* of the two. Only configurations that survive both halves are worth
considering.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from itertools import product

import pandas as pd

from . import backtest, store


def split_dates(feat: pd.DataFrame) -> tuple[str, str, str]:
    dates = sorted(pd.to_datetime(feat["date"].unique()))
    mid = dates[len(dates) // 2]
    return str(dates[0].date()), str(pd.Timestamp(mid).date()), str(dates[-1].date())


def param_study(grid: dict[str, list] | None = None, feat: pd.DataFrame | None = None,
                log=print) -> pd.DataFrame:
    if feat is None:
        feat = store.load_table("features")
    if feat.empty:
        raise RuntimeError("No features. Run ingest first.")

    grid = grid or {
        "top_n": [5, 10, 15, 20],
        "hold_days": [20, 40, 60],
        "atr_mult": [1.0, 1.5, 2.0],
        "trail": [False, True],
        "min_score": [55.0, 65.0],
    }
    d0, mid, d1 = split_dates(feat)
    backtest.prepare(feat)

    keys = list(grid)
    total = 1
    for k in keys:
        total *= len(grid[k])
    log(f"Studying {total} parameter sets across two halves...")

    rows = []
    for n, combo in enumerate(product(*[grid[k] for k in keys]), 1):
        kw = dict(zip(keys, combo))
        base = backtest.BTParams(**kw)
        try:
            r_is = backtest.run(replace(base, start=d0, end=mid), feat)
            r_oos = backtest.run(replace(base, start=mid, end=d1), feat)
        except Exception as exc:  # noqa: BLE001
            log(f"  {kw} failed: {type(exc).__name__}")
            continue
        rows.append({
            **kw,
            "is_cagr": r_is.get("cagr_pct"),
            "is_sharpe": r_is.get("sharpe"),
            "is_trades": r_is.get("n_trades"),
            "is_win": r_is.get("win_rate_pct"),
            "oos_cagr": r_oos.get("cagr_pct"),
            "oos_sharpe": r_oos.get("sharpe"),
            "oos_trades": r_oos.get("n_trades"),
            "oos_win": r_oos.get("win_rate_pct"),
            "oos_excess": r_oos.get("excess_return_pct"),
        })
        if n % 20 == 0:
            log(f"  {n}/{total}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["worst_sharpe"] = df[["is_sharpe", "oos_sharpe"]].min(axis=1)
    df["avg_sharpe"] = df[["is_sharpe", "oos_sharpe"]].mean(axis=1)
    df = df.sort_values("worst_sharpe", ascending=False).reset_index(drop=True)
    store.save_table("param_study", df)
    return df


def summarise(df: pd.DataFrame, top: int = 10) -> str:
    if df.empty:
        return "no results"
    cols = ["top_n", "hold_days", "atr_mult", "trail", "min_score",
            "is_cagr", "is_sharpe", "oos_cagr", "oos_sharpe", "worst_sharpe",
            "oos_trades", "oos_excess"]
    lines = [df[cols].head(top).to_string(index=False)]

    stable = df[(df["is_cagr"] > 0) & (df["oos_cagr"] > 0)]
    lines.append("")
    if stable.empty:
        lines.append("NOTHING IS STABLE: no parameter set is profitable in both halves.")
    else:
        lines.append(f"{len(stable)} of {len(df)} parameter sets are profitable in both halves.")
        best = stable.iloc[0]
        lines.append(
            "Best stable set: top_n=%s hold=%s atr=%s trail=%s min_score=%s "
            "-> IS CAGR %s%%, OOS CAGR %s%%" % (
                best["top_n"], best["hold_days"], best["atr_mult"], best["trail"],
                best["min_score"], best["is_cagr"], best["oos_cagr"],
            )
        )
    return "\n".join(lines)
