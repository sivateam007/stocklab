"""Factor diagnostics: does each signal actually predict forward returns?

The information coefficient (IC) is the cross-sectional rank correlation between
a factor and the next N days' return. A factor with a mean IC near zero has no
edge, no matter how sensible it sounds. Reporting IC per half tells you whether
a signal decayed or never worked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import backtest, store

KEY_FACTORS = [
    # block scores
    "s_trend", "s_momentum", "s_breakout", "s_volume", "s_rel_strength",
    "s_technical",
    # trend components
    "ema_stack", "above_ema200", "sma50_slope", "st_dir", "dist_ema20_atr",
    # momentum components
    "rsi_band", "rsi14", "macd_rising", "roc20", "roc60", "roc120",
    "adx14", "plus_di", "minus_di", "di_spread",
    # breakout components
    "dist_high20", "dist_high52", "new_high20",
    # volume components
    "vol_ratio", "vol_score_raw", "turnover_sma20", "obv_slope",
    "vol_price_confirm",
    # relative strength
    "rs_20", "rs_60", "rs_120",
    # volatility
    "atr_pct", "bb_width_pctile",
]


def _rank_ic(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 30:
        return np.nan
    x = pd.Series(a[mask]).rank().to_numpy()
    y = pd.Series(b[mask]).rank().to_numpy()
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def factor_ic(horizon: int = 20, step: int = 5, feat: pd.DataFrame | None = None,
              factors: list[str] | None = None, liquid_only: bool = True,
              log=print) -> pd.DataFrame:
    prep = backtest.prepare(feat)
    close, dates = prep["close"], prep["dates"]
    if feat is None:
        feat = store.load_table("features")
    if str(feat["symbol"].dtype) == "category":
        feat = feat.assign(symbol=feat["symbol"].astype(str))

    fwd = np.full_like(close, np.nan)
    fwd[:-horizon] = close[horizon:] / close[:-horizon] - 1.0

    # IC is only meaningful on names the strategy can actually buy.
    tradable = prep["elig"] if liquid_only else np.ones_like(prep["elig"])

    factors = factors or [f for f in KEY_FACTORS if f in feat.columns]
    half = len(dates) // 2
    rows = []

    for name in factors:
        mat = (
            feat[["date", "symbol", name]]
            .pivot(index="date", columns="symbol", values=name)
            .reindex(index=dates, columns=prep["syms"])
            .to_numpy(dtype=float)
        )
        mat = np.where(tradable, mat, np.nan)
        ics_full, ics_is, ics_oos = [], [], []
        for i in range(0, len(dates) - horizon, step):
            ic = _rank_ic(mat[i], fwd[i])
            if not np.isfinite(ic):
                continue
            ics_full.append(ic)
            (ics_is if i < half else ics_oos).append(ic)

        def summarise(v: list[float]) -> tuple[float, float]:
            if len(v) < 10:
                return np.nan, np.nan
            arr = np.asarray(v)
            t = arr.mean() / (arr.std(ddof=1) / np.sqrt(len(arr))) if arr.std() > 0 else np.nan
            return float(arr.mean()), float(t)

        m, t = summarise(ics_full)
        m_is, t_is = summarise(ics_is)
        m_oos, t_oos = summarise(ics_oos)
        rows.append({
            "factor": name,
            "ic": round(m, 4) if np.isfinite(m) else None,
            "t_stat": round(t, 2) if np.isfinite(t) else None,
            "ic_first_half": round(m_is, 4) if np.isfinite(m_is) else None,
            "t_first_half": round(t_is, 2) if np.isfinite(t_is) else None,
            "ic_second_half": round(m_oos, 4) if np.isfinite(m_oos) else None,
            "t_second_half": round(t_oos, 2) if np.isfinite(t_oos) else None,
            "n_obs": len(ics_full),
        })
        log(f"  {name}: IC {m:+.4f} (t {t:+.2f})")

    out = pd.DataFrame(rows).sort_values("ic", ascending=False).reset_index(drop=True)
    store.save_table("factor_ic", out)
    return out


def summarise(df: pd.DataFrame, horizon: int = 20) -> str:
    if df.empty:
        return "no results"
    lines = [f"Forward horizon: {horizon} sessions. |t| > 2 is roughly significant.\n"+ df.to_string(index=False)]
    strong = df[(df["t_stat"].abs() > 2) & (df["ic"].abs() > 0.02)]
    lines.append("")
    if strong.empty:
        lines.append("No factor is reliably predictive across the full sample.")
    else:
        lines.append("Predictive factors: " + ", ".join(
            f"{r.factor} ({r.ic:+.3f})" for r in strong.itertuples()
        ))

    flipped = df[(df["ic_first_half"].notna()) & (df["ic_second_half"].notna())
                 & (np.sign(df["ic_first_half"]) != np.sign(df["ic_second_half"]))
                 & (df["ic_first_half"].abs() > 0.02)]
    if not flipped.empty:
        lines.append("Factors whose sign flipped between halves (unstable): "
                     + ", ".join(flipped["factor"].tolist()))
    return "\n".join(lines)
