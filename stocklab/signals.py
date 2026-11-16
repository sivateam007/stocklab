"""Turn a ranked list into concrete trade plans: entry, stop, target, size."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import prices, store


def plan_trade(row: pd.Series, capital: float = 100_000.0,
               risk_pct: float | None = None) -> dict:
    """ATR stop with a structural floor.

    The stop must clear *both* normal noise (1.5 x ATR) and the recent swing low,
    so the risk distance is the larger of the two. Using the tighter of the two
    stops out momentum names on the first pullback.
    """
    risk_pct = C.RULES.risk_per_trade_pct if risk_pct is None else risk_pct
    entry = float(row["close"])
    atr = float(row.get("atr14") or np.nan)
    if not np.isfinite(atr) or atr <= 0:
        atr = entry * 0.02

    atr_risk = C.RULES.atr_stop_mult * atr
    swing = float(row.get("swing_low10") or np.nan)
    swing_risk = entry - swing if np.isfinite(swing) and swing < entry else 0.0
    risk_per_share = max(atr_risk, swing_risk)
    stop = entry - risk_per_share
    target = entry + C.RULES.target_r_multiple * risk_per_share

    risk_amount = capital * risk_pct / 100.0
    qty = int(risk_amount // risk_per_share) if risk_per_share > 0 else 0
    # Never commit more than 25% of capital to a single ATR stop idea.
    if qty * entry > capital * 0.25:
        qty = int((capital * 0.25) // entry)

    return {
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "risk_per_share": round(risk_per_share, 2),
        "rr": round(C.RULES.target_r_multiple, 2),
        "qty": qty,
        "capital_at_risk": round(qty * risk_per_share, 2),
        "position_value": round(qty * entry, 2),
        "stop_pct": round((stop / entry - 1.0) * 100.0, 2),
        "target_pct": round((target / entry - 1.0) * 100.0, 2),
    }


def attach_plans(df: pd.DataFrame, capital: float = 100_000.0, top: int | None = None) -> pd.DataFrame:
    if df.empty:
        return df
    work = df[df.get("eligible", True)].copy()
    if top:
        work = work.head(top)
    plans = work.apply(lambda r: pd.Series(plan_trade(r, capital)), axis=1)
    return pd.concat([work.reset_index(drop=True), plans], axis=1)


def position_summary(plans: pd.DataFrame) -> dict:
    if plans.empty:
        return {"positions": 0, "capital_deployed": 0.0, "max_loss": 0.0}
    return {
        "positions": int(len(plans)),
        "capital_deployed": float(plans["position_value"].sum()),
        "max_loss": float(plans["capital_at_risk"].sum()),
        "avg_stop_pct": float(plans["stop_pct"].mean()),
        "avg_target_pct": float(plans["target_pct"].mean()),
    }
