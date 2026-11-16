"""Cost- and tax-aware backtester with walk-forward validation.

Rules enforced to avoid self-deception:
  * signal is computed on the close of day T, entry fills at the open of T+1
  * if a bar touches both stop and target, the stop is assumed (conservative)
  * a gap through the stop fills at the open, not at the stop price
  * every statutory charge, STT, stamp duty, DP fee, slippage and 20% STCG is paid
  * STCG is netted per financial year with losses carried forward
  * entries are blocked when the market regime filter says risk-off
  * only the technical composite is used historically (news/fundamentals are
    current snapshots and would leak the future)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from . import config as C
from . import prices, store


@dataclass
class BTParams:
    top_n: int = 10
    hold_days: int = 40
    atr_mult: float = 1.5
    target_r: float = 2.5
    rebalance_every: int = 5
    capital: float = 1_000_000.0
    start: str | None = None
    end: str | None = None
    min_score: float = 55.0
    use_regime: bool = True
    trail: bool = False


def _costs(value: float, side: str) -> float:
    c = C.COSTS
    if side == "buy":
        pct = c.stt_buy + c.exchange_pct + c.sebi_pct + c.stamp_buy + c.slippage_pct
        gst = c.gst * (c.exchange_pct + c.sebi_pct)
        return value * (pct + gst) + c.brokerage_per_order
    pct = c.stt_sell + c.exchange_pct + c.sebi_pct + c.slippage_pct
    gst = c.gst * (c.exchange_pct + c.sebi_pct)
    return value * (pct + gst) + c.dp_per_sell + c.brokerage_per_order


def _fy(d: pd.Timestamp) -> int:
    """Indian financial year label: April-March."""
    return d.year if d.month >= 4 else d.year - 1


_PREP: dict = {}


def prepare(feat: pd.DataFrame | None = None, force: bool = False) -> dict:
    """Load and cache all matrices once. Repeated runs reuse them."""
    if feat is None:
        feat = store.load_table("features")
    if feat.empty:
        raise RuntimeError("No features. Run: python -m stocklab ingest")

    sig = (len(feat), str(feat["date"].min()), str(feat["date"].max()))
    if not force and _PREP.get("sig") == sig:
        return _PREP["data"]

    if str(feat["symbol"].dtype) == "category":
        feat = feat.assign(symbol=feat["symbol"].astype(str))

    wide = prices.build_wide()
    if not wide:
        raise RuntimeError("No price data.")
    close_df = wide["close"].sort_index()
    dates = close_df.index
    syms = list(close_df.columns)
    ix = {s: i for i, s in enumerate(syms)}

    def mat(field: str) -> np.ndarray:
        return wide[field].reindex(index=dates, columns=syms).to_numpy(dtype=float)

    score = (
        feat.pivot(index="date", columns="symbol", values="s_technical")
        .reindex(index=dates, columns=syms)
        .to_numpy(dtype=float)
    )
    liquid = (
        feat.pivot(index="date", columns="symbol", values="liquid")
        .reindex(index=dates, columns=syms)
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    atr = (
        feat.pivot(index="date", columns="symbol", values="atr14")
        .reindex(index=dates, columns=syms)
        .to_numpy(dtype=float)
    )
    elig = liquid & np.isfinite(score)

    ok = pd.Series(True, index=dates)
    bench = store.load_prices(C.BENCHMARK)
    if not bench.empty:
        b = prices.adjusted(bench)["close"]
        ok &= (b > b.rolling(200).mean()).reindex(dates).ffill().fillna(False).astype(bool)
    vix = store.load_prices(C.VIX)
    if not vix.empty:
        v = prices.adjusted(vix)["close"]
        ok &= (v < C.FILTERS.regime_vix_max).reindex(dates).ffill().fillna(False).astype(bool)

    data = {
        "dates": dates, "syms": syms, "ix": ix,
        "open": mat("open"), "high": mat("high"), "low": mat("low"),
        "close": mat("close"), "volume": mat("volume"),
        "score": score, "elig": elig, "atr": atr,
        "regime": ok.to_numpy(dtype=bool),
    }
    _PREP["sig"] = sig
    _PREP["data"] = data
    return data


def run(params: BTParams = BTParams(), feat: pd.DataFrame | None = None,
        verbose: bool = False) -> dict:
    prep = prepare(feat)
    dates, syms = prep["dates"], prep["syms"]
    o_, h_, l_, c_ = prep["open"], prep["high"], prep["low"], prep["close"]
    score, elig, atr_m = prep["score"], prep["elig"], prep["atr"]
    nd = len(dates)

    i0 = int(dates.searchsorted(pd.Timestamp(params.start))) if params.start else 0
    i1 = int(dates.searchsorted(pd.Timestamp(params.end), side="right")) if params.end else nd
    if i1 <= i0 + 2:
        return {"params": asdict(params), "error": "not enough data in range"}

    regime = prep["regime"] if params.use_regime else np.ones(nd, dtype=bool)

    cash = params.capital
    positions: dict[int, dict] = {}
    equity_rows: list[dict] = []
    trades: list[dict] = []
    last_rebalance = -10**9
    gross_costs = 0.0
    gross_turnover = 0.0
    tax_total = 0.0
    realized_by_fy: dict[int, float] = {}
    carry_loss = 0.0
    current_fy: int | None = None

    def settle(fy: int) -> None:
        nonlocal cash, tax_total, carry_loss
        net = realized_by_fy.pop(fy, 0.0) + carry_loss
        if net > 0:
            tax = net * C.COSTS.stcg_tax
            cash -= tax
            tax_total += tax
            carry_loss = 0.0
        else:
            carry_loss = net

    def equity_at(i: int) -> float:
        val = cash
        for j, p in positions.items():
            px = c_[i, j]
            val += p["qty"] * (px if np.isfinite(px) else p["entry"])
        return val

    for i in range(i0, i1):
        d = dates[i]
        if current_fy is None:
            current_fy = _fy(d)
        elif _fy(d) != current_fy:
            settle(current_fy)
            current_fy = _fy(d)

        # ---------------- exits on today's bar
        for j in list(positions):
            p = positions[j]
            if p["entry_i"] >= i:
                continue
            op, hi, lo = o_[i, j], h_[i, j], l_[i, j]
            cl = c_[i, j]
            if not np.isfinite(cl):
                continue
            exit_px, reason = None, None
            if np.isfinite(op) and op <= p["stop"]:
                exit_px, reason = op, "gap-stop"
            elif np.isfinite(lo) and lo <= p["stop"]:
                exit_px, reason = p["stop"], ("trail" if p["stop"] > p["entry"] else "stop")
            elif np.isfinite(hi) and hi >= p["target"]:
                exit_px, reason = p["target"], "target"
            elif (i - p["entry_i"]) >= params.hold_days:
                exit_px, reason = cl, "time"
            if exit_px is None:
                continue

            proceeds = p["qty"] * exit_px
            cost = _costs(proceeds, "sell")
            cash += proceeds - cost
            gross_costs += cost
            gross_turnover += proceeds
            gross = proceeds - p["qty"] * p["entry"] - p["buy_cost"] - cost
            realized_by_fy[_fy(d)] = realized_by_fy.get(_fy(d), 0.0) + gross
            trades.append({
                "symbol": syms[j], "entry_date": p["entry_date"], "exit_date": d,
                "entry": round(p["entry"], 2), "exit": round(exit_px, 2),
                "qty": p["qty"], "gross": round(gross, 2),
                "net": round(gross, 2), "reason": reason,
                "days": int(i - p["entry_i"]),
                "ret_pct": round((exit_px / p["entry"] - 1) * 100, 2),
            })
            del positions[j]

        # ---------------- entries at today's open from yesterday's close
        if i > i0 and (i - last_rebalance) >= params.rebalance_every and regime[i - 1]:
            prev = i - 1
            row = score[prev]
            mask = elig[prev] & (row >= params.min_score)
            cand = np.where(mask)[0]
            max_pos = min(params.top_n, C.RULES.max_positions)
            slots = max_pos - len(positions)
            if slots > 0 and cand.size:
                order = cand[np.argsort(-row[cand])]
                picked = [j for j in order if j not in positions][:slots]
                eq_now = equity_at(i)
                per_trade = eq_now / max(1, max_pos)
                for j in picked:
                    op = o_[i, j]
                    if not np.isfinite(op) or op <= 0:
                        continue
                    a = atr_m[prev, j]
                    if not np.isfinite(a) or a <= 0:
                        a = op * 0.02
                    risk = params.atr_mult * a
                    lo10 = np.nanmin(l_[max(0, prev - 9): prev + 1, j])
                    if np.isfinite(lo10) and lo10 < op:
                        risk = max(risk, op - lo10)
                    risk = min(risk, op * 0.12)
                    if risk <= 0:
                        continue
                    stop = op - risk
                    target = np.inf if params.trail else op + params.target_r * risk
                    qty = int(per_trade // op)
                    if qty <= 0 or qty * op < 1000:
                        continue
                    value = qty * op
                    cost = _costs(value, "buy")
                    if value + cost > cash:
                        continue
                    cash -= value + cost
                    gross_costs += cost
                    gross_turnover += value
                    positions[j] = {
                        "entry": op, "qty": qty, "stop": stop, "target": target,
                        "entry_i": i, "entry_date": d, "buy_cost": cost, "risk": risk,
                    }
                last_rebalance = i

        # ---------------- trailing stop update from today's completed bar
        if params.trail:
            for j, p in positions.items():
                if p["entry_i"] >= i:
                    continue
                a = atr_m[i, j]
                if not np.isfinite(a) or not np.isfinite(h_[i, j]):
                    continue
                new_stop = h_[i, j] - params.atr_mult * a
                if new_stop > p["stop"]:
                    p["stop"] = new_stop

        equity_rows.append({"date": d, "equity": equity_at(i), "n_pos": len(positions)})

    if current_fy is not None:
        settle(current_fy)
    if equity_rows:
        last = i1 - 1
        equity_rows[-1]["equity"] = cash + sum(
            p["qty"] * (c_[last, j] if np.isfinite(c_[last, j]) else p["entry"])
            for j, p in positions.items()
        )

    eq = pd.DataFrame(equity_rows).set_index("date") if equity_rows else pd.DataFrame()
    if eq.empty:
        return {"params": asdict(params), "error": "no data in range"}

    tr = pd.DataFrame(trades)
    res = _metrics(eq, tr, params, gross_costs, gross_turnover, tax_total)

    bench = store.load_prices(C.BENCHMARK)
    if not bench.empty:
        b = prices.adjusted(bench)["close"]
        b = b[(b.index >= eq.index[0]) & (b.index <= eq.index[-1])]
        if len(b) > 1:
            res["benchmark_return_pct"] = round(float((b.iloc[-1] / b.iloc[0] - 1) * 100), 2)
            res["benchmark_cagr_pct"] = round(
                float(((b.iloc[-1] / b.iloc[0]) ** (252 / max(1, len(b))) - 1) * 100), 2
            )
            res["excess_return_pct"] = round(
                res["total_return_pct"] - res["benchmark_return_pct"], 2
            )
    return res


def _metrics(eq: pd.DataFrame, tr: pd.DataFrame, params: BTParams,
             costs: float, turnover: float, tax_total: float = 0.0) -> dict:
    initial = params.capital
    final = float(eq["equity"].iloc[-1])
    days = (eq.index[-1] - eq.index[0]).days or 1
    years = days / 365.25
    ret = eq["equity"].pct_change().dropna()
    cagr = (final / initial) ** (1 / years) - 1 if years > 0 and final > 0 else np.nan
    sharpe = (ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else np.nan
    dd = (eq["equity"] / eq["equity"].cummax() - 1).min()
    downside = ret[ret < 0].std()
    sortino = (ret.mean() / downside * np.sqrt(252)) if downside and downside > 0 else np.nan

    out = {
        "params": asdict(params),
        "start": str(eq.index[0].date()),
        "end": str(eq.index[-1].date()),
        "initial": initial,
        "final": round(final, 2),
        "total_return_pct": round((final / initial - 1) * 100, 2),
        "cagr_pct": round(cagr * 100, 2) if np.isfinite(cagr) else None,
        "sharpe": round(float(sharpe), 2) if np.isfinite(sharpe) else None,
        "sortino": round(float(sortino), 2) if np.isfinite(sortino) else None,
        "max_drawdown_pct": round(float(dd) * 100, 2),
        "n_trades": int(len(tr)),
        "costs_paid": round(costs, 0),
        "cost_drag_pct_of_turnover": round(costs / turnover * 100, 3) if turnover else None,
        "tax_paid": round(float(tax_total), 0),
    }
    if tr.empty:
        out.update({"win_rate_pct": None, "avg_win": None, "avg_loss": None,
                    "profit_factor": None, "expectancy": None, "avg_hold_days": None})
        out["equity"] = eq
        out["trades"] = tr
        return out

    wins = tr[tr["net"] > 0]
    losses = tr[tr["net"] <= 0]
    gross_win = float(wins["net"].sum())
    gross_loss = float(-losses["net"].sum())
    out.update({
        "win_rate_pct": round(len(wins) / len(tr) * 100, 1),
        "avg_win": round(float(wins["net"].mean()), 0) if not wins.empty else 0,
        "avg_loss": round(float(losses["net"].mean()), 0) if not losses.empty else 0,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        "expectancy": round(float(tr["net"].mean()), 0),
        "avg_hold_days": round(float(tr["days"].mean()), 1),
        "avg_ret_pct": round(float(tr["ret_pct"].mean()), 2),
        "best_trade_pct": round(float(tr["ret_pct"].max()), 1),
        "worst_trade_pct": round(float(tr["ret_pct"].min()), 1),
        "exit_mix": tr["reason"].value_counts().to_dict(),
    })
    out["equity"] = eq
    out["trades"] = tr
    return out


def walk_forward(feat: pd.DataFrame | None = None, params: BTParams = BTParams()) -> dict:
    """Split history in half: first half for design, second half as the real test."""
    if feat is None:
        feat = store.load_table("features")
    dates = sorted(pd.to_datetime(feat["date"].unique()))
    mid = dates[len(dates) // 2]
    p1 = BTParams(**{**asdict(params), "start": str(pd.Timestamp(dates[0]).date()),
                     "end": str(pd.Timestamp(mid).date())})
    p2 = BTParams(**{**asdict(params), "start": str(pd.Timestamp(mid).date()),
                     "end": str(pd.Timestamp(dates[-1]).date())})
    return {"in_sample": run(p1, feat), "out_of_sample": run(p2, feat)}


def sweep(grid: dict[str, list] | None = None, feat: pd.DataFrame | None = None,
          log=print) -> pd.DataFrame:
    grid = grid or {
        "top_n": [5, 10, 15],
        "hold_days": [20, 40, 60],
        "atr_mult": [1.0, 1.5, 2.0],
    }
    import itertools

    prepare(feat)  # load matrices once
    keys = list(grid)
    total = int(np.prod([len(grid[k]) for k in keys]))
    log(f"Sweeping {total} parameter combinations...")
    rows = []
    for n, combo in enumerate(itertools.product(*[grid[k] for k in keys]), 1):
        kw = dict(zip(keys, combo))
        try:
            r = run(BTParams(**kw), feat)
        except Exception as exc:  # noqa: BLE001
            log(f"  {kw} failed: {type(exc).__name__}: {exc}")
            continue
        rows.append({
            **kw,
            "cagr_pct": r.get("cagr_pct"),
            "sharpe": r.get("sharpe"),
            "max_dd_pct": r.get("max_drawdown_pct"),
            "win_rate_pct": r.get("win_rate_pct"),
            "profit_factor": r.get("profit_factor"),
            "trades": r.get("n_trades"),
            "expectancy": r.get("expectancy"),
            "excess_vs_nifty": r.get("excess_return_pct"),
        })
        if n % 10 == 0:
            log(f"  {n}/{total}")
    out = pd.DataFrame(rows)
    if not out.empty:
        store.save_table("backtest_sweep", out)
    return out
