"""Console and HTML reporting."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from . import config as C
from . import signals


def _fmt_money(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-"
    if abs(v) >= 1e7:
        return f"{v / 1e7:,.1f} Cr"
    if abs(v) >= 1e5:
        return f"{v / 1e5:,.1f} L"
    return f"{v:,.0f}"


def print_market(market: dict) -> None:
    print("\n" + "=" * 78)
    print(f"  MARKET REGIME  ({market.get('as_of')})")
    print("=" * 78)
    print(f"  Nifty close        : {market.get('nifty_close')}")
    print(f"  vs 200-DMA         : {market.get('nifty_vs_200dma_pct')}%")
    print(f"  India VIX          : {market.get('vix')}")
    print(f"  Breadth > 50-DMA   : {market.get('breadth_50dma')}%")
    print(f"  Breadth > 200-DMA  : {market.get('breadth_200dma')}%")
    regime = market.get("regime", "unknown")
    flag = "RISK-ON  (longs allowed)" if regime == "risk-on" else "DEFENSIVE (no new longs)"
    print(f"  Regime             : {flag}")
    if market.get("reasons"):
        print(f"  Reasons            : {'; '.join(market['reasons'])}")
    print("=" * 78)


def print_screen(df: pd.DataFrame, market: dict | None = None, top: int = 20) -> None:
    if market:
        print_market(market)
    if df.empty:
        print("No rows to display.")
        return

    work = df[df.get("eligible", True)].head(top).copy()
    if work.empty:
        print("No eligible stocks passed the liquidity/size filters.")
        return

    print(f"\n  TOP {len(work)} SHORT-TERM CANDIDATES "
          f"(technical 60 / news 20 / fundamentals+results 20)\n")
    header = f"{'#':>3} {'SYMBOL':<12} {'SCORE':>5} {'CMP':>9} {'ENTRY':>9} {'STOP':>9} {'TARGET':>9} {'QTY':>5} {'ATR%':>5} {'TURN':>8}"
    print(header)
    print("-" * len(header))
    for _, r in work.iterrows():
        print(
            f"{int(r['rank']):>3} {str(r['symbol'])[:12]:<12} "
            f"{r['total']:>5.1f} {r['close']:>9.2f} {r.get('entry', np.nan):>9.2f} "
            f"{r.get('stop', np.nan):>9.2f} {r.get('target', np.nan):>9.2f} "
            f"{int(r.get('qty', 0)):>5} {r.get('atr_pct', np.nan):>5.1f} "
            f"{_fmt_money(r.get('turnover_sma20', 0) * 1e7):>8}"
        )

    print("\n  WHY (top 3 by score)")
    print("-" * 78)
    for _, r in work.head(3).iterrows():
        name = str(r.get("name", ""))[:40]
        print(f"  {r['symbol']} - {name}")
        print(f"    {r.get('why', '')}")
        if isinstance(r.get("news_top"), str) and r["news_top"]:
            print(f"    news: {r['news_top'][:100]}")
        if pd.notna(r.get("days_to_result")):
            d = int(r["days_to_result"])
            when = f"in {d}d" if d >= 0 else f"{-d}d ago"
            print(f"    results: {when}")
        print()

    summary = signals.position_summary(work)
    print("-" * 78)
    print(f"  Positions: {summary['positions']}  "
          f"Max loss if all stops hit: Rs {summary['max_loss']:,.0f}  "
          f"Avg stop: {summary.get('avg_stop_pct', 0):.1f}%  "
          f"Avg target: {summary.get('avg_target_pct', 0):.1f}%")
    print(f"  {C.DISCLAIMER}")


def print_backtest(result: dict, title: str = "BACKTEST") -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)
    if "error" in result:
        print(f"  {result['error']}")
        return
    p = result["params"]
    print(f"  Period            : {result['start']} -> {result['end']}")
    print(f"  Rules             : top {p['top_n']}, hold {p['hold_days']}d, "
          f"ATR stop {p['atr_mult']}x, target {p['target_r']}R, "
          f"rebalance {p['rebalance_every']}d")
    print(f"  Capital           : Rs {result['initial']:,.0f} -> Rs {result['final']:,.0f}")
    print(f"  Total return      : {result['total_return_pct']}%")
    print(f"  CAGR              : {result['cagr_pct']}%")
    print(f"  Sharpe / Sortino  : {result['sharpe']} / {result['sortino']}")
    print(f"  Max drawdown      : {result['max_drawdown_pct']}%")
    print(f"  Trades            : {result['n_trades']}   "
          f"win rate {result['win_rate_pct']}%   "
          f"profit factor {result['profit_factor']}")
    print(f"  Expectancy/trade  : Rs {result['expectancy']}")
    print(f"  Avg win / loss    : Rs {result['avg_win']} / Rs {result['avg_loss']}")
    print(f"  Avg hold          : {result['avg_hold_days']} days   "
          f"avg trade {result.get('avg_ret_pct')}%")
    print(f"  Costs paid        : Rs {result['costs_paid']:,.0f} "
          f"({result['cost_drag_pct_of_turnover']}% of turnover)")
    print(f"  STCG tax paid     : Rs {result['tax_paid']:,.0f}")
    if "benchmark_return_pct" in result:
        print(f"  Nifty same period : {result['benchmark_return_pct']}%  "
              f"(edge {result.get('excess_return_pct')}%)")
    if result.get("exit_mix"):
        print(f"  Exits             : {result['exit_mix']}")
    print("=" * 78)


def html_report(df: pd.DataFrame, market: dict, path=None, top: int = 25) -> str:
    path = path or (C.REPORT_DIR / f"screen_{datetime.now():%Y%m%d_%H%M}.html")
    work = df[df.get("eligible", True)].head(top).copy()

    cols = ["rank", "symbol", "name", "sector", "close", "total", "s_technical",
            "s_news", "s_fundamental", "s_results", "atr_pct", "turnover_sma20",
            "entry", "stop", "target", "qty", "why"]
    cols = [c for c in cols if c in work.columns]
    tbl = work[cols].copy()
    for c in ("close", "entry", "stop", "target"):
        if c in tbl:
            tbl[c] = tbl[c].map(lambda v: f"{v:,.2f}" if pd.notna(v) else "-")
    for c in ("total", "s_technical", "s_news", "s_fundamental", "s_results", "atr_pct"):
        if c in tbl:
            tbl[c] = tbl[c].map(lambda v: f"{v:.1f}" if pd.notna(v) else "-")
    if "turnover_sma20" in tbl:
        tbl["turnover_sma20"] = tbl["turnover_sma20"].map(lambda v: f"{v:,.0f} Cr")
    tbl.columns = [str(c).replace("_", " ").title() for c in tbl.columns]

    regime_cls = "on" if market.get("regime") == "risk-on" else "off"
    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Stocklab screen {market.get('as_of')}</title>
<style>
 body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color:#1a1a1a; background:#fafafa; }}
 h1 {{ font-size: 20px; margin: 0 0 4px 0; }}
 .sub {{ color:#666; font-size: 13px; margin-bottom: 16px; }}
 .badge {{ display:inline-block; padding:4px 10px; border-radius: 12px; font-size:12px; font-weight:600; }}
 .on {{ background:#e6f4ea; color:#137333; }}
 .off {{ background:#fce8e6; color:#c5221f; }}
 .stats {{ display:flex; gap:18px; flex-wrap:wrap; margin:14px 0 20px 0; font-size:13px; }}
 .stats div {{ background:#fff; border:1px solid #e5e5e5; border-radius:8px; padding:8px 12px; }}
 table {{ border-collapse: collapse; width:100%; background:#fff; font-size:12px; }}
 th {{ background:#f1f3f4; text-align:left; padding:7px 8px; border-bottom:1px solid #ddd; position:sticky; top:0; }}
 td {{ padding:6px 8px; border-bottom:1px solid #f0f0f0; vertical-align: top; }}
 tr:hover td {{ background:#f8fbff; }}
 .why {{ color:#555; font-size:11px; }}
 .foot {{ margin-top:18px; font-size:11px; color:#777; }}
</style></head><body>
<h1>Nifty 500 short-term screen &mdash; {market.get('as_of')}</h1>
<div class="sub">Composite: technical 60% &middot; news 20% &middot; fundamentals &amp; results 20%</div>
<span class="badge {regime_cls}">{market.get('regime', 'unknown').upper()}</span>
<div class="stats">
  <div>Nifty {market.get('nifty_close')} ({market.get('nifty_vs_200dma_pct')}% vs 200-DMA)</div>
  <div>India VIX {market.get('vix')}</div>
  <div>Breadth&gt;50DMA {market.get('breadth_50dma')}%</div>
  <div>Breadth&gt;200DMA {market.get('breadth_200dma')}%</div>
</div>
{tbl.to_html(index=False, escape=False, classes='t', na_rep='-')}
<div class="foot">{C.DISCLAIMER}<br>
Generated {datetime.now():%d %b %Y %H:%M} &middot; costs assumed: STT 0.1%/side, slippage 0.15%/side, STCG 20%.</div>
</body></html>"""
    path = str(path)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path
