"""Command line interface:  python -m stocklab <command>"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime

import pandas as pd

from . import backtest, config as C, earnings, fundamentals, news, prices
from . import report, scoring, signals, store, technical, universe


def cmd_ingest(args) -> None:
    t0 = time.time()
    run = store.start_run("ingest")
    syms = universe.universe_symbols(force=args.refresh_universe)
    print(f"Universe: {len(syms)} Nifty 500 symbols")
    bench = [C.BENCHMARK, C.VIX]
    print("\n[1/4] Prices")
    r = prices.update_symbols(syms + bench, force=args.force)
    print(f"  {r['ok']} symbols ok, {len(r['failed'])} failed, {r['rows']:,} rows")

    if not args.skip_features:
        print("\n[2/4] Technical features")
        feat = technical.build_features(syms)
        print(f"  {len(feat):,} rows, {feat['symbol'].nunique()} symbols")

    if not args.skip_fundamentals:
        print("\n[3/4] Fundamentals")
        fund = fundamentals.fetch_fundamentals(syms, force=args.force_fundamentals)
        print(f"  {len(fund)} companies")

    if not args.skip_earnings:
        print("\n[4/4] Earnings calendar")
        cal = earnings.fetch_calendar(syms)
        print(f"  {len(cal)} companies with dates")

    store.finish_run(run, "ok", f"{time.time() - t0:.0f}s")
    print(f"\nDone in {time.time() - t0:.0f}s")


def cmd_features(args) -> None:
    feat = technical.build_features()
    print(f"{len(feat):,} rows, {feat['symbol'].nunique()} symbols")


def cmd_screen(args) -> None:
    market = technical.market_state()
    df = scoring.screen(with_news=not args.no_news,
                        refresh_news_for=args.news_top,
                        news_backend=args.news_backend)
    plans = signals.attach_plans(df, capital=args.capital, top=args.top)
    report.print_screen(plans, market, top=args.top)

    if not args.no_report:
        path = report.html_report(df, market, top=args.top)
        print(f"\nHTML report: {path}")

    store.save_table("screen_latest", df)
    store.kv_set("screen_asof", market.get("as_of"))


def cmd_backtest(args) -> None:
    if args.sweep:
        out = backtest.sweep()
        if out.empty:
            print("No results.")
            return
        out = out.sort_values("sharpe", ascending=False)
        print("\nParameter sweep (sorted by Sharpe)\n")
        print(out.to_string(index=False))
        return

    p = backtest.BTParams(top_n=args.top, hold_days=args.hold, atr_mult=args.atr)
    res = backtest.run(p)
    report.print_backtest(res)

    if args.walk_forward:
        wf = backtest.walk_forward(params=p)
        report.print_backtest(wf["in_sample"], "WALK-FORWARD - IN SAMPLE (first half)")
        report.print_backtest(wf["out_of_sample"], "WALK-FORWARD - OUT OF SAMPLE (second half)")

    if not args.no_report:
        eq = res.get("equity")
        if eq is not None:
            path = C.REPORT_DIR / f"backtest_{datetime.now():%Y%m%d_%H%M}.csv"
            eq.to_csv(path)
            print(f"\nEquity curve saved: {path}")


def cmd_report(args) -> None:
    df = store.load_table("screen_latest")
    if df.empty:
        print("Run `screen` first.")
        return
    market = technical.market_state()
    path = report.html_report(df, market, top=args.top)
    print(f"HTML report: {path}")


def cmd_news(args) -> None:
    uni = store.load_table("universe")
    syms = args.symbols or uni["symbol"].tolist()[: args.limit]
    names = dict(zip(uni["symbol"], uni["name"])) if not uni.empty else {}
    df = news.refresh(syms, names)
    if df.empty:
        print("No news retrieved.")
        return
    scored = news.score_news(df)
    print(f"{len(df)} headlines, {len(scored)} symbols")
    print(scored.sort_values("s_news", ascending=False).head(20).to_string(index=False))


def cmd_status(args) -> None:
    syms = store.available_symbols()
    feat = store.load_table("features")
    print(f"Price series      : {len(syms)}")
    if syms:
        d = store.load_prices(syms[0])
        print(f"Latest session    : {d.index.max().date() if not d.empty else 'n/a'}")
    print(f"Feature rows      : {len(feat):,}")
    print(f"Fundamentals      : {len(store.load_table('fundamentals'))}")
    print(f"Earnings calendar : {len(store.load_table('earnings_calendar'))}")
    market = technical.market_state(feat)
    print(f"Regime            : {market['regime']}")
    print("\nRecent runs:")
    print(store.recent_runs(8).to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stocklab", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("ingest", help="download prices, build features, fetch fundamentals")
    i.add_argument("--force", action="store_true")
    i.add_argument("--force-fundamentals", action="store_true")
    i.add_argument("--refresh-universe", action="store_true")
    i.add_argument("--skip-features", action="store_true")
    i.add_argument("--skip-fundamentals", action="store_true")
    i.add_argument("--skip-earnings", action="store_true")
    i.set_defaults(func=cmd_ingest)

    f = sub.add_parser("features", help="rebuild technical features only")
    f.set_defaults(func=cmd_features)

    s = sub.add_parser("screen", help="rank the universe and build trade plans")
    s.add_argument("--top", type=int, default=20)
    s.add_argument("--capital", type=float, default=100_000)
    s.add_argument("--no-news", action="store_true")
    s.add_argument("--news-top", type=int, default=60)
    s.add_argument("--news-backend", choices=["lexicon", "finbert"], default="lexicon",
                   help="lexicon is instant; finbert is better but slow on CPU")
    s.add_argument("--no-report", action="store_true")
    s.set_defaults(func=cmd_screen)

    b = sub.add_parser("backtest", help="validate the strategy on history")
    b.add_argument("--top", type=int, default=10)
    b.add_argument("--hold", type=int, default=40)
    b.add_argument("--atr", type=float, default=1.5)
    b.add_argument("--sweep", action="store_true")
    b.add_argument("--walk-forward", action="store_true")
    b.add_argument("--no-report", action="store_true")
    b.set_defaults(func=cmd_backtest)

    r = sub.add_parser("report", help="regenerate HTML report from last screen")
    r.add_argument("--top", type=int, default=25)
    r.set_defaults(func=cmd_report)

    n = sub.add_parser("news", help="fetch and score news")
    n.add_argument("--symbols", nargs="*")
    n.add_argument("--limit", type=int, default=25)
    n.set_defaults(func=cmd_news)

    st = sub.add_parser("status", help="show data state")
    st.set_defaults(func=cmd_status)

    sd = sub.add_parser("study", help="test parameter robustness across two halves")
    sd.add_argument("--top", type=int, default=10)
    sd.set_defaults(func=cmd_study)

    ic = sub.add_parser("ic", help="factor predictive power (information coefficient)")
    ic.add_argument("--horizon", type=int, default=20)
    ic.set_defaults(func=cmd_ic)
    return p


def cmd_ic(args) -> None:
    from . import diagnostics

    df = diagnostics.factor_ic(horizon=args.horizon)
    print(diagnostics.summarise(df, horizon=args.horizon))


def cmd_study(args) -> None:
    from . import study

    df = study.param_study()
    print(study.summarise(df, top=args.top))


def _configure_stdout() -> None:
    """Windows consoles default to cp1252 and crash on ₹ and other non-ASCII
    characters that appear in headlines. Force UTF-8 with replacement so a
    piped run (scheduled task, Tee-Object) can never die mid-report."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def main(argv=None) -> int:
    _configure_stdout()
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
