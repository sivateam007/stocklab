"""Minimal self-checks. Run: python tests/test_units.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from stocklab import config as C
from stocklab import news, prices, signals, technical, universe


def test_cost_model_is_sane() -> None:
    rt = C.COSTS.round_trip_pct()
    assert 0.002 < rt < 0.006, f"round-trip friction {rt:.4f} out of range"
    print(f"  round-trip friction: {rt * 100:.2f}%")


def test_universe_sector_map() -> None:
    assert universe.broad_sector("Financial Services") == "Financials"
    assert universe.broad_sector("Information Technology") == "IT"
    assert universe.broad_sector(None) == "Other"
    print("  sector mapping ok")


def _synthetic(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    ret = rng.normal(0.0008, 0.018, n)
    close = 100 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    vol = rng.integers(100_000, 3_000_000, n).astype(float)
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                         "Adj Close": close, "Volume": vol}, index=idx)


def test_indicators_and_factors() -> None:
    raw = _synthetic()
    df = prices.adjusted(raw)
    for col in ("open", "high", "low", "close", "volume"):
        assert col in df.columns
    f = technical.compute_stock(raw)
    assert not f.empty
    needed = {"ema20", "ema50", "ema200", "rsi14", "adx14", "atr14", "vol_ratio",
              "dist_high52", "st_dir", "rsi_band", "vol_score_raw"}
    missing = needed - set(f.columns)
    assert not missing, f"missing {missing}"
    tail = f.tail(60)
    assert tail["rsi14"].between(0, 100).all()
    assert tail["atr_pct"].gt(0).all()
    assert set(tail["st_dir"].unique()) <= {-1.0, 0.0, 1.0}
    print("  indicators ok")


def test_trade_plan_ordering() -> None:
    row = pd.Series({"close": 100.0, "atr14": 3.0, "swing_low10": 93.0})
    plan = signals.plan_trade(row, capital=200_000)
    assert plan["stop"] < plan["entry"] < plan["target"], plan
    assert plan["qty"] > 0
    assert plan["position_value"] <= 200_000 * 0.25 + 1
    print(f"  plan: {plan['entry']} / {plan['stop']} / {plan['target']} x{plan['qty']}")


def test_sentiment_direction() -> None:
    sc = news.LexiconScorer()
    good = float(sc.score(["Strong results, profit surges 40 percent"])[0])
    bad = float(sc.score(["Company reports fraud, CEO resigns"])[0])
    assert good > 0.25, good
    assert bad < -0.25, bad
    print(f"  lexicon: positive {good:+.2f}, negative {bad:+.2f}")


def test_supertrend_warmup() -> None:
    raw = _synthetic(300)
    st = technical.supertrend(prices.adjusted(raw))
    assert st["st_dir"].abs().sum() > 0
    assert len(st) == len(raw)
    print("  supertrend ok")


def test_results_state_not_blank() -> None:
    """s_results must be anchored on the fresh calendar + our own prices.

    Yahoo's EPS surprise history is stale for .NS tickers, so the block used
    to come back all-NaN for every symbol.  Check it now actually votes, and
    that no stale surprise number sneaks through.
    """
    from stocklab import earnings

    r = earnings.results_state()
    assert not r.empty, "results_state returned nothing"
    for col in ("days_to_result", "results_watch", "surprise_pct",
                "days_since_result", "reaction_pct", "s_results"):
        assert col in r.columns, f"missing column {col}"
    scored = r[r["s_results"].notna()]
    assert len(scored) > 0, "s_results is all-NaN again"
    assert scored["s_results"].between(0, 100).all()
    assert scored["reaction_pct"].notna().all(), "scored rows need a price reaction"
    fresh = r[r["surprise_pct"].notna()]
    if len(fresh):
        age = fresh["days_since_result"]
        assert age.le(60).all(), "stale surprise leaked into surprise_pct"
    print(f"  results: {len(scored)}/{len(r)} scored, "
          f"{int(r['results_watch'].sum())} in the 0-7d window")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            print(f"{t.__name__}")
            t()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
