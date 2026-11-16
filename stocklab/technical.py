"""Technical, momentum, breakout, volume and relative-strength factor engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib

from . import config as C
from . import prices, store

EPS = 1e-9


# ------------------------------------------------------------------ helpers
def _arr(s: pd.Series) -> np.ndarray:
    return np.ascontiguousarray(s.to_numpy(dtype=np.float64, na_value=np.nan))


def _pct(s: pd.Series) -> pd.Series:
    return s.rank(pct=True) * 100.0


def _slope(s: pd.Series, window: int = 20) -> pd.Series:
    """Normalised linear-regression slope over `window` bars."""
    x = np.arange(window, dtype=np.float64)
    x = x - x.mean()
    denom = float((x ** 2).sum())

    def _fit(v: np.ndarray) -> float:
        if np.isnan(v).any():
            return np.nan
        return float(np.dot(v - v.mean(), x) / denom)

    raw = s.rolling(window).apply(_fit, raw=True)
    return raw / s.abs().replace(0, np.nan) * 100.0


def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.DataFrame:
    """Classic Supertrend. Returns columns st_line and st_dir (+1 up / -1 down)."""
    high, low, close = _arr(df["high"]), _arr(df["low"]), _arr(df["close"])
    n = len(df)
    out_line = np.full(n, np.nan)
    out_dir = np.zeros(n, dtype=np.int8)
    if n < period + 2:
        return pd.DataFrame({"st_line": out_line, "st_dir": out_dir}, index=df.index)

    atr = talib.ATR(high, low, close, timeperiod=period)
    hl2 = (high + low) / 2.0
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr

    final_upper = upper.copy()
    final_lower = lower.copy()
    direction = np.ones(n, dtype=np.int8)

    for i in range(1, n):
        if np.isnan(atr[i]):
            direction[i] = 1
            continue
        if np.isnan(final_upper[i - 1]):
            final_upper[i] = upper[i]
            final_lower[i] = lower[i]
            direction[i] = 1 if close[i] > final_upper[i] else -1
        else:
            final_upper[i] = (
                upper[i]
                if (upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1])
                else final_upper[i - 1]
            )
            final_lower[i] = (
                lower[i]
                if (lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1])
                else final_lower[i - 1]
            )
            if direction[i - 1] == 1 and close[i] < final_lower[i]:
                direction[i] = -1
            elif direction[i - 1] == -1 and close[i] > final_upper[i]:
                direction[i] = 1
            else:
                direction[i] = direction[i - 1]

    for i in range(n):
        out_dir[i] = direction[i]
        out_line[i] = final_lower[i] if direction[i] == 1 else final_upper[i]
    return pd.DataFrame({"st_line": out_line, "st_dir": out_dir}, index=df.index)


# ------------------------------------------------------------------ per stock
def compute_stock(df: pd.DataFrame) -> pd.DataFrame:
    """Raw technical factors for one symbol, indexed by date (adjusted prices)."""
    if df is None or len(df) < 60:
        return pd.DataFrame()
    d = prices.adjusted(df).copy()
    o, h, l, c, v = (_arr(d[k]) for k in ("open", "high", "low", "close", "volume"))

    out = pd.DataFrame(index=d.index)
    out["close"] = c
    out["volume"] = v

    # ---- trend
    for p in (20, 50, 200):
        out[f"ema{p}"] = talib.EMA(c, timeperiod=p)
    out["sma50"] = talib.SMA(c, timeperiod=50)
    out["sma200"] = talib.SMA(c, timeperiod=200)
    out["sma50_slope"] = _slope(out["sma50"], 20)
    out["ema_stack"] = (
        (out["ema20"] > out["ema50"]) & (out["ema50"] > out["ema200"])
    ).astype(float)
    out["above_ema200"] = (out["close"] > out["ema200"]).astype(float)
    out["dist_ema20_atr"] = (out["close"] - out["ema20"]) / talib.ATR(h, l, c, 14).clip(1e-9)

    st = supertrend(d)
    out["st_dir"] = st["st_dir"].astype(float)
    out["st_line"] = st["st_line"]

    # ---- momentum
    out["rsi14"] = talib.RSI(c, timeperiod=14)
    macd, sig, hist = talib.MACD(c, 12, 26, 9)
    out["macd_hist"] = hist
    out["macd_hist_prev"] = pd.Series(hist, index=d.index).shift(3)
    out["macd_rising"] = (out["macd_hist"] > out["macd_hist_prev"]).astype(float)
    out["roc20"] = talib.ROC(c, 20)
    out["roc60"] = talib.ROC(c, 60)
    out["roc120"] = talib.ROC(c, 120)
    out["adx14"] = talib.ADX(h, l, c, 14)
    out["plus_di"] = talib.PLUS_DI(h, l, c, 14)
    out["minus_di"] = talib.MINUS_DI(h, l, c, 14)
    out["di_spread"] = out["plus_di"] - out["minus_di"]

    # ---- volatility
    atr = talib.ATR(h, l, c, 14)
    out["atr14"] = atr
    out["atr_pct"] = atr / np.maximum(c, EPS) * 100.0
    ub, mb, lb = talib.BBANDS(c, 20, 2, 2, 0)
    out["bb_width"] = (ub - lb) / np.maximum(mb, EPS) * 100.0
    out["bb_width_pctile"] = out["bb_width"].rolling(252, min_periods=60).rank(pct=True) * 100.0
    out["atr_pct_pctile"] = out["atr_pct"].rolling(252, min_periods=60).rank(pct=True) * 100.0

    # ---- breakout structure
    s = pd.Series(c, index=d.index)
    out["high20"] = pd.Series(h, index=d.index).rolling(20).max()
    out["low20"] = pd.Series(l, index=d.index).rolling(20).min()
    out["high52"] = pd.Series(h, index=d.index).rolling(252, min_periods=120).max()
    out["dist_high20"] = (out["close"] / out["high20"] - 1.0) * 100.0
    out["dist_high52"] = (out["close"] / out["high52"] - 1.0) * 100.0
    out["new_high20"] = (s >= out["high20"]).astype(float)
    out["swing_low10"] = pd.Series(l, index=d.index).rolling(10).min()

    # ---- volume
    out["vol_sma20"] = pd.Series(v, index=d.index).rolling(20).mean()
    out["vol_ratio"] = v / np.maximum(out["vol_sma20"].to_numpy(), EPS)
    out["turnover_cr"] = c * v / 1e7
    out["turnover_sma20"] = out["turnover_cr"].rolling(20).mean()
    out["obv"] = talib.OBV(c, v)
    out["obv_slope"] = _slope(out["obv"], 20)
    out["vol_price_confirm"] = (
        (out["vol_ratio"] > 1.5) & (out["close"] > s.shift(1))
    ).astype(float)

    # ---- shaped (better direction than raw percentiles)
    out["rsi_band"] = (1.0 - (out["rsi14"] - 60.0).abs() / 25.0).clip(0.0, 1.0) * 100.0
    out["vol_score_raw"] = np.log1p(out["vol_ratio"].clip(0, 10)) * 100.0
    return out


# ------------------------------------------------------------------ cross section
def _block(scores: list[pd.Series], weights: list[float]) -> pd.Series:
    num = None
    den = None
    for sc, w in zip(scores, weights):
        term = sc.fillna(sc.median()) * w
        num = term if num is None else num + term
        den = w if den is None else den + w
    return num / den


def build_features(symbols: list[str] | None = None, save: bool = True) -> pd.DataFrame:
    symbols = symbols or [s for s in store.available_symbols() if not s.startswith("^")]
    frames = []
    for i, sym in enumerate(symbols):
        raw = store.load_prices(sym)
        if raw.empty:
            continue
        f = compute_stock(raw)
        if f.empty:
            continue
        f["symbol"] = sym
        frames.append(f.reset_index())
        if (i + 1) % 100 == 0:
            print(f"  features {i + 1}/{len(symbols)}")

    if not frames:
        return pd.DataFrame()

    feat = pd.concat(frames, ignore_index=True)

    # Sector must ride on the row before scoring: s_rel_strength ranks each
    # stock against its sector's median momentum.
    uni = store.load_table("universe")
    if not uni.empty and "sector" in uni.columns:
        smap = dict(zip(uni["symbol"], uni["sector"]))
        feat["sector"] = feat["symbol"].map(smap)

    by_day = feat.groupby("date", group_keys=False)

    def r(col: str) -> pd.Series:
        return by_day[col].transform(_pct)

    def inv(col: str) -> pd.Series:
        """Rank reversed: a high raw value must score low (stable-negative IC)."""
        return 100.0 - r(col)

    # ---- scores
    # Voting rule (two-half IC evidence, 20d and 40d horizons): a member's IC
    # sign must survive both halves of history at both horizons, and it must
    # reach |t| >= 1.5 somewhere (obv_slope, vol_price_confirm fail); weights
    # inside a block are proportional to the member's mean |IC|.
    feat["s_trend"] = _block(
        [r("ema_stack"), r("adx14"), r("above_ema200"), r("st_dir"), inv("minus_di")],
        [0.30, 0.25, 0.15, 0.15, 0.15],
    )
    feat["s_momentum"] = _block(
        [r("roc120"), inv("macd_rising")], [0.70, 0.30],
    )
    feat["s_breakout"] = _block([r("dist_high52")], [1.0])
    feat["s_volume"] = _block(
        [inv("turnover_sma20"), r("vol_ratio")], [0.90, 0.10],
    )

    # Index-relative rs_N = roc_N - index_return is rank-identical to roc_N
    # (the index return is constant across a date), so it is kept only for
    # display/diagnostics. The block votes on sector-relative momentum.
    bench = store.load_prices(C.BENCHMARK)
    if not bench.empty:
        b = prices.adjusted(bench)
        bret = {20: b["close"].pct_change(20) * 100,
                60: b["close"].pct_change(60) * 100,
                120: b["close"].pct_change(120) * 100}
        for n in (20, 60, 120):
            feat[f"rs_{n}"] = feat[f"roc{n}"] - feat["date"].map(bret[n])
    else:
        for n in (20, 60, 120):
            feat[f"rs_{n}"] = np.nan

    if "sector" in feat.columns:
        feat["sector_rs_120"] = feat["roc120"] - feat.groupby(
            ["date", "sector"], observed=True)["roc120"].transform("median")
        feat["s_rel_strength"] = _block([r("sector_rs_120")], [1.0])
    else:
        # No sector map: fall back to index-relative 120d RS (rank-identical
        # to roc120, i.e. a momentum duplicate) so scoring still runs.
        feat["sector_rs_120"] = np.nan
        feat["s_rel_strength"] = _block([r("rs_120")], [1.0])

    # technical composite 0-100
    w = C.WEIGHTS
    tech = (
        feat["s_trend"] * w.trend
        + feat["s_momentum"] * w.momentum
        + feat["s_breakout"] * w.breakout
        + feat["s_volume"] * w.volume
        + feat["s_rel_strength"].fillna(feat["s_rel_strength"].median()) * w.rel_strength
    ) / (w.trend + w.momentum + w.breakout + w.volume + w.rel_strength)
    feat["s_technical"] = tech.clip(0, 100)

    # ---- liquidity gate
    f = C.FILTERS
    feat["liquid"] = (
        (feat["turnover_sma20"] >= f.min_turnover_cr)
        & (feat["close"] >= f.min_price)
        & (feat["atr_pct"] <= f.max_atr_pct)
    )

    feat = feat.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Raw intermediates that nothing downstream reads. Dropping them keeps the
    # history file small enough to load on a low-memory machine.
    drop = ["ema20", "ema50", "ema200", "high20", "low20", "high52",
            "vol_sma20", "turnover_cr", "obv", "st_line", "macd_hist_prev"]
    feat = feat.drop(columns=[c for c in drop if c in feat.columns])

    # Store compactly: 32-bit floats and categorical columns.
    for col in feat.columns:
        if feat[col].dtype == "float64":
            feat[col] = feat[col].astype("float32")
    feat["symbol"] = feat["symbol"].astype("category")
    if "sector" in feat.columns:
        feat["sector"] = feat["sector"].astype("category")

    if save:
        store.save_table("features", feat)
        store.save_table("features_latest", feat[feat["date"] == feat["date"].max()])
    return feat


# ------------------------------------------------------------------ market state
def market_state(feat: pd.DataFrame | None = None) -> dict:
    """Regime snapshot: index trend, VIX, breadth."""
    bench = store.load_prices(C.BENCHMARK)
    vix = store.load_prices(C.VIX)
    out: dict = {"as_of": None, "nifty_close": None, "nifty_vs_200dma_pct": None,
                 "vix": None, "breadth_50dma": None, "breadth_200dma": None,
                 "regime": "unknown", "reasons": []}

    if not bench.empty:
        b = prices.adjusted(bench)["close"]
        sma200 = b.rolling(200).mean()
        sma50 = b.rolling(50).mean()
        out["as_of"] = str(b.index.max().date())
        out["nifty_close"] = round(float(b.iloc[-1]), 2)
        out["nifty_vs_200dma_pct"] = round(float((b.iloc[-1] / sma200.iloc[-1] - 1) * 100), 2)
        out["nifty_above_50dma"] = bool(b.iloc[-1] > sma50.iloc[-1])
        out["nifty_above_200dma"] = bool(b.iloc[-1] > sma200.iloc[-1])

    if not vix.empty:
        v = prices.adjusted(vix)["close"]
        out["vix"] = round(float(v.iloc[-1]), 2)

    if feat is None:
        feat = store.load_table("features_latest")
        if feat.empty:
            feat = store.load_table("features")
    if not feat.empty:
        last_date = feat["date"].max()
        today = feat[feat["date"] == last_date]
        if "above_ema200" in today:
            out["breadth_200dma"] = round(float(today["above_ema200"].mean() * 100), 1)
        if "close" in today and "sma50" in today:
            out["breadth_50dma"] = round(
                float((today["close"] > today["sma50"]).mean() * 100), 1
            )

    f = C.FILTERS
    ok = True
    if out["nifty_above_200dma"] is False:
        ok = False
        out["reasons"].append("Nifty below 200-DMA")
    if out["vix"] is not None and out["vix"] > f.regime_vix_max:
        ok = False
        out["reasons"].append(f"India VIX {out['vix']} > {f.regime_vix_max}")
    if out["breadth_50dma"] is not None and out["breadth_50dma"] < f.regime_breadth_min:
        ok = False
        out["reasons"].append(f"breadth {out['breadth_50dma']}% below {f.regime_breadth_min}%")
    out["regime"] = "risk-on" if ok else "defensive"
    return out
