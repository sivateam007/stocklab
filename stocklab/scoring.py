"""Composite scoring: technical + news + fundamentals + results, then ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import earnings, fundamentals, news as news_mod, store, technical

# Weights for the 60/20/20 model. Fundamentals and results share the 20%.
FUND_SPLIT = {"fundamental": 0.60, "results": 0.40}


def _composite(row: pd.Series) -> tuple[float, dict]:
    """Weighted score that renormalises when a block is unavailable."""
    parts = {
        "technical": (row.get("s_technical"), C.WEIGHTS.trend + C.WEIGHTS.momentum
                      + C.WEIGHTS.breakout + C.WEIGHTS.volume + C.WEIGHTS.rel_strength),
        "news": (row.get("s_news"), C.WEIGHTS.news),
        "fundamental": (row.get("s_fundamental"), C.WEIGHTS.fundamental * FUND_SPLIT["fundamental"]),
        "results": (row.get("s_results"), C.WEIGHTS.fundamental * FUND_SPLIT["results"]),
    }
    num = den = 0.0
    used = {}
    for name, (val, w) in parts.items():
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        num += float(val) * w
        den += w
        used[name] = round(float(val), 1)
    if den == 0:
        return np.nan, used
    return num / den, used


def _why(row: pd.Series) -> str:
    bits = []
    if pd.notna(row.get("dist_high52")) and row["dist_high52"] > -5:
        bits.append(f"{abs(row['dist_high52']):.1f}% from 52w high")
    if pd.notna(row.get("vol_ratio")) and row["vol_ratio"] > 1.5:
        bits.append(f"volume {row['vol_ratio']:.1f}x avg")
    if pd.notna(row.get("rs_20")):
        bits.append(f"RS {row['rs_20']:+.1f}% vs Nifty (20d)")
    if pd.notna(row.get("sma50_slope")) and row["sma50_slope"] > 0:
        bits.append("50-DMA rising")
    if pd.notna(row.get("adx14")) and row["adx14"] > 25:
        bits.append(f"ADX {row['adx14']:.0f} (strong trend)")
    if pd.notna(row.get("st_dir")) and row["st_dir"] > 0:
        bits.append("Supertrend up")
    if pd.notna(row.get("s_results")) and row["s_results"] > 60:
        sp = row.get("surprise_pct")
        if pd.notna(sp):
            bits.append(f"results drift +{sp:.0f}% surprise")
        else:
            bits.append(f"post-results drift {row['reaction_pct']:+.1f}%"
                        if pd.notna(row.get("reaction_pct")) else "positive post-results drift")
    if pd.notna(row.get("s_news")) and row["s_news"] > 65:
        bits.append("positive news flow")
    return " | ".join(bits[:4]) if bits else "no standout factor"


def latest_rows(feat: pd.DataFrame | None = None) -> pd.DataFrame:
    if feat is None:
        feat = store.load_table("features")
    if feat.empty:
        return pd.DataFrame()
    last = feat["date"].max()
    out = feat[feat["date"] == last].copy()
    if str(out["symbol"].dtype) == "category":
        out["symbol"] = out["symbol"].astype(str)
    return out


def screen(asof=None, refresh_news_for: int = 60, with_news: bool = True,
           news_backend: str = "lexicon", log=print) -> pd.DataFrame:
    """Full live screener for the most recent session."""
    feat = store.load_table("features_latest")
    if feat.empty:
        feat = store.load_table("features")
    if feat.empty:
        raise RuntimeError("No features found. Run: python -m stocklab ingest")
    latest = latest_rows(feat)
    symbols = latest["symbol"].tolist()

    uni = store.load_table("universe")
    if not uni.empty:
        latest = latest.merge(
            uni[["symbol", "name", "industry", "sector"]].rename(columns={"sector": "sector_universe"}),
            on="symbol", how="left",
        )
        if "sector" in latest.columns:
            latest["sector"] = latest["sector"].fillna(latest["sector_universe"])
        else:
            latest["sector"] = latest["sector_universe"]
        latest = latest.drop(columns=["sector_universe"], errors="ignore")
    else:
        latest["name"] = latest["symbol"]
        latest["industry"] = ""
        latest["sector"] = "Other"

    # ---- fundamentals
    fund = fundamentals.score_fundamentals()
    if not fund.empty:
        cols = ["symbol", "s_quality", "s_growth", "s_valuation", "s_fundamental",
                "trailingPE", "priceToBook", "returnOnEquity", "debtToEquity",
                "revenueGrowth_pct", "earningsGrowth_pct"]
        cols = [c for c in cols if c in fund.columns]
        latest = latest.merge(fund[cols], on="symbol", how="left")
    else:
        latest["s_fundamental"] = np.nan

    # ---- news
    if with_news:
        ranked = (latest.sort_values("s_technical", ascending=False)["symbol"]
                  .astype(str).head(refresh_news_for).tolist())
        news_df = store.load_table("news")
        names = dict(zip(latest["symbol"].astype(str), latest["name"])) if "name" in latest else {}

        stale = True
        if not news_df.empty and "published" in news_df:
            newest = pd.to_datetime(news_df["published"], errors="coerce", utc=True).max()
            stale = pd.isna(newest) or (pd.Timestamp.now(tz="UTC") - newest) > pd.Timedelta(hours=12)

        if news_df.empty or stale:
            try:
                pulled = news_mod.refresh(ranked, names, log=log)
                if not pulled.empty:
                    news_df = pulled
            except Exception as exc:  # noqa: BLE001
                log(f"news refresh failed: {type(exc).__name__}")

        if not news_df.empty:
            # Score headline flow only for names that can actually be bought.
            keep = set(ranked)
            subset = news_df[news_df["symbol"].astype(str).isin(keep)]
            if subset.empty:
                subset = news_df
            if news_backend == "finbert":
                scorer = news_mod.get_scorer()
            else:
                scorer = news_mod.LexiconScorer()
            scored = news_mod.score_news(subset, scorer=scorer)
            latest = latest.merge(scored, on="symbol", how="left")
    if "s_news" not in latest:
        latest["s_news"] = np.nan

    # ---- results
    res = earnings.results_state()
    if not res.empty:
        latest = latest.merge(
            res[["symbol", "days_to_result", "results_watch", "surprise_pct",
                 "days_since_result", "reaction_pct", "s_results"]],
            on="symbol", how="left",
        )

    # ---- composite
    comp = latest.apply(_composite, axis=1, result_type="expand")
    latest["total"] = comp[0]
    latest["blocks_used"] = comp[1].map(lambda d: ",".join(d.keys()))

    penalty = np.where(latest.get("days_to_result", pd.Series(np.nan, index=latest.index)).between(0, 5), 5.0, 0.0)
    latest["total"] = (latest["total"] - penalty).clip(0, 100)

    latest["eligible"] = latest["liquid"].fillna(False) & latest["total"].notna()
    latest = latest.sort_values("total", ascending=False).reset_index(drop=True)
    latest["rank"] = np.where(latest["eligible"], np.arange(1, len(latest) + 1), np.nan)
    latest["why"] = latest.apply(_why, axis=1)
    return latest


def technical_history(feat: pd.DataFrame | None = None) -> pd.DataFrame:
    """Point-in-time technical composite for every date (used by the backtester).

    Fundamentals and news are *current snapshots*, so including them in a
    historical test would be look-ahead bias. This keeps history honest.
    """
    if feat is None:
        feat = store.load_table("features")
    if feat.empty:
        return feat
    df = feat.copy()
    df["total"] = df["s_technical"]
    df["eligible"] = df["liquid"].fillna(False) & df["total"].notna()
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)
