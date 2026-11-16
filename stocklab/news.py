"""News and corporate-announcement engine with sentiment scoring.

Sources:
  * NSE corporate announcements (authoritative, exchange filings)
  * Google News RSS per company
  * Economic Times / Moneycontrol market RSS

Sentiment uses FinBERT when the model is available, otherwise a finance
lexicon with negation handling. News is 20% of the composite score by design:
headline sentiment is noisy and must never dominate price and volume evidence.
"""

from __future__ import annotations

import html
import re
import time
import warnings
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests

from . import config as C
from . import store

warnings.filterwarnings("ignore")

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)
HEADERS = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}

BOILERPLATE = re.compile(
    r"(share price|stock price|live nse|live bse|price today|share price today|"
    r"outlook for the week|stock analysis|technical view|market live|"
    r"stocks to watch|buy or sell|target price)",
    re.I,
)

# Exchange filings that carry no price information. Scoring sentiment on these
# only adds noise, so they are dropped before sentiment analysis.
ROUTINE_FILING = re.compile(
    r"(newspaper publication|investor presentation|earnings call transcript|"
    r"transcript|shareholding pattern|compliance certificate|"
    r"trading window|regulation 30|regulation 7|reconciliation of share capital|"
    r"certificate under|annual report|agm|postal ballot|scrutinizer|"
    r"disclosure under|clarification|general updates|analysts/institutional investor meet)",
    re.I,
)

# ---------------------------------------------------------------- lexicon
POS_WORDS = {
    "surge": 0.7, "surges": 0.7, "soar": 0.8, "soars": 0.8, "jump": 0.6, "jumps": 0.6,
    "rally": 0.6, "rallies": 0.6, "gain": 0.5, "gains": 0.5, "rise": 0.4, "rises": 0.4,
    "record": 0.4, "beats": 0.7, "beat": 0.7, "upgrade": 0.7, "upgrades": 0.7,
    "outperform": 0.6, "buy": 0.4, "positive": 0.4, "strong": 0.5, "robust": 0.5,
    "growth": 0.4, "profit": 0.4, "profits": 0.4, "expansion": 0.4, "expand": 0.4,
    "wins": 0.6, "win": 0.6, "order": 0.3, "contract": 0.3, "approval": 0.4,
    "approved": 0.4, "dividend": 0.4, "bonus": 0.4, "buyback": 0.5, "acquisition": 0.3,
    "partnership": 0.3, "deal": 0.3, "launch": 0.3, "higher": 0.4, "up": 0.3,
    "tops": 0.5, "multibagger": 0.5, "bullish": 0.6, "breakout": 0.5, "new high": 0.6,
}
NEG_WORDS = {
    "fall": 0.5, "falls": 0.5, "drop": 0.5, "drops": 0.5, "slump": 0.6, "slumps": 0.6,
    "plunge": 0.8, "plunges": 0.8, "crash": 0.9, "crashes": 0.9, "decline": 0.5,
    "declines": 0.5, "loss": 0.6, "losses": 0.6, "miss": 0.6, "misses": 0.6,
    "downgrade": 0.8, "downgrades": 0.8, "underperform": 0.6, "sell": 0.4,
    "weak": 0.5, "weakness": 0.5, "concern": 0.4, "concerns": 0.4, "probe": 0.7,
    "penalty": 0.7, "fine": 0.6, "fraud": 0.9, "scam": 0.9, "default": 0.8,
    "resignation": 0.6, "resigns": 0.6, "layoff": 0.6, "layoffs": 0.6, "warning": 0.5,
    "cut": 0.5, "cuts": 0.5, "slide": 0.5, "slides": 0.5, "tumble": 0.7, "tumbles": 0.7,
    "bearish": 0.6, "selloff": 0.7, "sell-off": 0.7, "lower": 0.4, "down": 0.3,
    "below": 0.3, "worst": 0.6, "lawsuit": 0.6, "disruption": 0.5, "recall": 0.6,
}
NEGATORS = {"not", "no", "never", "without", "fails", "failed", "unlikely", "denies", "denied"}


class LexiconScorer:
    name = "lexicon"

    def score(self, texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            low = f" {t.lower()} "
            words = re.findall(r"[a-z\-]+", low)
            s, hits = 0.0, 0
            for i, w in enumerate(words):
                val = POS_WORDS.get(w, 0.0) - NEG_WORDS.get(w, 0.0)
                if val == 0:
                    continue
                prev = words[max(0, i - 3):i]
                if any(p in NEGATORS for p in prev):
                    val = -val
                s += val
                hits += 1
            out.append(float(np.tanh(s / max(1.5, hits))))
        return np.array(out, dtype=float)


class FinBertScorer:
    name = "finbert"

    def __init__(self) -> None:
        from transformers import pipeline

        self._pipe = pipeline(
            "sentiment-analysis", model=C.SENTIMENT_MODEL,
            truncation=True, max_length=64, device=-1,
        )

    def score(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([], dtype=float)
        raw = self._pipe([t[:400] for t in texts], batch_size=16)
        out = []
        for r in raw:
            label = str(r["label"]).lower()
            p = float(r["score"])
            if "pos" in label:
                out.append(p)
            elif "neg" in label:
                out.append(-p)
            else:
                out.append(0.0)
        return np.array(out, dtype=float)


_SCORER = None


def get_scorer():
    global _SCORER
    if _SCORER is not None:
        return _SCORER
    try:
        _SCORER = FinBertScorer()
    except Exception:
        _SCORER = LexiconScorer()
    return _SCORER


# ---------------------------------------------------------------- sources
def _nse_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    try:
        s.get("https://www.nseindia.com", timeout=20)
        s.get("https://www.nseindia.com/market-data/live-equity-market", timeout=20)
    except Exception:
        pass
    return s


def nse_announcements(days: int = 7, log=print) -> pd.DataFrame:
    """All equity corporate announcements in the window (one call)."""
    today = date.today()
    frm = today - timedelta(days=days)
    url = (
        "https://www.nseindia.com/api/corporate-announcements"
        f"?index=equities&from_date={frm:%d-%m-%Y}&to_date={today:%d-%m-%Y}"
    )
    try:
        resp = _nse_session().get(url, timeout=30)
        if resp.status_code != 200:
            return pd.DataFrame()
        data = resp.json()
    except Exception:
        return pd.DataFrame()
    if not isinstance(data, list):
        return pd.DataFrame()

    rows = []
    for item in data:
        sym = str(item.get("symbol", "")).strip()
        if not sym:
            continue
        subj = str(item.get("desc") or item.get("subject") or "").strip()
        body = str(item.get("attchmntText") or "").strip()
        txt = " ".join(x for x in (subj, body) if x)[:500]
        rows.append({
            "symbol": sym,
            "published": pd.to_datetime(item.get("an_dt"), format="%d-%b-%Y %H:%M:%S", errors="coerce"),
            "headline": txt or subj,
            "source": "NSE Filing",
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.dropna(subset=["published"])
    # Drop routine filings; keep material ones (results, orders, ratings, penalties...).
    df = df[~df["headline"].str.contains(ROUTINE_FILING, na=False)]
    return df


def google_news(query: str, days: int = 14, max_items: int = 20) -> list[dict]:
    q = requests.utils.quote(f"{query} when:{days}d")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            return []
        root = ET.fromstring(resp.content)
    except Exception:
        return []
    out = []
    for item in root.findall(".//item")[: max_items * 2]:
        title = html.unescape((item.findtext("title") or "").strip())
        if not title or BOILERPLATE.search(title):
            continue
        pub = item.findtext("pubDate")
        when = pd.to_datetime(pub, errors="coerce", utc=True) if pub else pd.NaT
        out.append({"headline": title, "published": when, "source": "Google News"})
    return out[:max_items]


def market_headlines(log=print) -> pd.DataFrame:
    rows = []
    for name, url in C.RSS_FEEDS.items():
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            root = ET.fromstring(resp.content)
            for item in root.findall(".//item")[:60]:
                title = html.unescape((item.findtext("title") or "").strip())
                if not title:
                    continue
                rows.append({
                    "headline": title,
                    "published": pd.to_datetime(item.findtext("pubDate"), errors="coerce", utc=True),
                    "source": name,
                })
        except Exception:
            continue
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- scoring
_EVENT_WORDS = {
    "results": 0.15, "q1": 0.1, "q2": 0.1, "q3": 0.1, "q4": 0.1,
    "board meeting": 0.05, "dividend": 0.1, "bonus": 0.1, "split": 0.05,
    "order": 0.15, "contract": 0.15, "acquisition": 0.1, "merger": 0.1,
    "rating": 0.1, "upgrade": 0.2, "downgrade": -0.2, "guidance": 0.1,
    "probe": -0.25, "penalty": -0.25, "fraud": -0.4, "resignation": -0.2,
}


def score_news(news: pd.DataFrame, asof: pd.Timestamp | None = None,
               scorer=None) -> pd.DataFrame:
    """Aggregate headlines per symbol into a 0-100 news score."""
    if news is None or news.empty:
        return pd.DataFrame(columns=["symbol", "s_news", "news_raw", "n_news",
                                     "sent_mean", "news_top"])
    asof = asof or pd.Timestamp.now(tz="UTC")
    scorer = scorer or get_scorer()

    df = news.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["published"]):
        df["published"] = pd.to_datetime(df["published"], errors="coerce")
    try:
        df["published"] = pd.to_datetime(df["published"], utc=True, errors="coerce")
    except Exception:
        pass
    df = df.dropna(subset=["published"])
    if df.empty:
        return pd.DataFrame(columns=["symbol", "s_news", "news_raw", "n_news",
                                     "sent_mean", "news_top"])

    texts = df["headline"].astype(str).tolist()
    hashes = [store.hash_headline(t) for t in texts]
    cached = store.sentiment_lookup(texts)
    scores = np.array([cached.get(h, np.nan) for h in hashes], dtype=float)
    missing = np.isnan(scores)
    if missing.any():
        idx = np.where(missing)[0]
        fresh = scorer.score([texts[i] for i in idx])
        scores[idx] = fresh
        store.sentiment_store(
            [(texts[i], float(scores[i])) for i in idx], getattr(scorer, "name", "unknown")
        )
    df["sent"] = scores
    age_days = (asof - df["published"]).dt.total_seconds() / 86400.0
    df["age_days"] = age_days.clip(lower=0)
    df["recency_w"] = 0.5 ** (df["age_days"] / 7.0)

    def event_adj(text: str) -> float:
        low = text.lower()
        return sum(v for k, v in _EVENT_WORDS.items() if k in low)

    df["event"] = df["headline"].map(event_adj)

    rows = []
    for sym, g in df.groupby("symbol"):
        w = g["recency_w"].to_numpy(dtype=float)
        s = g["sent"].to_numpy(dtype=float)
        raw = float((s * w).sum() / max(w.sum(), 1e-9))
        # Major corporate events tilt the read; cap so a single filing cannot dominate.
        raw = float(np.clip(raw + float(g["event"].mean()) * 0.5, -1, 1))
        top = g.sort_values(["recency_w", "sent"], ascending=False).iloc[0]["headline"]
        rows.append({
            "symbol": sym,
            "news_raw": round(raw, 3),
            "s_news": round((raw + 1.0) * 50.0, 2),
            "n_news": int(len(g)),
            "sent_mean": round(float(s.mean()), 3),
            "news_top": str(top)[:180],
        })
    return pd.DataFrame(rows)


def refresh(symbols: list[str], names: dict[str, str], days: int | None = None,
            log=print) -> pd.DataFrame:
    """Pull NSE filings for everything plus Google News for the given symbols."""
    days = days or C.NEWS_LOOKBACK_DAYS
    frames = []

    ann = nse_announcements(days=days, log=log)
    if not ann.empty:
        ann = ann[ann["symbol"].isin(set(symbols)) | True]
        frames.append(ann[["symbol", "published", "headline", "source"]])
        log(f"  NSE filings: {len(ann)}")
    else:
        log("  NSE filings unavailable")

    for i, sym in enumerate(symbols):
        q = names.get(sym, sym)
        items = google_news(f'"{q}" stock', days=days, max_items=C.NEWS_MAX_PER_SYMBOL)
        for it in items:
            it["symbol"] = sym
        if items:
            frames.append(pd.DataFrame(items)[["symbol", "published", "headline", "source"]])
        if (i + 1) % 10 == 0:
            log(f"  google news {i + 1}/{len(symbols)}")
        time.sleep(0.4)

    if not frames:
        return pd.DataFrame()
    news = pd.concat(frames, ignore_index=True)
    news = news.drop_duplicates(subset=["symbol", "headline"])
    store.save_table("news", news)
    return news
