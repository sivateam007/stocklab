"""Central configuration. Every tunable number in the engine lives here."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PRICE_DIR = DATA_DIR / "prices"
REPORT_DIR = ROOT / "reports"
CACHE_DIR = DATA_DIR / "cache"

for _d in (DATA_DIR, PRICE_DIR, REPORT_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

UNIVERSE_CSV = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
BENCHMARK = "^NSEI"
VIX = "^INDIAVIX"

# ---------------------------------------------------------------- data ingest
PRICE_PERIOD = "5y"
BATCH_SIZE = 40
DOWNLOAD_WORKERS = 5
RETRIES = 3

# ---------------------------------------------------------------- scoring
# Weights must sum to 1.0.
#
# The technical sub-weights follow two-half IC evidence (see README): only
# factors whose IC sign survives both halves of history at both the 20d and
# 40d horizons are allowed to vote, blocks carrying the stronger stable
# factors get the larger share, and the split itself is derived from
# first-half block ICs only (experiments/block_weight_schemes.py) so the
# second half stays a genuine out-of-sample test.
@dataclass(frozen=True)
class Weights:
    # technical block = 0.60
    trend: float = 0.11
    momentum: float = 0.15
    breakout: float = 0.07
    volume: float = 0.18
    rel_strength: float = 0.09
    # other blocks
    news: float = 0.20
    fundamental: float = 0.20

    def as_dict(self) -> dict[str, float]:
        return {
            "trend": self.trend,
            "momentum": self.momentum,
            "breakout": self.breakout,
            "volume": self.volume,
            "rel_strength": self.rel_strength,
            "news": self.news,
            "fundamental": self.fundamental,
        }


WEIGHTS = Weights()

# ---------------------------------------------------------------- filters
@dataclass(frozen=True)
class Filters:
    min_turnover_cr: float = 5.0        # 20d avg traded value, crore rupees
    min_price: float = 20.0
    min_history_days: int = 250
    max_atr_pct: float = 8.0            # skip wildly volatile names
    regime_vix_max: float = 22.0
    regime_breadth_min: float = 40.0    # % of universe above 50-DMA


FILTERS = Filters()

# ---------------------------------------------------------------- trading rules
@dataclass(frozen=True)
class TradeRules:
    hold_days_max: int = 40
    atr_stop_mult: float = 1.5
    target_r_multiple: float = 2.5
    risk_per_trade_pct: float = 1.0
    max_positions: int = 10


RULES = TradeRules()

# ---------------------------------------------------------------- costs (Indian delivery equity)
@dataclass(frozen=True)
class Costs:
    stt_buy: float = 0.001          # 0.1%
    stt_sell: float = 0.001         # 0.1%
    exchange_pct: float = 0.0000325  # NSE 0.00325%
    sebi_pct: float = 0.000001
    stamp_buy: float = 0.00015      # 0.015% buy side only
    dp_per_sell: float = 15.34      # per sell transaction
    brokerage_per_order: float = 0.0  # discount broker delivery
    gst: float = 0.18
    slippage_pct: float = 0.0015
    stcg_tax: float = 0.20

    def round_trip_pct(self) -> float:
        """Approximate all-in friction for one buy+sell, as fraction of value."""
        statutory = (
            self.stt_buy
            + self.stt_sell
            + 2 * self.exchange_pct
            + 2 * self.sebi_pct
            + self.stamp_buy
        )
        gst = self.gst * (2 * self.exchange_pct + 2 * self.sebi_pct)
        return statutory + gst + 2 * self.slippage_pct


COSTS = Costs()

# ---------------------------------------------------------------- news
NEWS_LOOKBACK_DAYS = 14
NEWS_MAX_PER_SYMBOL = 25
SENTIMENT_MODEL = "ProsusAI/finbert"
SENTIMENT_FALLBACK = True

RSS_FEEDS = {
    "ET Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "Moneycontrol": "https://www.moneycontrol.com/rss/business.xml",
}

DISCLAIMER = (
    "Research tool only. Not investment advice. Not a SEBI-registered "
    "research analyst or investment adviser. Past performance does not "
    "guarantee future returns. Verify every number independently."
)
