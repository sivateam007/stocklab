# Stocklab — Nifty 500 short-term analysis engine

A local research tool that scans the Nifty 500 every evening and ranks stocks by
how likely they are to move up over the next 2–8 weeks, with a concrete entry,
stop-loss, target and position size for each idea.

**Not investment advice.** Not a SEBI-registered research analyst. Nothing here
guarantees profit. Everything is computed from free data and can be wrong.

---

## What it does

```
Nifty 500 ──> prices+volume ──> technical factors ──┐
           └─ fundamentals ──────────────────────────┤──> score /100 ──> ranked list
           └─ NSE filings + news ──> sentiment ──────┤
           └─ earnings calendar ──> drift score ─────┘
                                                       └──> backtest ──> dashboard
```

**Composite score** = technical 60% + news 20% + (fundamentals & results) 20%.
Every block is percentile-ranked 0–100. If a block is missing for a stock the
weight is redistributed rather than scored as zero.

| Block | Weight | Inputs |
|---|---|---|
| Trend | 11% | EMA 20/50/200 stack, ADX14, above 200-EMA, Supertrend direction, *inverted* Minus-DI |
| Momentum | 15% | ROC 120-day, *inverted* MACD-histogram-rising |
| Breakout | 7% | distance to 52-week high |
| Volume | 18% | *inverted* 20-day turnover (dominant), volume vs 20-day average |
| Relative strength | 9% | 120-day ROC minus the stock's **sector** median ROC |
| News | 20% | NSE filings + Google News + ET/Moneycontrol, FinBERT sentiment |
| Fundamentals | 12% | ROE, margins, D/E, growth, PE/PB/PEG — all *sector-relative* |
| Results drift | 8% | price reaction to the last report (EPS surprise only when fresh), decaying over 45 days |

### How the technical factors were chosen

Every raw factor was scored by information coefficient (rank correlation with
the next 20 and 40 sessions' return, liquid names only), split at the midpoint
of history (2024-05-21). A factor may vote in a block only if:

1. its IC **sign survives both halves** of history at **both** horizons, and
2. it reaches |t| ≥ 1.5 somewhere (otherwise it is noise regardless of sign).

Stable-negative factors (`turnover_sma20`, `macd_rising`, `minus_di`) vote
*inverted*. Block weights come from first-half block ICs only
(`experiments/block_weight_schemes.py`), so the second half remains an honest
out-of-sample test. Experiments that killed candidates:

* `rs_N = roc_N − index return` is **rank-identical** to `roc_N` (the index
  return is constant per date), so the old relative-strength block was a
  momentum duplicate — replaced by sector-median-relative 120-day ROC
  (`experiments/rel_strength_variants.py`).
* Sector-relative 60-day momentum failed the rule; only the 120-day variant
  passed (`experiments/sector_rs_ic.py`).

Result of the rework (20-day horizon, liquid names):

| Score | IC before | IC after | IC after, 2nd half |
|---|---|---|---|
| `s_technical` | +0.011 (t 1.3) | **+0.048 (t 6.1)** | +0.036 (t 3.1) |
| `s_volume` | −0.008 | +0.038 (t 5.6) | +0.032 (t 3.1) |

The old `s_volume` was net wrong-signed: the single strongest factor in the
universe (`turnover_sma20`, IC −0.037 at 20d / −0.063 at 40d) was weighted
positively along with two other stably-negative members.

**Regime filter (hard):** no new longs when Nifty is below its 200-DMA, India VIX
is above 22, or fewer than 40% of stocks are above their 50-DMA. Most damage in
short-term trading comes from buying into a falling market.

---

## Install

Everything needed is already present on this machine. For a fresh box:

```powershell
pip install -r requirements.txt
```

## Commands

```powershell
python -m stocklab status      # what data is cached
python -m stocklab ingest      # prices + features + fundamentals + earnings (~5-10 min first time)
python -m stocklab screen      # rank today's opportunities
python -m stocklab backtest --walk-forward
python -m stocklab backtest --sweep
python -m stocklab report      # regenerate the HTML report
python -m stocklab news        # fetch and score headlines

streamlit run app/dashboard.py
```

Daily routine after market close:

```powershell
python -m stocklab ingest ; python -m stocklab screen
```

---

## Web app (local + Render)

The dashboard is the same Streamlit app locally and in the cloud:

```powershell
streamlit run app/dashboard.py      # http://localhost:8501
```

**Deploy it on [Render](https://render.com)** — the repo already contains a
`Dockerfile`, a `render.yaml` Blueprint, `requirements-web.txt` (no torch /
transformers, so the image stays small) and `.streamlit/config.toml`.

1. Create an empty GitHub repository and push this project:
   ```powershell
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
   Note: `data/features.parquet` is ~90 MB (under GitHub's 100 MB limit) —
   the first push is ~120 MB and takes a minute.
2. In Render: **New → Blueprint** → select that repository. Render reads
   `render.yaml`, builds the Docker image and starts the service.
3. Open the URL it gives you (`https://stocklab-<something>.onrender.com`).
   Health checks hit Streamlit's built-in `/_stcore/health`.

Updating the live app = refreshing data locally and pushing:

```powershell
python -m stocklab ingest ; python -m stocklab screen
git add -A ; git commit -m "data update" ; git push
```

`autoDeployTrigger: commit` redeploy automatically on every push.

Known trade-offs of the free plan (0.1 CPU / 512 MB): the service **spins down
after ~15 min idle**, so the first page load takes ~30-60 s; the Backtest page
can take minutes instead of seconds. Move `plan:` in `render.yaml` to
`0.5c-512mb` or `1c-2g` if that bothers you. The deployed app only *reads*
the precomputed data baked into the image — run `ingest`/`screen` on your own
machine to refresh it.

---

## Files produced

| Path | Contents |
|---|---|
| `data/prices/*.parquet` | one file per symbol, 5 years of daily bars |
| `data/features.parquet` | every technical factor for every symbol-day |
| `data/fundamentals.parquet` | company financials snapshot |
| `data/news.parquet` | headlines with timestamps |
| `data/screen_latest.parquet` | last screener output with trade plans |
| `reports/screen_*.html` | shareable daily report |
| `data/stocklab.db` | run log, ingest state, small key/value store |

---

## Backtest honesty rules

The backtester deliberately refuses to flatter itself:

* signal is computed on the **close of day T**, entry fills at the **open of T+1**
* if a bar touches both the stop and the target, the **stop** is assumed
* a gap through the stop fills at the **open**, not at the stop price
* every trade pays STT (0.1% each side), exchange charges, SEBI fee, stamp duty,
  DP charge, GST, 0.15% slippage per side, and **20% STCG** on gains
* only the **technical** composite is used historically — news and fundamentals
  are current snapshots and would leak the future into a past test
* `--walk-forward` splits the history in half and reports the second half
  separately; that is the only number worth trusting

Known biases that are **not** removed:

1. **Survivorship** — today's Nifty 500 excludes companies that were dropped or
   delisted, so losers are underrepresented.
2. **Fundamental look-ahead** — live screens use latest-reported financials,
   which were not knowable at the time of an old price. The backtest therefore
   excludes fundamentals entirely.
3. **Yahoo data quality** — occasionally stale or mis-scaled. Quality gates
   quarantine obvious breakage but cannot catch everything.

### What the walk-forward actually shows (2021-11 → 2026-11, default params)

| | In-sample (1st half) | Out-of-sample (2nd half) |
|---|---|---|
| CAGR | 30.3% | **2.7%** |
| Sharpe | 1.66 | **0.25** |
| vs Nifty | +69.2% | **−5.1%** |
| Win rate | 51.8% | 38.5% |

The signal ranks stocks reliably (composite IC +0.048, OOS-half t = 3.1) but
the **default trading rules do not convert that into benchmark-beating money
out-of-sample**. The parameter study (`python -m stocklab study`) finds 42 of
144 sets profitable in both halves; the best stable set (top 15, hold 60d,
ATR stop 1.0×, no trail) does 27.9% IS / **9.3% OOS, Sharpe 0.62**. Treat that
as the upper bound of what this engine has demonstrated, not as a promise.

---

## Trading rules the plans assume

| Rule | Value |
|---|---|
| Max holding period | 40 sessions |
| Stop | max(1.5 × ATR14, 10-day swing low) |
| Target | 2.5 × risk |
| Risk per trade | 1% of capital |
| Max concurrent positions | 10 |
| Position cap | 25% of capital |
| Liquidity floor | ₹5 crore average daily turnover |

All of these live in `stocklab/config.py`.

---

## Limitations, stated plainly

* Headline sentiment misreads Indian English idioms and is capped at 20% of the
  score for that reason.
* Delivery percentage and block/bulk deals are not yet included (needs NSE
  bhavcopy); options OI/PCR are not included.
* Yahoo's Indian fundamentals often lag a quarter, so earnings-day reactions are
  captured from price, not from the filing itself.
* Yahoo's EPS *surprise history* (`yfinance.earnings_dates`, which calls
  `/v1/finance/visualization`) is stale for `.NS` tickers — its newest row can
  be more than a year old. The screener therefore anchors `s_results` on the
  fresh earnings *calendar* date plus the price reaction around it, reports
  `surprise_pct` only when it is genuinely ≤60 days old, and leaves the block
  blank otherwise — at the last run that meant 82 of 497 names scored (255 have
  a Yahoo date at all, 85 of those are past reports, 82 had a measurable
  post-report move). Coverage is partial by design — better no vote than a
  stale number pretending to be current.
