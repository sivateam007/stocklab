"""Streamlit dashboard:  streamlit run app/dashboard.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from stocklab import backtest, config as C, news as news_mod, prices, scoring, signals, store, technical

st.set_page_config(page_title="Stocklab - Nifty 500", layout="wide",
                   page_icon="📈", initial_sidebar_state="expanded")

UP, DOWN = "#0f9d58", "#d93025"


# ------------------------------------------------------------------ loaders
@st.cache_data(ttl=300, show_spinner=False)
def load_features(columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    return store.load_table("features", columns=list(columns) if columns else None)


@st.cache_data(ttl=300, show_spinner=False)
def load_screen() -> pd.DataFrame:
    return store.load_table("screen_latest")


@st.cache_data(ttl=300, show_spinner=False)
def load_universe() -> pd.DataFrame:
    return store.load_table("universe")


@st.cache_data(ttl=300, show_spinner=False)
def load_news() -> pd.DataFrame:
    return store.load_table("news")


@st.cache_data(ttl=300, show_spinner=False)
def load_prices(symbol: str) -> pd.DataFrame:
    return prices.adjusted(store.load_prices(symbol))


@st.cache_data(ttl=600, show_spinner=False)
def load_market() -> dict:
    return technical.market_state()


# ------------------------------------------------------------------ charts
def candlestick_panel(df: pd.DataFrame, symbol: str, name: str = "") -> go.Figure:
    d = df.tail(260)
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.60, 0.18, 0.22],
        subplot_titles=(f"{symbol} {name}", "RSI(14)", "Volume"),
    )
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
        increasing_line_color=UP, decreasing_line_color=DOWN, name="price",
    ), row=1, col=1)

    for period, colour in ((20, "#1a73e8"), (50, "#f9ab00"), (200, "#9334e6")):
        ema = d["close"].ewm(span=period, adjust=False).mean()
        fig.add_trace(go.Scatter(x=d.index, y=ema, name=f"EMA{period}",
                                 line=dict(color=colour, width=1)), row=1, col=1)

    # supertrend from the raw feature computation
    raw = store.load_prices(symbol)
    st_df = technical.supertrend(prices.adjusted(raw)).tail(260)
    fig.add_trace(go.Scatter(x=st_df.index, y=st_df["st_line"], name="Supertrend",
                             line=dict(color="#7cb342", width=1, dash="dot")), row=1, col=1)

    delta = d["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    fig.add_trace(go.Scatter(x=d.index, y=rsi, name="RSI",
                             line=dict(color="#5f6368", width=1)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#d93025", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#0f9d58", row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    colours = [UP if c >= o else DOWN for o, c in zip(d["open"], d["close"])]
    fig.add_trace(go.Bar(x=d.index, y=d["volume"], marker_color=colours, name="volume"),
                  row=3, col=1)

    fig.update_layout(height=720, template="plotly_white", showlegend=True,
                      margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_rangeslider_visible=False, hovermode="x unified")
    return fig


def breadth_chart(feat: pd.DataFrame) -> go.Figure:
    g = feat.groupby("date")
    above50 = g.apply(lambda x: (x["close"] > x["sma50"]).mean() * 100, include_groups=False)
    above200 = g.apply(lambda x: (x["close"] > x["sma200"]).mean() * 100, include_groups=False)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=above50.index, y=above50, name="% above 50-DMA",
                             line=dict(color="#1a73e8")))
    fig.add_trace(go.Scatter(x=above200.index, y=above200, name="% above 200-DMA",
                             line=dict(color="#9334e6")))
    fig.add_hline(y=50, line_dash="dash", line_color="#999")
    fig.update_layout(height=340, template="plotly_white", margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_title="% of universe", hovermode="x unified")
    return fig


# ------------------------------------------------------------------ pages
def page_screener() -> None:
    df = load_screen()
    if df.empty:
        st.warning("No screen yet. Run `python -m stocklab screen` first.")
        return

    st.subheader("Short-term candidates")
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1])
    sectors = ["All"] + sorted(df["sector"].dropna().unique().tolist())
    sector = c1.selectbox("Sector", sectors)
    min_score = c2.slider("Min score", 0, 100, 55)
    only_eligible = c3.checkbox("Liquid only", value=True)
    capital = c4.number_input("Capital (Rs)", 10_000, 100_000_000, 100_000, step=10_000)

    view = df.copy()
    if sector != "All":
        view = view[view["sector"] == sector]
    if only_eligible:
        view = view[view["eligible"]]
    view = view[view["total"] >= min_score]

    plans = signals.attach_plans(view, capital=float(capital), top=50)

    show = ["rank", "symbol", "name", "sector", "close", "total", "s_technical",
            "s_news", "s_fundamental", "s_results", "atr_pct", "turnover_sma20",
            "entry", "stop", "target", "qty", "why"]
    show = [c for c in show if c in plans.columns]
    st.dataframe(
        plans[show], use_container_width=True, height=560, hide_index=True,
        column_config={
            "total": st.column_config.ProgressColumn("score", min_value=0, max_value=100, format="%.1f"),
            "s_technical": st.column_config.NumberColumn("tech", format="%.0f"),
            "s_news": st.column_config.NumberColumn("news", format="%.0f"),
            "s_fundamental": st.column_config.NumberColumn("fund", format="%.0f"),
            "s_results": st.column_config.NumberColumn("results", format="%.0f"),
            "close": st.column_config.NumberColumn("CMP", format="%.2f"),
            "entry": st.column_config.NumberColumn("entry", format="%.2f"),
            "stop": st.column_config.NumberColumn("stop", format="%.2f"),
            "target": st.column_config.NumberColumn("target", format="%.2f"),
            "qty": st.column_config.NumberColumn("qty", format="%d"),
            "atr_pct": st.column_config.NumberColumn("ATR%", format="%.1f"),
            "turnover_sma20": st.column_config.NumberColumn("turnover Cr", format="%.0f"),
        },
    )

    summary = signals.position_summary(plans)
    m1, m2, m3 = st.columns(3)
    m1.metric("Positions", summary["positions"])
    m2.metric("Capital deployed", f"Rs {summary['capital_deployed']:,.0f}")
    m3.metric("Max loss at stops", f"Rs {summary['max_loss']:,.0f}")


def page_stock() -> None:
    df = load_screen()
    if df.empty:
        st.warning("No screen data.")
        return
    uni = load_universe()
    names = dict(zip(uni["symbol"], uni["name"])) if not uni.empty else {}
    options = df["symbol"].tolist()
    symbol = st.selectbox("Stock", options, index=0)
    row = df[df["symbol"] == symbol].iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Score", f"{row['total']:.1f}")
    c2.metric("Close", f"{row['close']:,.2f}")
    c3.metric("RSI", f"{row.get('rsi14', 0):.0f}")
    c4.metric("ADX", f"{row.get('adx14', 0):.0f}")
    c5.metric("ATR %", f"{row.get('atr_pct', 0):.1f}%")

    px = load_prices(symbol)
    if not px.empty:
        st.plotly_chart(candlestick_panel(px, symbol, names.get(symbol, "")),
                        use_container_width=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Trade plan**")
        plan = signals.plan_trade(row, capital=100_000)
        st.table(pd.DataFrame([plan]).T.rename(columns={0: "value"}))
        if row.get("why"):
            st.info(row["why"])
    with col2:
        st.markdown("**Score breakdown**")
        parts = {k: row.get(k) for k in
                 ("s_trend", "s_momentum", "s_breakout", "s_volume", "s_rel_strength",
                  "s_news", "s_fundamental", "s_results")}
        parts = {k: v for k, v in parts.items() if pd.notna(v)}
        if parts:
            st.plotly_chart(
                go.Figure(go.Bar(x=list(parts.values()), y=list(parts.keys()),
                                 orientation="h", marker_color="#1a73e8")),
                use_container_width=True, config={"displayModeBar": False},
            )
        fund = store.load_table("fundamentals")
        if not fund.empty:
            f = fund[fund["symbol"] == symbol]
            if not f.empty:
                f = f.iloc[0]
                st.markdown("**Fundamentals**")
                st.write({
                    "PE": f.get("trailingPE"), "P/B": f.get("priceToBook"),
                    "ROE %": f.get("returnOnEquity"), "D/E": f.get("debtToEquity"),
                    "Rev growth %": f.get("revenueGrowth_pct"),
                    "PAT growth %": f.get("earningsGrowth_pct"),
                })

    news = load_news()
    if not news.empty:
        n = news[news["symbol"] == symbol].sort_values("published", ascending=False).head(15)
        if not n.empty:
            st.markdown("**Recent headlines**")
            st.dataframe(n[["published", "headline", "source"]],
                         use_container_width=True, hide_index=True)


def page_market() -> None:
    m = load_market()
    feat = load_features(("date", "close", "sma50", "sma200", "s_technical", "sector"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nifty", m.get("nifty_close"), f"{m.get('nifty_vs_200dma_pct')}% vs 200DMA")
    c2.metric("India VIX", m.get("vix"))
    c3.metric("Breadth > 50DMA", f"{m.get('breadth_50dma')}%")
    c4.metric("Regime", str(m.get("regime")).upper())

    if m.get("regime") == "risk-on":
        st.success("Risk-on: long setups are being taken.")
    else:
        st.error(f"Defensive: no new longs. {', '.join(m.get('reasons', []))}")

    if not feat.empty:
        st.plotly_chart(breadth_chart(feat), use_container_width=True)

        last = feat[feat["date"] == feat["date"].max()]
        uni = load_universe()
        if not uni.empty and "sector" in last:
            sec = (last.groupby("sector")["s_technical"].mean()
                   .sort_values(ascending=False).reset_index())
            fig = go.Figure(go.Bar(x=sec["s_technical"], y=sec["sector"],
                                   orientation="h", marker_color="#1a73e8"))
            fig.update_layout(title="Sector strength (avg technical score)",
                              height=420, template="plotly_white",
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)


def page_backtest() -> None:
    st.subheader("Validate the strategy")
    c1, c2, c3 = st.columns(3)
    top = c1.slider("Top N", 3, 25, 10)
    hold = c2.slider("Max hold days", 5, 90, 40)
    atr = c3.slider("ATR stop multiple", 0.5, 4.0, 1.5, 0.1)

    if st.button("Run backtest", type="primary"):
        with st.spinner("Running..."):
            res = backtest.run(backtest.BTParams(top_n=top, hold_days=hold, atr_mult=atr))
        if "error" in res:
            st.error(res["error"])
            return
        st.session_state["bt"] = {k: v for k, v in res.items() if k not in ("equity", "trades")}
        st.session_state["bt_equity"] = res["equity"]
        st.session_state["bt_trades"] = res["trades"]

    if "bt" in st.session_state:
        r = st.session_state["bt"]
        a, b, c, d = st.columns(4)
        a.metric("CAGR", f"{r['cagr_pct']}%")
        b.metric("Sharpe", r["sharpe"])
        c.metric("Max DD", f"{r['max_drawdown_pct']}%")
        d.metric("Win rate", f"{r['win_rate_pct']}%")
        e, f, g, h = st.columns(4)
        e.metric("Trades", r["n_trades"])
        f.metric("Profit factor", r["profit_factor"])
        g.metric("Expectancy", f"Rs {r['expectancy']:,}" if r["expectancy"] else "-")
        h.metric("Costs paid", f"Rs {r['costs_paid']:,.0f}")

        eq = st.session_state["bt_equity"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq.index, y=eq["equity"], name="equity",
                                 line=dict(color="#1a73e8", width=1.5)))
        fig.update_layout(height=380, template="plotly_white", title="Equity curve",
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

        tr = st.session_state["bt_trades"]
        if not tr.empty:
            st.markdown("**Trades**")
            st.dataframe(tr.sort_values("exit_date", ascending=False),
                         use_container_width=True, height=320, hide_index=True)

        st.caption(
            "History uses the technical composite only: fundamentals and news are "
            "current snapshots and would leak the future. Costs include STT, stamp "
            "duty, DP charges, GST, slippage and 20% STCG. Survivorship bias remains."
        )


def page_news() -> None:
    news_df = load_news()
    if news_df.empty:
        st.warning("No news cached. Run `python -m stocklab screen --news-backend finbert`.")
        return
    scored = news_mod.score_news(news_df, scorer=news_mod.LexiconScorer())
    c1, c2 = st.columns([1, 2])
    c1.metric("Headlines", len(news_df))
    c2.metric("Symbols covered", len(scored))

    st.dataframe(scored.sort_values("s_news", ascending=False).head(40),
                 use_container_width=True, height=420, hide_index=True)

    st.markdown("**All headlines**")
    sym = st.selectbox("Filter symbol", ["All"] + sorted(news_df["symbol"].unique().tolist()))
    view = news_df if sym == "All" else news_df[news_df["symbol"] == sym]
    st.dataframe(view.sort_values("published", ascending=False).head(200),
                 use_container_width=True, height=340, hide_index=True)


# ------------------------------------------------------------------ shell
def main() -> None:
    st.sidebar.title("Stocklab")
    m = load_market()
    badge = "🟢 RISK-ON" if m.get("regime") == "risk-on" else "🔴 DEFENSIVE"
    st.sidebar.markdown(f"### {badge}")
    st.sidebar.caption(f"As of {m.get('as_of')} · VIX {m.get('vix')} · "
                       f"breadth {m.get('breadth_50dma')}%")

    page = st.sidebar.radio("Page", ["Screener", "Stock detail", "Market", "News", "Backtest"])
    st.sidebar.divider()
    st.sidebar.caption(C.DISCLAIMER)

    {"Screener": page_screener, "Stock detail": page_stock, "Market": page_market,
     "News": page_news, "Backtest": page_backtest}[page]()


if __name__ == "__main__":
    main()
