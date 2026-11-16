"""Nifty 500 universe: download constituents, map to sectors and Yahoo tickers."""

from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
import requests

from . import config as C
from . import store

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# NSE industry -> broad sector, used for sector-relative comparison.
_BROAD = {
    "Financial Services": "Financials",
    "Information Technology": "IT",
    "Oil Gas & Consumable Fuels": "Energy",
    "Fast Moving Consumer Goods": "FMCG",
    "Automobile and Auto Components": "Auto",
    "Healthcare": "Healthcare",
    "Capital Goods": "Industrials",
    "Construction": "Industrials",
    "Construction Materials": "Materials",
    "Consumer Durables": "Consumer",
    "Consumer Services": "Consumer",
    "Power": "Utilities",
    "Utilities": "Utilities",
    "Realty": "Realty",
    "Telecommunication": "Telecom",
    "Metals & Mining": "Materials",
    "Chemicals": "Materials",
    "Cement & Cement Products": "Materials",
    "Fertilizers & Agrochemicals": "Materials",
    "Forest Materials": "Materials",
    "Textiles": "Materials",
    "Media Entertainment & Publication": "Media",
    "Services": "Services",
    "Diversified": "Diversified",
}


def broad_sector(industry: str) -> str:
    if not isinstance(industry, str):
        return "Other"
    return _BROAD.get(industry.strip(), industry.strip() or "Other")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop placeholder rows that appear in NSE's index CSV.

    Example: "Dummy HEG Ltd." with ISIN DUM545A01024 — no listing, no prices,
    and it pollutes the universe with a permanent download failure. Genuine
    Indian ISINs always start with "IN".
    """
    if df.empty:
        return df
    isin = df.get("isin", pd.Series("", index=df.index)).astype(str)
    name = df.get("name", pd.Series("", index=df.index)).astype(str)
    bad = (
        isin.str.strip().str.upper().str.startswith(("DUM", "TEST"))
        | name.str.contains(r"\bdummy\b", case=False, na=False)
        | ~isin.str.strip().str.upper().str.startswith("IN")
    )
    # Rows with a blank ISIN are kept only if the name looks real.
    blank = isin.str.strip() == ""
    return df[~(bad & ~blank)].reset_index(drop=True)


def fetch_universe(force: bool = False) -> pd.DataFrame:
    """Return DataFrame[symbol, name, industry, sector, isin, yahoo, series]."""
    cached = store.load_table("universe")
    if not cached.empty and not force:
        cleaned = _clean(cached)
        if len(cleaned) != len(cached):
            store.save_table("universe", cleaned)
        return cleaned

    resp = requests.get(C.UNIVERSE_CSV, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = [c.strip() for c in df.columns]

    rename = {
        "Company Name": "name",
        "Industry": "industry",
        "Symbol": "symbol",
        "Series": "series",
        "ISIN Code": "isin",
    }
    df = df.rename(columns=rename)
    for col in ("name", "industry", "symbol", "series", "isin"):
        if col not in df.columns:
            df[col] = ""

    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    df["industry"] = df["industry"].astype(str).str.strip()
    df["sector"] = df["industry"].map(broad_sector)
    df["yahoo"] = df["symbol"] + ".NS"
    df = df[df["series"].isin(["EQ", "BE", "BZ"]) | (df["series"] == "")].copy()
    df = df.drop_duplicates("symbol").reset_index(drop=True)
    df["updated_at"] = datetime.now().isoformat(timespec="seconds")

    store.save_table("universe", df)
    return df


def universe_symbols(force: bool = False) -> list[str]:
    return fetch_universe(force=force)["symbol"].tolist()


def sector_map() -> dict[str, str]:
    u = fetch_universe()
    return dict(zip(u["symbol"], u["sector"]))


def name_map() -> dict[str, str]:
    u = fetch_universe()
    return dict(zip(u["symbol"], u["name"]))
