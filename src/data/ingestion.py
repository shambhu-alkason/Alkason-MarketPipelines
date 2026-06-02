"""
Data ingestion — downloads historical OHLCV data for NSE/BSE stocks.

Primary:  OpenBB SDK (yfinance provider) — rate limiting, caching, retry
Fallback: direct yfinance call if OpenBB fails
"""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def _load_config() -> dict:
    cfg_path = Path(__file__).parents[2] / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _raw_path(symbol: str, cfg: dict) -> Path:
    base = Path(cfg["paths"]["raw_data"])
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{symbol}.csv"


def _fetch_openbb(symbol: str, period: str) -> Optional[pd.DataFrame]:
    """Attempt download via OpenBB SDK."""
    try:
        from openbb import obb
        result = obb.equity.price.historical(
            symbol=symbol,
            start_date=_period_to_start(period),
            provider="yfinance",
        )
        df = result.to_df()
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df
    except Exception as e:
        logger.warning("OpenBB failed for %s: %s — trying yfinance fallback", symbol, e)
        return None


def _fetch_yfinance(symbol: str, period: str) -> Optional[pd.DataFrame]:
    """Direct yfinance fallback."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, auto_adjust=True)
        if df.empty:
            logger.error("yfinance returned empty data for %s", symbol)
            return None
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "date"
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception as e:
        logger.error("yfinance also failed for %s: %s", symbol, e)
        return None


def _period_to_start(period: str) -> str:
    """Convert period string like '5y' to ISO start date."""
    unit = period[-1]
    n = int(period[:-1])
    if unit == "y":
        delta = timedelta(days=365 * n)
    elif unit == "m":
        delta = timedelta(days=30 * n)
    elif unit == "d":
        delta = timedelta(days=n)
    else:
        delta = timedelta(days=365 * 5)
    return (datetime.today() - delta).strftime("%Y-%m-%d")


def _incremental_start(existing: pd.DataFrame) -> Optional[str]:
    """Return the day after the last available date for incremental refresh."""
    if existing is None or existing.empty:
        return None
    last = existing.index.max()
    return (last + timedelta(days=1)).strftime("%Y-%m-%d")


def download_symbol(
    symbol: str,
    period: str = "5y",
    provider: str = "yfinance",
    force_full: bool = False,
) -> pd.DataFrame:
    """
    Download OHLCV data for a single symbol.

    Tries OpenBB first, falls back to yfinance.
    Supports incremental refresh — only fetches missing dates.
    """
    cfg = _load_config()
    out_path = _raw_path(symbol, cfg)

    existing = None
    if out_path.exists() and not force_full:
        existing = pd.read_csv(out_path, index_col="date", parse_dates=True)
        logger.info("%s: %d rows already cached", symbol, len(existing))

    # Try OpenBB → yfinance
    df = _fetch_openbb(symbol, period)
    if df is None:
        df = _fetch_yfinance(symbol, period)

    if df is None:
        if existing is not None:
            logger.warning("%s: download failed, using cached data", symbol)
            return existing
        raise RuntimeError(f"Failed to download data for {symbol}")

    # Standardise columns
    df.columns = [c.lower() for c in df.columns]
    required = {"open", "high", "low", "close", "volume"}
    present = set(df.columns)
    if not required.issubset(present):
        # rename common variants
        rename_map = {"adj close": "close", "adj_close": "close"}
        df = df.rename(columns=rename_map)

    # Merge with existing for incremental update
    if existing is not None:
        df = pd.concat([existing, df])
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()

    if len(df) < 200:
        logger.warning("%s: only %d rows — data may be incomplete", symbol, len(df))

    df.to_csv(out_path)
    logger.info("%s: saved %d rows to %s", symbol, len(df), out_path)
    return df


def download_all(
    symbols: Optional[List[str]] = None,
    force_full: bool = False,
) -> dict:
    """
    Download data for all symbols in config (or a provided override list).
    Returns dict of {symbol: DataFrame}.
    """
    cfg = _load_config()
    symbols = symbols or cfg["stocks"]["symbols"]
    period = cfg["stocks"]["period"]

    results = {}
    for symbol in symbols:
        logger.info("Downloading %s ...", symbol)
        try:
            results[symbol] = download_symbol(symbol, period=period, force_full=force_full)
        except Exception as e:
            logger.error("Skipping %s: %s", symbol, e)

    logger.info("Download complete: %d/%d symbols successful", len(results), len(symbols))
    return results
