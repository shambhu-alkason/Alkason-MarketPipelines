"""
Feature engineering — computes 60+ technical indicators and generates 5-class labels.

Library: pandas-ta-classic (active May 2026 fork; falls back to pandas-ta).
Sentiment features are accepted as input (computed by src/slm/sentiment.py).
Historical dates with no news use neutral imputation: score=0.0, confidence=0.5, count=0, trend=0.0.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Try pandas-ta-classic first, fall back to pandas-ta
try:
    import pandas_ta as ta
    logger.info("Using pandas-ta-classic / pandas-ta")
except ImportError:
    raise ImportError("Install pandas-ta-classic or pandas-ta: pip install pandas-ta-classic")


def _load_config() -> dict:
    cfg_path = Path(__file__).parents[2] / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ── Label Generation ──────────────────────────────────────────────────

def _make_labels(close: pd.Series, cfg: dict) -> pd.Series:
    """5-class label based on next-day forward return."""
    fwd_return = close.shift(-1) / close - 1.0
    thresholds = cfg["labels"]
    sb = thresholds["strong_buy_threshold"]
    b  = thresholds["buy_threshold"]
    s  = thresholds["sell_threshold"]
    ss = thresholds["strong_sell_threshold"]

    label = pd.Series(2, index=close.index, name="label", dtype=int)  # default: Hold
    label[fwd_return > sb]  = 4  # Strong Buy
    label[(fwd_return > b) & (fwd_return <= sb)] = 3  # Buy
    label[(fwd_return < s) & (fwd_return >= ss)] = 1  # Sell
    label[fwd_return < ss] = 0  # Strong Sell
    return label


# ── Technical Indicators ──────────────────────────────────────────────

def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 60+ technical indicators using pandas-ta."""
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    feat = pd.DataFrame(index=df.index)

    # Returns
    for n in [1, 3, 5, 10, 20]:
        feat[f"log_return_{n}d"] = np.log(c / c.shift(n))
    feat["overnight_gap"] = np.log(o / c.shift(1))

    # Moving averages
    for n in [5, 10, 20, 50, 200]:
        feat[f"sma_{n}"] = ta.sma(c, length=n)
        feat[f"price_sma{n}_ratio"] = c / feat[f"sma_{n}"]
    for n in [5, 10, 20, 50]:
        feat[f"ema_{n}"] = ta.ema(c, length=n)
        feat[f"price_ema{n}_ratio"] = c / feat[f"ema_{n}"]

    # Momentum
    feat["rsi_7"]    = ta.rsi(c, length=7)
    feat["rsi_14"]   = ta.rsi(c, length=14)
    stoch = ta.stoch(h, l, c)
    if stoch is not None and not stoch.empty:
        feat["stoch_k"] = stoch.iloc[:, 0]
        feat["stoch_d"] = stoch.iloc[:, 1]
    feat["willr_14"] = ta.willr(h, l, c, length=14)
    feat["cci_20"]   = ta.cci(h, l, c, length=20)
    feat["roc_10"]   = ta.roc(c, length=10)
    feat["roc_20"]   = ta.roc(c, length=20)

    # Trend
    macd_df = ta.macd(c)
    if macd_df is not None and not macd_df.empty:
        # pandas-ta macd() column order: MACD line, Histogram, Signal line
        feat["macd"]          = macd_df.iloc[:, 0]
        feat["macd_hist"]     = macd_df.iloc[:, 1]
        feat["macd_signal"]   = macd_df.iloc[:, 2]
    adx_df = ta.adx(h, l, c)
    if adx_df is not None and not adx_df.empty:
        feat["adx"] = adx_df.iloc[:, 0]
    aroon_df = ta.aroon(h, l)
    if aroon_df is not None and not aroon_df.empty:
        # pandas-ta aroon() column order: AROOND (down) first, AROONU (up) second
        feat["aroon_down"] = aroon_df.iloc[:, 0]
        feat["aroon_up"]   = aroon_df.iloc[:, 1]

    # Volatility
    bb = ta.bbands(c)
    if bb is not None and not bb.empty:
        feat["bb_lower"]  = bb.iloc[:, 0]
        feat["bb_mid"]    = bb.iloc[:, 1]
        feat["bb_upper"]  = bb.iloc[:, 2]
        feat["bb_width"]  = (bb.iloc[:, 2] - bb.iloc[:, 0]) / bb.iloc[:, 1]
        feat["bb_pct_b"]  = (c - bb.iloc[:, 0]) / (bb.iloc[:, 2] - bb.iloc[:, 0] + 1e-9)
    feat["atr_14"]      = ta.atr(h, l, c, length=14)
    feat["hist_vol_20"] = c.pct_change().rolling(20).std() * np.sqrt(252)

    # Volume
    feat["obv"]              = ta.obv(c, v)
    feat["vwap"]             = (c * v).cumsum() / v.cumsum()
    feat["mfi_14"]           = ta.mfi(h, l, c, v, length=14)
    feat["volume_sma_ratio"] = v / v.rolling(20).mean()
    feat["volume_roc"]       = v.pct_change(5)

    # Candlestick patterns
    feat["body_size"]    = (c - o).abs() / (h - l + 1e-9)
    feat["upper_shadow"] = (h - pd.concat([c, o], axis=1).max(axis=1)) / (h - l + 1e-9)
    feat["lower_shadow"] = (pd.concat([c, o], axis=1).min(axis=1) - l) / (h - l + 1e-9)
    feat["doji"]         = (feat["body_size"] < 0.1).astype(int)
    feat["hammer"]       = ((feat["lower_shadow"] > 2 * feat["body_size"]) & (feat["upper_shadow"] < 0.1)).astype(int)
    prev_c = c.shift(1); prev_o = o.shift(1)
    feat["bullish_engulf"] = (
        (o < prev_c) & (c > prev_o) & (c > o) & (prev_c < prev_o)
    ).astype(int)

    # Calendar
    feat["day_of_week"] = df.index.dayofweek
    feat["month"]       = df.index.month
    feat["quarter"]     = df.index.quarter
    feat["is_month_end"] = df.index.is_month_end.astype(int)

    return feat


# ── Sentiment Merge ───────────────────────────────────────────────────

def _merge_sentiment(feat: pd.DataFrame, sentiment_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge FinBERT sentiment features.
    Any date without sentiment data gets neutral imputation:
      score=0.0, confidence=0.5, count=0, trend=0.0
    """
    neutral = {
        "sentiment_score":      0.0,
        "sentiment_confidence": 0.5,
        "news_count_7d":        0.0,
        "sentiment_trend_3d":   0.0,
    }

    if sentiment_df is None or sentiment_df.empty:
        for col, val in neutral.items():
            feat[col] = val
        return feat

    sentiment_df = sentiment_df.reindex(feat.index)
    for col, val in neutral.items():
        if col not in sentiment_df.columns:
            feat[col] = val
        else:
            feat[col] = sentiment_df[col].fillna(val)

    return feat


# ── Main Entry Point ──────────────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    symbol: str,
    sentiment_df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline for one symbol.

    Args:
        df:           Raw OHLCV DataFrame (date index, lowercase columns).
        symbol:       Stock symbol string (for logging and file naming).
        sentiment_df: Optional DataFrame with sentiment columns indexed by date.
        save:         Save processed Parquet to data/processed/.

    Returns:
        Processed DataFrame with 60+ features, 4 sentiment features, and label column.
        Last row is dropped (no forward return available for label).
    """
    cfg = _load_config()
    logger.info("Engineering features for %s (%d rows)", symbol, len(df))

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    feat = _compute_indicators(df)
    feat = _merge_sentiment(feat, sentiment_df)

    label = _make_labels(df["close"], cfg)
    feat["label"] = label

    # Drop last row — no next-day return for label
    feat = feat.iloc[:-1]

    # Drop rows where core indicators are all NaN (warm-up period)
    core_cols = ["rsi_14", "macd", "sma_50"]
    available_core = [c for c in core_cols if c in feat.columns]
    if available_core:
        feat = feat.dropna(subset=available_core)

    # Fill any remaining NaNs with forward-fill then 0
    feat = feat.ffill().fillna(0.0)

    # Ensure label is int
    feat["label"] = feat["label"].astype(int)

    logger.info("%s: %d rows × %d features after engineering", symbol, len(feat), len(feat.columns))

    if save:
        out_dir = Path(cfg["paths"]["processed_data"])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{symbol}.parquet"
        feat.to_parquet(out_path)
        logger.info("%s: saved to %s", symbol, out_path)

    return feat


def engineer_all(
    raw_data: dict,
    sentiment_data: Optional[dict] = None,
) -> dict:
    """
    Engineer features for all symbols.

    Args:
        raw_data:      {symbol: raw_df}
        sentiment_data: {symbol: sentiment_df} — optional

    Returns:
        {symbol: processed_df}
    """
    results = {}
    for symbol, df in raw_data.items():
        sent = (sentiment_data or {}).get(symbol)
        try:
            results[symbol] = engineer_features(df, symbol, sentiment_df=sent)
        except Exception as e:
            logger.error("Feature engineering failed for %s: %s", symbol, e)
    return results


def load_processed(symbol: str) -> pd.DataFrame:
    """Load a saved processed Parquet file."""
    cfg = _load_config()
    path = Path(cfg["paths"]["processed_data"]) / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No processed data found for {symbol} at {path}")
    return pd.read_parquet(path)
