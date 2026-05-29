"""
Data validation — Pandera schema checks for raw OHLCV and processed feature data.
Raises SchemaError on first violation so the pipeline fails fast with a clear message.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check, Index

logger = logging.getLogger(__name__)

# ── Raw OHLCV Schema ──────────────────────────────────────────────────

RAW_SCHEMA = DataFrameSchema(
    columns={
        "open":   Column(float, Check.greater_than(0), nullable=False),
        "high":   Column(float, Check.greater_than(0), nullable=False),
        "low":    Column(float, Check.greater_than(0), nullable=False),
        "close":  Column(float, Check.greater_than(0), nullable=False),
        "volume": Column(float, Check.greater_than_or_equal_to(0), nullable=False),
    },
    index=Index(pa.DateTime, name="date"),
    checks=[
        Check(lambda df: len(df) >= 200, error="Need at least 200 rows of data"),
        Check(lambda df: df.index.is_monotonic_increasing, error="Date index must be sorted ascending"),
        Check(lambda df: not df.index.duplicated().any(), error="Duplicate dates found in index"),
        Check(lambda df: (df["high"] >= df["low"]).all(), error="High must be >= Low"),
        Check(lambda df: (df["high"] >= df["open"]).all(), error="High must be >= Open"),
        Check(lambda df: (df["high"] >= df["close"]).all(), error="High must be >= Close"),
    ],
    coerce=True,
)

# ── Processed Feature Schema ──────────────────────────────────────────

SENTIMENT_FEATURES = [
    "sentiment_score",
    "sentiment_confidence",
    "news_count_7d",
    "sentiment_trend_3d",
]

REQUIRED_FEATURE_GROUPS = {
    "returns":    ["log_return_1d", "log_return_5d", "log_return_20d"],
    "moving_avg": ["sma_20", "ema_20", "price_sma200_ratio"],
    "momentum":   ["rsi_14", "macd", "stoch_k"],
    "volatility": ["bb_width", "atr_14"],
    "volume":     ["obv", "mfi_14"],
    "sentiment":  SENTIMENT_FEATURES,
    "label":      ["label"],
}


def _processed_schema(feature_cols: list) -> DataFrameSchema:
    cols = {}
    for col in feature_cols:
        if col == "label":
            cols[col] = Column(int, Check.isin([0, 1, 2, 3, 4]), nullable=False)
        elif col in SENTIMENT_FEATURES:
            cols[col] = Column(float, nullable=False)
        else:
            cols[col] = Column(float, nullable=True)  # some indicators have NaN at edges

    return DataFrameSchema(
        columns=cols,
        index=Index(pa.DateTime, name="date"),
        checks=[
            Check(lambda df: len(df) >= 100, error="Need at least 100 processed rows"),
        ],
        coerce=True,
    )


# ── Public API ────────────────────────────────────────────────────────

def validate_raw(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Validate raw OHLCV DataFrame. Raises pa.errors.SchemaError on failure."""
    logger.info("Validating raw data for %s (%d rows)", symbol, len(df))
    try:
        validated = RAW_SCHEMA.validate(df)
        logger.info("%s raw data: OK", symbol)
        return validated
    except pa.errors.SchemaError as e:
        logger.error("Raw data validation FAILED for %s: %s", symbol, e)
        raise


def validate_processed(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Validate processed feature DataFrame."""
    logger.info("Validating processed features for %s (%d rows)", symbol, len(df))

    # Check all required feature groups have at least one representative column
    missing_groups = []
    for group, cols in REQUIRED_FEATURE_GROUPS.items():
        if not any(c in df.columns for c in cols):
            missing_groups.append(group)
    if missing_groups:
        raise ValueError(f"{symbol}: missing feature groups: {missing_groups}")

    # Build dynamic schema from actual columns present
    schema = _processed_schema(list(df.columns))
    try:
        validated = schema.validate(df)
        logger.info("%s processed features: OK (%d rows, %d cols)", symbol, len(df), len(df.columns))
        return validated
    except pa.errors.SchemaError as e:
        logger.error("Processed data validation FAILED for %s: %s", symbol, e)
        raise


def validate_all_raw(raw_data: dict) -> dict:
    """Validate a dict of {symbol: DataFrame}. Returns only valid symbols."""
    valid = {}
    for symbol, df in raw_data.items():
        try:
            valid[symbol] = validate_raw(df, symbol)
        except Exception as e:
            logger.error("Dropping %s from pipeline: %s", symbol, e)
    return valid


def validate_all_processed(processed_data: dict) -> dict:
    """Validate a dict of processed {symbol: DataFrame}."""
    valid = {}
    for symbol, df in processed_data.items():
        try:
            valid[symbol] = validate_processed(df, symbol)
        except Exception as e:
            logger.error("Dropping %s from pipeline: %s", symbol, e)
    return valid
