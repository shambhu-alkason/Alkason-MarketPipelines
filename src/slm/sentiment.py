"""
FinBERT sentiment analysis — converts financial headlines into 4 ML features.

Model: ProsusAI/finbert (110M params, 97% accuracy on Financial PhraseBank)
Output: sentiment_score, sentiment_confidence, news_count_7d, sentiment_trend_3d
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NEUTRAL_FEATURES = {
    "sentiment_score":      0.0,
    "sentiment_confidence": 0.5,
    "news_count_7d":        0.0,
    "sentiment_trend_3d":   0.0,
}

_model = None
_tokenizer = None


def _load_finbert():
    """Lazy-load FinBERT — only when first needed."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        logger.info("Loading ProsusAI/finbert ...")
        _tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        _model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        _model.eval()
        logger.info("FinBERT loaded successfully")
    except Exception as e:
        logger.error("Failed to load FinBERT: %s", e)
        _model = None
        _tokenizer = None
    return _model, _tokenizer


def _score_headline(text: str) -> Dict[str, float]:
    """
    Run one headline through FinBERT.
    Returns {label: str, score: float, confidence: float}
    Labels: positive (+1), neutral (0), negative (-1)
    """
    model, tokenizer = _load_finbert()
    if model is None:
        return {"label": "neutral", "score": 0.0, "confidence": 0.5}

    try:
        import torch
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()

        # FinBERT label order: positive(0), negative(1), neutral(2)
        label_map = {0: "positive", 1: "negative", 2: "neutral"}
        score_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

        idx = int(np.argmax(probs))
        label = label_map[idx]
        confidence = probs[idx]
        score = score_map[label]
        return {"label": label, "score": score, "confidence": confidence}
    except Exception as e:
        logger.warning("FinBERT scoring failed for headline: %s", e)
        return {"label": "neutral", "score": 0.0, "confidence": 0.5}


def score_headlines(headlines: List[str]) -> Dict[str, float]:
    """
    Score a list of headlines and return aggregate sentiment features.
    Returns neutral imputation if headlines list is empty.
    """
    if not headlines:
        return dict(NEUTRAL_FEATURES)

    scores = [_score_headline(h) for h in headlines]
    sentiment_scores = [s["score"] for s in scores]
    confidences = [s["confidence"] for s in scores]

    return {
        "sentiment_score":      float(np.mean(sentiment_scores)),
        "sentiment_confidence": float(np.mean(confidences)),
        "news_count_7d":        float(len(headlines)),
        "sentiment_trend_3d":   0.0,  # requires rolling window — computed in build_sentiment_df
    }


def build_sentiment_df(
    symbol: str,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Build a sentiment DataFrame for all dates in the index.

    For each date:
    - If date > 90 days ago: fetch headlines, run FinBERT
    - Else: neutral imputation (deterministic, logged)

    The sentiment_trend_3d is computed as the rolling 3-day change in sentiment_score.

    Returns DataFrame indexed by date with 4 sentiment columns.
    """
    from src.slm.news_scraper import fetch_headlines

    today = datetime.today()
    records = []

    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        days_ago = (today - date.to_pydatetime()).days

        if days_ago > 90:
            # Historical — neutral imputation
            records.append({
                "date": date,
                "sentiment_score": 0.0,
                "sentiment_confidence": 0.5,
                "news_count_7d": 0.0,
            })
        else:
            headlines = fetch_headlines(symbol, date_str)
            agg = score_headlines(headlines)
            records.append({
                "date": date,
                "sentiment_score":      agg["sentiment_score"],
                "sentiment_confidence": agg["sentiment_confidence"],
                "news_count_7d":        agg["news_count_7d"],
            })

    df = pd.DataFrame(records).set_index("date")
    df.index = pd.to_datetime(df.index)

    # Compute 3-day sentiment trend (change in score over last 3 trading days)
    df["sentiment_trend_3d"] = df["sentiment_score"].diff(3).fillna(0.0)

    logger.info(
        "%s: sentiment computed for %d dates (%d with real news)",
        symbol,
        len(df),
        int((df["news_count_7d"] > 0).sum()),
    )
    return df


def get_live_sentiment(symbol: str) -> Dict[str, float]:
    """Get current-day sentiment for live predictions (API use)."""
    from src.slm.news_scraper import fetch_recent_headlines
    headlines = fetch_recent_headlines(symbol, lookback_days=7)
    result = score_headlines(headlines)
    result["news_count_7d"] = float(len(headlines))
    return result
