"""
IBM Granite 4.1-3B plain-English prediction explainer via Ollama.

Graceful fallback: if Ollama is not running/installed, returns a
template-based explanation rather than crashing.
"""

import logging
import os
from typing import Dict, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

SIGNAL_LABELS = {4: "Strong Buy", 3: "Buy", 2: "Hold", 1: "Sell", 0: "Strong Sell"}

SIGNAL_CONTEXT = {
    4: "very strong bullish signals",
    3: "bullish signals",
    2: "neutral or mixed signals",
    1: "bearish signals",
    0: "very strong bearish signals",
}


def _template_explanation(
    symbol: str,
    signal: str,
    confidence: float,
    top_features: Dict[str, float],
    sentiment_score: float,
) -> str:
    """Rule-based fallback explanation when Ollama is unavailable."""
    feat_str = ", ".join(
        f"{k.replace('_', ' ')}={v:.2f}" for k, v in list(top_features.items())[:3]
    )
    sentiment_label = (
        "positive" if sentiment_score > 0.1 else
        "negative" if sentiment_score < -0.1 else
        "neutral"
    )
    class_key = next((k for k, v in SIGNAL_LABELS.items() if v == signal), 2)
    return (
        f"{symbol} shows {SIGNAL_CONTEXT.get(class_key, 'mixed signals')} "
        f"(confidence {confidence:.0%}). "
        f"Key drivers: {feat_str}. "
        f"News sentiment is {sentiment_label} (score: {sentiment_score:+.2f}). "
        f"[Note: Install Ollama + Granite 4.1-3B for detailed AI explanations: "
        f"brew install ollama && ollama pull granite4.1:3b]"
    )


def _call_ollama(prompt: str, model: str, base_url: str) -> Optional[str]:
    """Call Ollama REST API to generate explanation."""
    try:
        import ollama
        client = ollama.Client(host=base_url)
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": 150},
        )
        return response.message.content.strip()
    except Exception as e:
        logger.warning("Ollama call failed (%s): %s", model, e)
        return None


def explain_prediction(
    symbol: str,
    signal_class: int,
    confidence: float,
    top_features: Dict[str, float],
    sentiment_score: float = 0.0,
    sentiment_label: str = "neutral",
) -> str:
    """
    Generate a plain-English explanation for a stock prediction.

    Uses IBM Granite 4.1-3B via Ollama if available.
    Falls back to a template explanation if Ollama is not installed/running.

    Args:
        symbol:          Stock symbol, e.g. "RELIANCE.NS"
        signal_class:    Prediction class 0–4
        confidence:      Model confidence [0, 1]
        top_features:    Top SHAP features {name: value}
        sentiment_score: FinBERT score in [-1, 1]
        sentiment_label: "positive" / "neutral" / "negative"

    Returns:
        2–3 sentence plain English explanation.
    """
    signal = SIGNAL_LABELS.get(signal_class, "Hold")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = "granite4.1:3b"

    # Build prompt for Granite
    feat_lines = "\n".join(
        f"  - {k.replace('_', ' ')} = {v:.4f}" for k, v in list(top_features.items())[:5]
    )
    prompt = (
        f"You are a financial analyst. Explain this stock recommendation in 2-3 clear sentences "
        f"for a retail investor. Be specific about the reasons.\n\n"
        f"Stock: {symbol}\n"
        f"Signal: {signal} (confidence: {confidence:.0%})\n"
        f"Key signals:\n{feat_lines}\n"
        f"News sentiment: {sentiment_label} (score: {sentiment_score:+.2f})\n\n"
        f"Explanation:"
    )

    explanation = _call_ollama(prompt, model, base_url)

    if explanation:
        logger.info("Granite explanation generated for %s", symbol)
        return explanation

    # Graceful fallback
    logger.info("Using template explanation for %s (Ollama not available)", symbol)
    return _template_explanation(symbol, signal, confidence, top_features, sentiment_score)
