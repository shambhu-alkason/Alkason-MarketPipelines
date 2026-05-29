"""
News headline fetcher using ddgs (DuckDuckGo) with optional NewsAPI fallback.
Headlines are cached per (symbol, date) to avoid re-fetching.
Historical dates with no cached news return an empty list — features.py will impute neutral sentiment.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import yaml
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def _load_config() -> dict:
    cfg_path = Path(__file__).parents[2] / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _cache_path(symbol: str, date_str: str, cfg: dict) -> Path:
    news_dir = Path(cfg["paths"]["news_cache"])
    news_dir.mkdir(parents=True, exist_ok=True)
    return news_dir / f"{symbol}_{date_str}.json"


def _load_cache(path: Path) -> Optional[List[str]]:
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            return data.get("headlines", [])
        except Exception:
            return None
    return None


def _save_cache(path: Path, headlines: List[str]) -> None:
    with open(path, "w") as f:
        json.dump({"headlines": headlines, "fetched_at": datetime.now().isoformat()}, f)


def _fetch_ddgs(query: str, max_results: int) -> List[str]:
    """Fetch headlines from DuckDuckGo News."""
    try:
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.news(query, max_results=max_results):
                title = r.get("title", "")
                if title:
                    results.append(title)
        return results
    except Exception as e:
        logger.warning("ddgs fetch failed: %s", e)
        return []


def _fetch_newsapi(query: str, max_results: int, api_key: str) -> List[str]:
    """Fetch headlines from NewsAPI (optional)."""
    try:
        from newsapi import NewsApiClient
        client = NewsApiClient(api_key=api_key)
        resp = client.get_everything(
            q=query,
            language="en",
            sort_by="relevancy",
            page_size=min(max_results, 100),
        )
        return [a["title"] for a in resp.get("articles", []) if a.get("title")]
    except Exception as e:
        logger.warning("NewsAPI fetch failed: %s", e)
        return []


def fetch_headlines(
    symbol: str,
    date: Optional[str] = None,
) -> List[str]:
    """
    Fetch financial news headlines for a symbol on a given date.

    For historical dates (> 90 days ago), returns [] immediately — no internet lookup
    possible. features.py will apply neutral sentiment imputation for these rows.

    For recent dates, checks cache first, then fetches from ddgs + optional NewsAPI.
    """
    cfg = _load_config()
    slm_cfg = cfg["slm"]
    today = datetime.today()
    date_str = date or today.strftime("%Y-%m-%d")
    date_dt = datetime.strptime(date_str, "%Y-%m-%d")

    # Historical dates — cannot retrieve from internet; return empty for neutral imputation
    if (today - date_dt).days > 90:
        return []

    cache_file = _cache_path(symbol, date_str, cfg)
    cached = _load_cache(cache_file)
    if cached is not None:
        logger.debug("%s %s: %d headlines from cache", symbol, date_str, len(cached))
        return cached

    max_h = slm_cfg.get("max_headlines_per_stock", 20)
    company_map = {
        "RELIANCE.NS": "Reliance Industries",
        "TCS.NS": "Tata Consultancy Services TCS",
        "INFY.NS": "Infosys",
        "HDFCBANK.NS": "HDFC Bank",
        "WIPRO.NS": "Wipro",
    }
    company = company_map.get(symbol, symbol.replace(".NS", "").replace(".BO", ""))
    query = f"{company} stock NSE news"

    headlines = _fetch_ddgs(query, max_h)

    # Optional NewsAPI supplement
    api_key = os.getenv("NEWSAPI_KEY", "")
    if api_key and len(headlines) < max_h:
        extra = _fetch_newsapi(query, max_h - len(headlines), api_key)
        headlines = list(dict.fromkeys(headlines + extra))  # deduplicate

    headlines = headlines[:max_h]

    if slm_cfg.get("cache_news", True):
        _save_cache(cache_file, headlines)

    logger.info("%s %s: fetched %d headlines", symbol, date_str, len(headlines))
    return headlines


def fetch_recent_headlines(symbol: str, lookback_days: int = 7) -> List[str]:
    """Fetch and combine headlines for the last N days."""
    cfg = _load_config()
    lookback = lookback_days or cfg["slm"].get("news_lookback_days", 7)
    all_headlines = []
    for i in range(lookback):
        date_str = (datetime.today() - timedelta(days=i)).strftime("%Y-%m-%d")
        all_headlines.extend(fetch_headlines(symbol, date_str))
    return list(dict.fromkeys(all_headlines))  # deduplicate while preserving order
