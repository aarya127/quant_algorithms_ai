"""
Alpaca news — recent market headlines from Alpaca's REST news API.

Fetched on request and cached briefly. (This replaced a WebSocket stream that was
never started, never reconnected, and only held news that arrived while the
process was up — so on a sleeping free-tier instance it was always empty.)
"""

import os
import logging
import threading
from typing import Dict, List

import requests
from cachetools import TTLCache

logger = logging.getLogger(__name__)

# Alpaca API credentials — set via environment variables (never hardcode here)
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"

_cache = TTLCache(maxsize=64, ttl=120)
_cache_lock = threading.Lock()


def get_recent_news(count: int = 20, symbol: str = None) -> List[Dict]:
    """Newest-first Alpaca news, optionally for one symbol; [] when not configured."""
    if not (ALPACA_API_KEY and ALPACA_SECRET_KEY):
        return []
    key = (symbol.upper() if symbol else None, count)
    with _cache_lock:
        if key in _cache:
            return _cache[key]

    params = {"limit": min(max(count, 1), 50), "sort": "desc"}   # API max is 50
    if symbol:
        params["symbols"] = symbol.upper()
    resp = requests.get(NEWS_URL, params=params, timeout=10, headers={
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    })
    resp.raise_for_status()

    news = [{
        'id': n.get('id'),
        'headline': n.get('headline'),
        'summary': n.get('summary'),
        'author': n.get('author'),
        'created_at': n.get('created_at'),
        'updated_at': n.get('updated_at'),
        'url': n.get('url'),
        'symbols': n.get('symbols', []),
        'source': f"Alpaca - {n.get('source', 'Unknown')}",
        'type': 'realtime',
    } for n in resp.json().get('news', [])]

    with _cache_lock:
        _cache[key] = news
    return news
