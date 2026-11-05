"""
sentiment_feed.py — data for the Sentiment tab.

One Polygon /v2/reference/news call (newest first): each article carries a
per-ticker sentiment — positive, neutral or negative — and a one-line reason.
When Polygon fails or has no articles, Finnhub company news fills the table as
unscored headlines. At most two upstream calls and no sleeps, so a click can't
outlast the single worker's timeout; scored answers are cached per symbol and
window. The ML pipeline's multi-source daily history
(extractor.news_sentiment_history) is a separate, offline path.

Alpha Vantage NEWS_SENTIMENT used to score this tab; it became a premium
endpoint for this key in Nov 2026.

Scores: positive +1, neutral 0, negative −1. The overall and daily scores are
their mean — the net share of positive over negative articles — in [−1, 1],
labelled with a ±0.15 neutral band. Windows with no scored article report no
score rather than a neutral 0.
"""
import datetime
import os
import threading

import requests
from cachetools import TTLCache

from common import keys_txt_value

LABEL_BAND = 0.15        # |mean| < 0.15 is neutral
POLYGON_URL = "https://api.polygon.io/v2/reference/news"
POLYGON_LIMIT = 1000     # Polygon's per-call article cap
MAX_ARTICLES = 100       # rows sent to the browser (stats use every scored article)
CACHE_TTL = 1800         # seconds
SCORES = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

_cache = TTLCache(maxsize=128, ttl=CACHE_TTL)
_lock = threading.Lock()

NOTES = {
    "no_key": "No Polygon key is configured (POLYGON_API_KEY) — headlines are shown unscored.",
    "auth": "Polygon rejected the API key — headlines are shown unscored.",
    "rate_limited": "Polygon allows 5 calls a minute — try again shortly; headlines are shown unscored.",
    "error": "Polygon didn't respond — headlines are shown unscored.",
}


def label_for(score):
    if score is None:
        return "unscored"
    return "positive" if score >= LABEL_BAND else "negative" if score <= -LABEL_BAND else "neutral"


def _headline_key(text):
    return " ".join(str(text).lower().split())


def _utc(iso):
    return datetime.datetime.fromisoformat(str(iso).replace("Z", "+00:00")).astimezone(datetime.timezone.utc)


def parse_polygon(results, symbol):
    """Polygon news `results` → article dicts for `symbol` (newest first).

    An article without an insight for `symbol` itself (only for other tickers it
    mentions) is kept unscored rather than borrowing another ticker's view.
    """
    out = []
    for art in results or []:
        try:
            dt = _utc(art.get("published_utc"))
        except (TypeError, ValueError):
            continue
        ins = next((i for i in art.get("insights") or []
                    if str(i.get("ticker", "")).upper() == symbol), None)
        score = SCORES.get(str((ins or {}).get("sentiment", "")).lower())
        out.append({"date": dt.date().isoformat(), "published": dt.replace(tzinfo=None).isoformat(),
                    "headline": str(art.get("title") or ""),
                    "source": str((art.get("publisher") or {}).get("name") or "Polygon"),
                    "url": str(art.get("article_url") or ""), "provider": "polygon",
                    "score": score, "label": label_for(score),
                    "reason": str((ins or {}).get("sentiment_reasoning") or "") if score is not None else ""})
    return sorted(out, key=lambda a: a["published"], reverse=True)


def parse_finnhub(items, start):
    """Finnhub company-news items on/after `start` (ISO date) → unscored article dicts."""
    out = []
    for it in items or []:
        try:
            dt = datetime.datetime.fromtimestamp(int(it.get("datetime")), datetime.timezone.utc)
        except (TypeError, ValueError, OverflowError, OSError):
            continue
        if dt.date().isoformat() < start:
            continue
        out.append({"date": dt.date().isoformat(), "published": dt.replace(tzinfo=None).isoformat(),
                    "headline": str(it.get("headline") or ""),
                    "source": str(it.get("source") or "Finnhub"),
                    "url": str(it.get("url") or ""), "provider": "finnhub",
                    "score": None, "label": "unscored", "reason": ""})
    return out


def merge(scored, unscored):
    """Scored articles plus unscored ones whose headline isn't already present, newest first."""
    seen = {_headline_key(a["headline"]) for a in scored}
    rest = [a for a in unscored if a["headline"] and _headline_key(a["headline"]) not in seen]
    return sorted(scored + rest, key=lambda a: a["published"], reverse=True)


def summarize(articles):
    """Mean score and label counts; score None when nothing is scored."""
    scored = [a for a in articles if a["score"] is not None]
    counts = {k: sum(a["label"] == k for a in scored) for k in ("positive", "neutral", "negative")}
    if not scored:
        return {"overall_score": None, "overall_label": "no_data", "scored_total": 0,
                "unscored_total": len(articles), **{f"{k}_count": 0 for k in counts}}
    mean = sum(a["score"] for a in scored) / len(scored)
    return {"overall_score": round(mean, 3), "overall_label": label_for(mean),
            "scored_total": len(scored), "unscored_total": len(articles) - len(scored),
            **{f"{k}_count": v for k, v in counts.items()}}


def daily_series(articles):
    """Per-day mean score (None on days with no scored article) and article count."""
    days = {}
    for a in articles:
        d = days.setdefault(a["date"], {"sum": 0.0, "n": 0, "count": 0})
        d["count"] += 1
        if a["score"] is not None:
            d["sum"] += a["score"]
            d["n"] += 1
    return [{"date": k, "score": round(v["sum"] / v["n"], 3) if v["n"] else None, "count": v["count"]}
            for k, v in sorted(days.items())]


def build_payload(symbol, days, start, poly_results, poly_status, fh_items):
    """Assemble the tab's JSON from raw upstream results (pure; no network)."""
    scored = parse_polygon(poly_results, symbol)
    articles = merge(scored, parse_finnhub(fh_items, start))
    notes = [NOTES[poly_status]] if poly_status in NOTES else []
    coverage_start = scored[-1]["date"] if scored else None
    if len(poly_results or []) >= POLYGON_LIMIT and coverage_start and coverage_start > start:
        notes.append(f"Polygon returns at most {POLYGON_LIMIT} articles, so scores cover "
                     f"{coverage_start} onward.")
    sources = {}
    for a in articles:
        sources[a["provider"]] = sources.get(a["provider"], 0) + 1
    return {"symbol": symbol, "window_days": days, "start": start,
            "coverage_start": coverage_start, **summarize(articles),
            "articles_total": len(articles), "source_counts": sources,
            "daily": daily_series(articles), "articles": articles[:MAX_ARTICLES],
            "notes": notes}


def polygon_key():
    return os.environ.get("POLYGON_API_KEY") or keys_txt_value("massive secret")


def polygon_news(symbol, start, key=None):
    """One Polygon news call → (results, status). Status: ok | no_key | auth | rate_limited | error."""
    key = key if key is not None else polygon_key()
    if not key:
        return [], "no_key"
    try:
        resp = requests.get(POLYGON_URL, timeout=15,
                            params={"ticker": symbol, "published_utc.gte": start,
                                    "order": "desc", "sort": "published_utc", "limit": POLYGON_LIMIT},
                            headers={"Authorization": f"Bearer {key}"})
    except requests.RequestException:
        return [], "error"
    if resp.status_code in (401, 403):
        return [], "auth"
    if resp.status_code == 429:
        return [], "rate_limited"
    try:
        data = resp.json()
    except ValueError:
        return [], "error"
    if resp.status_code != 200 or not isinstance(data, dict):
        return [], "error"
    return data.get("results") or [], "ok"


def get_feed(symbol, days, company_news, news=polygon_news, today=None):
    """Cached payload for `symbol` over the last `days` calendar days.

    company_news — callable(symbol, from, to) → Finnhub company-news list
    news         — callable(symbol, from) → (Polygon results, status)
    """
    today = today or datetime.date.today()
    start = (today - datetime.timedelta(days=days)).isoformat()
    end = today.isoformat()
    key = (symbol, days, end)
    with _lock:
        if key in _cache:
            return _cache[key]
    results, status = news(symbol, start)
    fh_items = []
    if not results:                     # Polygon down or silent: show Finnhub headlines instead
        try:
            fh_items = company_news(symbol, start, end) or []
        except Exception:
            fh_items = []
    payload = build_payload(symbol, days, start, results, status, fh_items)
    payload["as_of"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    # Only cache Polygon's real answer: a degraded one would pin "unscored" for
    # the whole TTL after Polygon recovers.
    if status == "ok":
        with _lock:
            _cache[key] = payload
    return payload
