"""
sentiment_feed.py — data for the Sentiment tab.

One Alpha Vantage NEWS_SENTIMENT call (ticker-scored, newest first) plus Finnhub
company news (unscored headlines), cached per symbol and window. At most two
upstream calls and no sleeps, so a click can't outlast the single worker's
timeout. The ML pipeline's multi-source daily history
(extractor.news_sentiment_history) is a separate, offline path.

Scores are Alpha Vantage's ticker_sentiment_score in [-1, 1], averaged with its
relevance_score as the weight; labels use Alpha Vantage's own neutral band.
Windows with no scored article report no score rather than a neutral 0.
"""
import datetime
import math
import threading

from cachetools import TTLCache

LABEL_BAND = 0.15        # Alpha Vantage: |score| < 0.15 is Neutral
MIN_RELEVANCE = 0.1      # floor on the weight, as in the pipeline
AV_LIMIT = 1000          # Alpha Vantage's per-call article cap
MAX_ARTICLES = 100       # rows sent to the browser (stats use every scored article)
AV_DAILY_BUDGET = 10     # of the key's 25/day; the daily retrain uses the same key
CACHE_TTL = 1800         # seconds

_cache = TTLCache(maxsize=128, ttl=CACHE_TTL)
_lock = threading.Lock()
_av_used = {"day": None, "calls": 0}


def label_for(score):
    if score is None:
        return "unscored"
    return "positive" if score >= LABEL_BAND else "negative" if score <= -LABEL_BAND else "neutral"


def _headline_key(text):
    return " ".join(str(text).lower().split())


def parse_av_feed(feed, symbol):
    """Alpha Vantage `feed` items → scored article dicts for `symbol` (newest first)."""
    out = []
    for art in feed or []:
        try:
            dt = datetime.datetime.strptime(str(art.get("time_published", "")), "%Y%m%dT%H%M%S")
        except ValueError:
            continue
        ts = next((t for t in art.get("ticker_sentiment") or []
                   if str(t.get("ticker", "")).upper() == symbol), None)
        if ts is None:
            continue
        try:
            score = float(ts.get("ticker_sentiment_score"))
            rel = float(ts.get("relevance_score"))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(score) and math.isfinite(rel)):
            continue
        out.append({"date": dt.date().isoformat(), "published": dt.isoformat(),
                    "headline": str(art.get("title") or ""),
                    "source": str(art.get("source") or "Alpha Vantage"),
                    "url": str(art.get("url") or ""), "provider": "alphavantage",
                    "score": round(score, 4), "relevance": round(max(rel, MIN_RELEVANCE), 4),
                    "label": label_for(score)})
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
                    "score": None, "relevance": None, "label": "unscored"})
    return out


def merge(scored, unscored):
    """Scored articles plus unscored ones whose headline isn't already scored, newest first."""
    seen = {_headline_key(a["headline"]) for a in scored}
    rest = [a for a in unscored if a["headline"] and _headline_key(a["headline"]) not in seen]
    return sorted(scored + rest, key=lambda a: a["published"], reverse=True)


def summarize(articles):
    """Relevance-weighted mean score and label counts; score None when nothing is scored."""
    scored = [a for a in articles if a["score"] is not None]
    counts = {k: sum(a["label"] == k for a in scored) for k in ("positive", "neutral", "negative")}
    if not scored:
        return {"overall_score": None, "overall_label": "no_data", "scored_total": 0,
                "unscored_total": len(articles), **{f"{k}_count": 0 for k in counts}}
    w = sum(a["relevance"] for a in scored)
    mean = sum(a["score"] * a["relevance"] for a in scored) / w
    return {"overall_score": round(mean, 3), "overall_label": label_for(mean),
            "scored_total": len(scored), "unscored_total": len(articles) - len(scored),
            **{f"{k}_count": v for k, v in counts.items()}}


def daily_series(articles):
    """Per-day relevance-weighted score (None on days with no scored article) and article count."""
    days = {}
    for a in articles:
        d = days.setdefault(a["date"], {"num": 0.0, "w": 0.0, "count": 0})
        d["count"] += 1
        if a["score"] is not None:
            d["num"] += a["score"] * a["relevance"]
            d["w"] += a["relevance"]
    return [{"date": k, "score": round(v["num"] / v["w"], 3) if v["w"] else None, "count": v["count"]}
            for k, v in sorted(days.items())]


def build_payload(symbol, days, start, av_feed, av_status, fh_items):
    """Assemble the tab's JSON from raw upstream results (pure; no network)."""
    scored = parse_av_feed(av_feed, symbol)
    articles = merge(scored, parse_finnhub(fh_items, start))
    notes = []
    if av_status == "rate_limited":
        notes.append("Alpha Vantage's daily limit is used up — headlines are shown unscored.")
    elif av_status == "budget":
        notes.append("This server's Alpha Vantage budget for today is used up — headlines are shown unscored.")
    elif av_status == "error":
        notes.append("Alpha Vantage didn't respond — headlines are shown unscored.")
    coverage_start = scored[-1]["date"] if scored else None
    if len(av_feed or []) >= AV_LIMIT and coverage_start and coverage_start > start:
        notes.append(f"Alpha Vantage returns at most {AV_LIMIT} articles, so scores cover "
                     f"{coverage_start} onward.")
    sources = {}
    for a in articles:
        sources[a["provider"]] = sources.get(a["provider"], 0) + 1
    return {"symbol": symbol, "window_days": days, "start": start,
            "coverage_start": coverage_start, **summarize(articles),
            "articles_total": len(articles), "source_counts": sources,
            "daily": daily_series(articles), "articles": articles[:MAX_ARTICLES],
            "notes": notes}


def _take_av_budget(today):
    with _lock:
        if _av_used["day"] != today:
            _av_used.update(day=today, calls=0)
        if _av_used["calls"] >= AV_DAILY_BUDGET:
            return False
        _av_used["calls"] += 1
        return True


def _fetch_av(av, symbol, start, end, today):
    if av is None or not getattr(av, "api_key", None):
        return [], "error"
    if not _take_av_budget(today):
        return [], "budget"
    try:
        data = av.get_market_news_sentiment(
            tickers=symbol, time_from=start.replace("-", "") + "T0000",
            time_to=end.replace("-", "") + "T2359", sort="LATEST", limit=AV_LIMIT)
    except Exception:
        return [], "error"
    if not isinstance(data, dict):
        return [], "error"
    if "feed" in data:
        return data["feed"], "ok"
    if "Information" in data or "Note" in data:
        return [], "rate_limited"
    return [], "ok"          # e.g. "Error Message" for a ticker AV doesn't cover: no news


def get_feed(symbol, days, av, company_news, today=None):
    """Cached payload for `symbol` over the last `days` calendar days.

    av           — data.alphavantage.AlphaVantage instance (or None)
    company_news — callable(symbol, from, to) → Finnhub company-news list
    """
    today = today or datetime.date.today()
    start = (today - datetime.timedelta(days=days)).isoformat()
    end = today.isoformat()
    key = (symbol, days, end)
    with _lock:
        if key in _cache:
            return _cache[key]
    av_feed, av_status = _fetch_av(av, symbol, start, end, end)
    try:
        fh_items = company_news(symbol, start, end) or []
    except Exception:
        fh_items = []
    payload = build_payload(symbol, days, start, av_feed, av_status, fh_items)
    payload["as_of"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    with _lock:
        _cache[key] = payload
    return payload
