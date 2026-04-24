"""
finbert.py — Multi-source FinBERT sentiment analyzer for financial text.

Exposes
-------
    FinBertAnalyzer                          — main class
    FinBertAnalyzer.get_sentiment(text)      — score one text, returns (label, scores)
    FinBertAnalyzer.fetch_articles(...)      — pull raw articles from all sources
    FinBertAnalyzer.analyze(symbol, ...)     — daily FinBERT sentiment timeseries

Backward-compatible module-level functions (legacy)
----------------------------------------------------
    get_sentiment(text)
    analyze_stock_sentiment(symbol, company_name, days=7)

Sources tried in fetch_articles (all results merged + de-duplicated)
--------------------------------------------------------------------
    1. Polygon v2 /reference/news  — full article text, 5 req/min
    2. Alpha Vantage NEWS_SENTIMENT — headline + summary, 25 req/day
    3. Finnhub company-news         — headline + summary, 60 req/min

FinBERT model: ProsusAI/finbert (loaded lazily on first call to get_sentiment)
Score formula: finbert_score = P(positive) - P(negative)  ∈ [-1, +1]
"""

from __future__ import annotations

import datetime
import logging
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger("FinBertAnalyzer")
logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

# ---------------------------------------------------------------------------
# Key resolution (self-contained — mirrors extractor.py, no circular import)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]   # quant_algorithms_ai/


def _read_keys_txt() -> dict:
    keys: dict = {}
    path = _REPO_ROOT / "keys.txt"
    if not path.exists():
        return keys
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        label = stripped.lstrip("#").strip().lower()
        if i + 1 < len(lines):
            value = lines[i + 1].strip()
            if value and not value.startswith("#"):
                keys[label] = value
    return keys


_KEYS = _read_keys_txt()


def _key(env_var: str, label: str) -> str:
    return os.environ.get(env_var) or _KEYS.get(label, "")


# ---------------------------------------------------------------------------
# Lazy FinBERT model loader
# ---------------------------------------------------------------------------
_tokenizer = None
_model = None


def _load_model():
    """Import torch/transformers and load ProsusAI/finbert on first use."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch  # noqa
        logger.info("Loading FinBERT model (one-time, ~5-15s)…")
        _tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        _model     = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        logger.info("FinBERT model ready")


# ---------------------------------------------------------------------------
# FinBertAnalyzer — main class
# ---------------------------------------------------------------------------

class FinBertAnalyzer:
    """
    Multi-source FinBERT sentiment analyzer.

    Fetches raw articles from Polygon, Alpha Vantage, and Finnhub,
    runs ProsusAI/finbert on each headline+summary, and aggregates
    the results to a daily timeseries.

    Score formula
    -------------
        finbert_score = P(positive) - P(negative)  ∈ [-1, +1]
    This is more informative than the binary +1/0/-1 produced by
    API pre-scores: a headline that is 70% positive / 25% negative
    gets +0.45, not the same +1.0 as a 99%/0% positive one.
    """

    _POLYGON_SLEEP = 12.0   # 5 req/min → 12 s between pages

    def __init__(self, max_articles: int = 500):
        self.max_articles = max_articles

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_sentiment(self, text: str):
        """
        Run FinBERT on a single text snippet.

        Returns
        -------
        (dominant_label, {positive, negative, neutral})  or  (None, None)
        """
        if not text or len(text.strip()) < 10:
            return None, None
        _load_model()
        import torch
        inputs = _tokenizer(
            text, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        )
        outputs = _model(**inputs)
        probs  = torch.nn.functional.softmax(outputs.logits, dim=-1)
        scores = probs[0].tolist()
        if hasattr(_model.config, "id2label"):
            labels = [_model.config.id2label[i] for i in range(len(scores))]
        else:
            labels = ["positive", "negative", "neutral"]
        result   = {labels[i]: scores[i] for i in range(len(scores))}
        dominant = labels[torch.argmax(probs[0]).item()]
        return dominant, result

    def fetch_articles(self, symbol: str, start: str, end: str) -> list[dict]:
        """
        Pull raw articles from all available sources and merge them.

        Returns a list of dicts with keys: date, headline, summary, source
        De-duplicated by (date-prefix, headline[:80]).
        """
        articles: list[dict] = []

        poly = self._fetch_polygon(symbol, start, end)
        articles.extend(poly)
        logger.info("Polygon    : %d articles for %s", len(poly), symbol)

        av = self._fetch_alphavantage(symbol, start, end)
        articles.extend(av)
        logger.info("AlphaVantage: %d articles for %s", len(av), symbol)

        fh = self._fetch_finnhub(symbol, start, end)
        articles.extend(fh)
        logger.info("Finnhub    : %d articles for %s", len(fh), symbol)

        # De-duplicate
        seen:   set       = set()
        unique: list[dict] = []
        for a in articles:
            key = (str(a.get("date", ""))[:10], a.get("headline", "")[:80].lower())
            if key not in seen:
                seen.add(key)
                unique.append(a)

        logger.info("Total unique articles for %s: %d", symbol, len(unique))
        return unique[: self.max_articles]

    def analyze(
        self,
        symbol: str,
        start:  str = "",
        end:    str = "",
    ) -> pd.DataFrame:
        """
        Run FinBERT on all fetched articles and aggregate to daily scores.

        Returns
        -------
        DataFrame indexed by date with columns:
            finbert_score  — mean(P(pos) - P(neg)) for the day
            finbert_pos    — mean P(positive)
            finbert_neg    — mean P(negative)
            finbert_neu    — mean P(neutral)
            article_count  — articles scored that day
        """
        if not start:
            start = (datetime.date.today() - datetime.timedelta(days=365)).isoformat()
        if not end:
            end = datetime.date.today().isoformat()

        articles = self.fetch_articles(symbol, start, end)
        if not articles:
            return pd.DataFrame()

        rows: list[dict] = []
        for a in articles:
            text = f"{a.get('headline', '')}. {a.get('summary', '')}".strip(". ")
            _, scores = self.get_sentiment(text)
            if scores is None:
                continue
            pos = scores.get("positive", 0.0)
            neg = scores.get("negative", 0.0)
            neu = scores.get("neutral",  0.0)
            rows.append({
                "date":          a["date"],
                "finbert_score": pos - neg,
                "finbert_pos":   pos,
                "finbert_neg":   neg,
                "finbert_neu":   neu,
                "source":        a.get("source", ""),
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        daily = (
            df.groupby("date")
            .agg(
                finbert_score  = ("finbert_score", "mean"),
                finbert_pos    = ("finbert_pos",   "mean"),
                finbert_neg    = ("finbert_neg",   "mean"),
                finbert_neu    = ("finbert_neu",   "mean"),
                article_count  = ("finbert_score", "count"),
            )
            .sort_index()
        )
        daily.index = pd.DatetimeIndex(daily.index)
        return daily

    def print_summary(self, symbol: str, daily: pd.DataFrame) -> None:
        """Print a human-readable summary of a daily FinBERT result."""
        if daily is None or daily.empty:
            print(f"   No data for {symbol}")
            return
        total   = int(daily["article_count"].sum())
        avg_pos = daily["finbert_pos"].mean()
        avg_neg = daily["finbert_neg"].mean()
        avg_neu = daily["finbert_neu"].mean()
        avg_scr = daily["finbert_score"].mean()
        label   = "POSITIVE" if avg_scr > 0.05 else ("NEGATIVE" if avg_scr < -0.05 else "NEUTRAL")
        conf    = max(avg_pos, avg_neg, avg_neu)
        strength = "STRONG" if conf > 0.7 else ("MODERATE" if conf > 0.5 else "WEAK")
        print(f"\n{'='*70}")
        print(f"  {symbol}  |  {label} ({strength})  |  score={avg_scr:+.3f}")
        print(f"  {total} articles over {len(daily)} trading days")
        print(f"  Avg: Pos={avg_pos:.1%}  Neg={avg_neg:.1%}  Neu={avg_neu:.1%}")
        print(f"{'='*70}")

    # ------------------------------------------------------------------
    # Private: per-source article fetchers
    # ------------------------------------------------------------------

    def _fetch_polygon(self, symbol: str, start: str, end: str) -> list[dict]:
        api_key = _key("POLYGON_API_KEY", "massive - 5 api calls/minute")
        if not api_key:
            return []
        articles: list[dict] = []
        url    = "https://api.polygon.io/v2/reference/news"
        params: dict = {
            "ticker":            symbol.upper(),
            "published_utc.gte": start,
            "published_utc.lte": end,
            "order":             "asc",
            "limit":             1000,
            "apiKey":            api_key,
        }
        fetched = 0
        try:
            while url and fetched < self.max_articles:
                resp = requests.get(url, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                for article in data.get("results", []):
                    pub = article.get("published_utc", "")
                    if not pub:
                        continue
                    articles.append({
                        "date":     pd.to_datetime(pub, utc=True).tz_convert(None).normalize(),
                        "headline": article.get("title", ""),
                        "summary":  article.get("description", ""),
                        "source":   "polygon",
                    })
                    fetched += 1
                next_url = data.get("next_url")
                if next_url and fetched < self.max_articles:
                    url    = next_url + ("&" if "?" in next_url else "?") + f"apiKey={api_key}"
                    params = {}
                    logger.info("Polygon: sleeping %.0fs (rate limit) before next page…", self._POLYGON_SLEEP)
                    time.sleep(self._POLYGON_SLEEP)
                else:
                    break
        except Exception as exc:
            logger.warning("Polygon fetch error for %s: %s", symbol, exc)
        return articles

    def _fetch_alphavantage(self, symbol: str, start: str, end: str) -> list[dict]:
        api_key = _key("ALPHAVANTAGE_API_KEY", "alphavantage - 25 requests per day")
        if not api_key:
            return []
        time_from = start.replace("-", "") + "T0000"
        time_to   = end.replace("-", "")   + "T2359"
        articles: list[dict] = []
        try:
            resp = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function":  "NEWS_SENTIMENT",
                    "tickers":   symbol.upper(),
                    "time_from": time_from,
                    "time_to":   time_to,
                    "limit":     1000,
                    "apikey":    api_key,
                },
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
            if "Note" in data or "Information" in data:
                logger.warning("AlphaVantage rate-limited for %s", symbol)
                return []
            for item in data.get("feed", []):
                pub = item.get("time_published", "")
                if not pub:
                    continue
                articles.append({
                    "date":     pd.to_datetime(pub, format="%Y%m%dT%H%M%S", errors="coerce").normalize(),
                    "headline": item.get("title", ""),
                    "summary":  item.get("summary", ""),
                    "source":   "alphavantage",
                })
        except Exception as exc:
            logger.warning("AlphaVantage fetch error for %s: %s", symbol, exc)
        return [a for a in articles if pd.notna(a.get("date"))]

    def _fetch_finnhub(self, symbol: str, start: str, end: str) -> list[dict]:
        api_key = _key("FINNHUB_API_KEY", "finhub - 60 api calls/minute")
        if not api_key:
            return []
        articles: list[dict] = []
        try:
            resp = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params={
                    "symbol": symbol.upper(),
                    "from":   start,
                    "to":     end,
                    "token":  api_key,
                },
                timeout=15,
            )
            resp.raise_for_status()
            for item in resp.json():
                ts = item.get("datetime", 0)
                if not ts:
                    continue
                articles.append({
                    "date":     pd.to_datetime(ts, unit="s").normalize(),
                    "headline": item.get("headline", ""),
                    "summary":  item.get("summary", ""),
                    "source":   "finnhub",
                })
        except Exception as exc:
            logger.warning("Finnhub fetch error for %s: %s", symbol, exc)
        return articles


# ---------------------------------------------------------------------------
# Backward-compatible module-level functions
# ---------------------------------------------------------------------------

# Default stocks for standalone runs
stocks = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp.",
    "TSLA": "Tesla Inc.",
    "NVDA": "NVIDIA Corp.",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon",
    "META": "Meta Platforms",
}


def get_sentiment(text: str):
    """Score one text snippet with FinBERT.

    Returns (dominant_label, {positive, negative, neutral}) or (None, None).
    Backward-compatible with the original finbert.py signature.
    """
    return FinBertAnalyzer().get_sentiment(text)


def analyze_stock_sentiment(symbol: str, company_name: str, days: int = 7):
    """
    Backward-compatible single-symbol analysis.

    Now uses all three sources (Polygon + Alpha Vantage + Finnhub) instead
    of only Finnhub.  Returns the same dict shape as the original function.
    """
    end   = datetime.date.today().isoformat()
    start = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()

    analyzer = FinBertAnalyzer()
    daily    = analyzer.analyze(symbol, start, end)

    print(f"\n{'='*70}")
    print(f"ANALYZING: {company_name} ({symbol})")
    analyzer.print_summary(symbol, daily)

    if daily is None or daily.empty:
        print("   WARNING: No data available for sentiment analysis")
        return None

    total   = int(daily["article_count"].sum())
    avg_pos = float(daily["finbert_pos"].mean())
    avg_neg = float(daily["finbert_neg"].mean())
    avg_neu = float(daily["finbert_neu"].mean())
    avg_scr = float(daily["finbert_score"].mean())
    overall = "positive" if avg_scr > 0.05 else ("negative" if avg_scr < -0.05 else "neutral")
    conf    = max(avg_pos, avg_neg, avg_neu)

    return {
        "symbol":           symbol,
        "company":          company_name,
        "overall_sentiment": overall,
        "positive":         avg_pos,
        "negative":         avg_neg,
        "neutral":          avg_neu,
        "confidence":       conf,
        "sources_count":    total,
        "articles_analyzed": total,
        "positive_ratio":   avg_pos,
        "negative_ratio":   avg_neg,
        "neutral_ratio":    avg_neu,
        "distribution":     {"positive": 0, "negative": 0, "neutral": 0},
        "summary":          f"Analyzed {total} articles over {len(daily)} trading days",
    }


def main():
    """Run multi-source FinBERT sentiment analysis on US stocks."""
    import argparse
    parser = argparse.ArgumentParser(description="FinBERT multi-source sentiment")
    parser.add_argument("symbols", nargs="*", default=list(stocks.keys()),
                        help="Ticker symbols to analyze (default: built-in list)")
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  US STOCK SENTIMENT ANALYSIS")
    print("  FinBERT + Polygon + Alpha Vantage + Finnhub")
    print("=" * 70)

    results = []
    analyzer = FinBertAnalyzer()

    for sym in args.symbols:
        company = stocks.get(sym.upper(), sym.upper())
        end   = datetime.date.today().isoformat()
        start = (datetime.date.today() - datetime.timedelta(days=args.days)).isoformat()
        try:
            daily = analyzer.analyze(sym, start, end)
            analyzer.print_summary(sym, daily)
            if daily is not None and not daily.empty:
                results.append({
                    "symbol":    sym.upper(),
                    "company":   company,
                    "sentiment": ("positive" if daily["finbert_score"].mean() > 0.05
                                  else "negative" if daily["finbert_score"].mean() < -0.05
                                  else "neutral"),
                    "score":     daily["finbert_score"].mean(),
                    "positive":  daily["finbert_pos"].mean(),
                    "negative":  daily["finbert_neg"].mean(),
                    "articles":  int(daily["article_count"].sum()),
                })
        except Exception as exc:
            print(f"\nERROR analyzing {sym}: {exc}")

    if results:
        print(f"\n\n{'='*70}")
        print("SUMMARY")
        print(f"{'Symbol':<8} {'Sentiment':<12} {'Score':>7} {'Pos':>7} {'Neg':>7} {'Articles':>9}")
        print(f"{'-'*70}")
        for r in results:
            print(f"{r['symbol']:<8} {r['sentiment']:<12} {r['score']:>+7.3f} "
                  f"{r['positive']:>7.1%} {r['negative']:>7.1%} {r['articles']:>9}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
