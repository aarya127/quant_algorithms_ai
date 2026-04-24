"""
finbert_canadian.py — FinBERT sentiment analysis for Canadian (TSX) stocks.

Delegates all article fetching and model inference to FinBertAnalyzer in
finbert.py.  Alpha Vantage covers Canadian tickers; Polygon and Finnhub
have limited or no coverage for .TO symbols, so coverage will be lower
than for US equities.

Usage
-----
    python sentiment/finbert_canadian.py              # runs built-in list
    python sentiment/finbert_canadian.py SHOP.TO TD.TO --days 14
"""

from __future__ import annotations

import argparse
import datetime

from sentiment.finbert import FinBertAnalyzer

# ---------------------------------------------------------------------------
# Canadian stock universe (TSX)
# ---------------------------------------------------------------------------
canadian_stocks: dict[str, str] = {
    "AC.TO":    "Air Canada",
    "TD.TO":    "TD Bank",
    "ENB.TO":   "Enbridge",
    "RCI-B.TO": "Rogers Communications",
    "SHOP.TO":  "Shopify",
    "BMO.TO":   "Bank of Montreal",
    "RY.TO":    "Royal Bank of Canada",
}


def analyze_canadian_stock_sentiment(symbol: str, company_name: str, days: int = 7):
    """
    Run multi-source FinBERT analysis on a Canadian ticker.

    Returns the same dict shape as the US analyze_stock_sentiment function.
    Alpha Vantage is the primary source; Polygon/Finnhub are tried as well
    but may return 0 articles for .TO symbols.
    """
    end   = datetime.date.today().isoformat()
    start = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()

    analyzer = FinBertAnalyzer()

    print(f"\n{'='*70}")
    print(f"ANALYZING (Canadian): {company_name} ({symbol})")
    print(f"{'='*70}")

    daily = analyzer.analyze(symbol, start, end)
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

    return {
        "symbol":            symbol,
        "company":           company_name,
        "overall_sentiment": overall,
        "positive":          avg_pos,
        "negative":          avg_neg,
        "neutral":           avg_neu,
        "confidence":        max(avg_pos, avg_neg, avg_neu),
        "sources_count":     total,
        "distribution":      {"positive": 0, "negative": 0, "neutral": 0},
    }


def main():
    parser = argparse.ArgumentParser(description="FinBERT sentiment for Canadian stocks")
    parser.add_argument("symbols", nargs="*", default=list(canadian_stocks.keys()))
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  CANADIAN STOCK SENTIMENT ANALYSIS")
    print("  FinBERT + Alpha Vantage  (Polygon/Finnhub coverage limited for .TO)")
    print("  Note: Alpha Vantage free tier = 25 requests/day")
    print("=" * 70)

    results = []
    for sym in args.symbols:
        company = canadian_stocks.get(sym.upper(), canadian_stocks.get(sym, sym))
        result = analyze_canadian_stock_sentiment(sym, company, days=args.days)
        if result:
            results.append(result)

    if results:
        print(f"\n\n{'='*70}")
        print("SUMMARY — CANADIAN STOCKS")
        print(f"{'Symbol':<12} {'Company':<22} {'Sentiment':<12} {'Pos':>7} {'Neg':>7} {'Articles':>9}")
        print(f"{'-'*70}")
        for r in results:
            print(f"{r['symbol']:<12} {r['company'][:22]:<22} {r['overall_sentiment']:<12} "
                  f"{r['positive']:>7.1%} {r['negative']:>7.1%} {r['sources_count']:>9}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

