"""
routes/news.py — news feed blueprint (Twitter/X, Alpaca stream, combined sources).

Extracted from app.py; behavior unchanged: each route catches its own errors and
returns {'success': False, 'error': ...} rather than raising.
"""
import os
import sys
import datetime

from flask import Blueprint, jsonify, request

# project root (for the `data` package) — app.py normally sets this up already
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from common import _int_arg  # noqa: E402

bp = Blueprint('news', __name__)

try:
    from data.twitter_feed import get_market_tweets, get_financial_news_feed
    from data.alpaca_news import get_recent_news
    from data.finnhub import get_company_news
except Exception as _e:
    # Routes degrade gracefully: their try/excepts surface the missing name.
    print(f"[NEWS] data layer failed to import: {_e}", flush=True)


@bp.route('/api/news/twitter')
def twitter_news():
    """Get latest market news from Twitter

    Query Parameters:
    - symbol: Optional stock symbol to filter tweets (e.g., AAPL)
    - count: Number of tweets to return (default: 20, max: 100)
    - type: 'market' for general market news, 'financial' for financial news sources
    """
    symbol = request.args.get('symbol', None)
    count = _int_arg('count', 20, max_val=100)
    news_type = request.args.get('type', 'market')

    try:
        if news_type == 'financial':
            # Get tweets from major financial news accounts
            tweets = get_financial_news_feed(count=count)
        else:
            # Get general market tweets or symbol-specific tweets
            tweets = get_market_tweets(symbol=symbol, count=count)

        return jsonify({
            'success': True,
            'count': len(tweets),
            'tweets': tweets,
            'source': 'twitter'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@bp.route('/api/news/alpaca')
def alpaca_news():
    """Get real-time news from Alpaca stream

    Query Parameters:
    - symbol: Optional stock symbol to filter news
    - count: Number of news items to return (default: 20)
    """
    symbol = request.args.get('symbol', None)
    count = _int_arg('count', 20, max_val=100)

    try:
        news_items = get_recent_news(count=count, symbol=symbol)

        return jsonify({
            'success': True,
            'count': len(news_items),
            'news': news_items,
            'source': 'alpaca'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@bp.route('/api/news/combined')
def combined_news():
    """Get combined news from all sources (Finnhub, Twitter, Alpaca)

    Query Parameters:
    - symbol: Optional stock symbol to filter news
    - count: Number of items per source (default: 10)
    """
    symbol = request.args.get('symbol', None)
    count = _int_arg('count', 10, max_val=100)

    try:
        all_news = []

        # Get Finnhub news for popular symbols even if no symbol specified
        symbols_to_check = []
        if symbol:
            symbols_to_check = [symbol.upper()]
        else:
            # Fetch news for popular stocks when no specific symbol
            symbols_to_check = ['NVDA', 'AAPL', 'MSFT', 'TSLA', 'GOOGL'][:2]  # Limit to 2 to respect API limits

        for sym in symbols_to_check:
            try:
                today = datetime.date.today()
                from_date = (today - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
                to_date = today.strftime('%Y-%m-%d')
                finnhub_news = get_company_news(sym, from_date, to_date)

                # Format Finnhub news (limit per symbol)
                items_per_symbol = count // len(symbols_to_check) if len(symbols_to_check) > 1 else count
                for article in finnhub_news[:items_per_symbol]:
                    all_news.append({
                        'source': 'Finnhub',
                        'headline': article.get('headline', ''),
                        'summary': article.get('summary', ''),
                        'url': article.get('url', ''),
                        'created_at': datetime.datetime.fromtimestamp(article.get('datetime', 0)).isoformat(),
                        'symbols': [sym],
                        'type': 'article'
                    })
            except Exception as e:
                print(f"⚠️  Error fetching Finnhub news for {sym}: {e}")

        # Get Twitter news
        twitter_error = None
        try:
            tweets = get_market_tweets(symbol=symbol, count=count)
            if tweets and len(tweets) > 0:
                for tweet in tweets:
                    all_news.append({
                        'source': 'Twitter',
                        'headline': f"@{tweet['author']['username']}: {tweet['text'][:100]}...",
                        'summary': tweet['text'],
                        'url': tweet['url'],
                        'created_at': tweet['created_at'],
                        'symbols': tweet.get('symbols', []),
                        'author': tweet['author'],
                        'metrics': tweet['metrics'],
                        'type': 'tweet'
                    })
            else:
                twitter_error = "Twitter API rate limit reached or no data available"
                print(f"⚠️  {twitter_error}")
        except Exception as e:
            twitter_error = f"Twitter API unavailable: {str(e)}"
            print(f"⚠️  {twitter_error}")

        # Get Alpaca news (if available)
        try:
            alpaca_items = get_recent_news(count=count, symbol=symbol)
            for item in alpaca_items:
                all_news.append({
                    'source': item['source'],
                    'headline': item['headline'],
                    'summary': item.get('summary', ''),
                    'url': item.get('url', ''),
                    'created_at': item['created_at'],
                    'symbols': item.get('symbols', []),
                    'author': item.get('author', ''),
                    'type': 'article'
                })
        except Exception as e:
            print(f"⚠️  Alpaca news not available: {e}")

        # Sort by created_at (newest first)
        all_news.sort(key=lambda x: x.get('created_at', ''), reverse=True)

        response = {
            'success': True,
            'count': len(all_news),
            'news': all_news
        }

        # Add warning if Twitter API failed
        if twitter_error:
            response['warning'] = twitter_error

        return jsonify(response)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })
