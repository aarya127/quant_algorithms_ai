"""
Comprehensive Stock Analysis Backend
Aggregates data from multiple sources and provides comparison

API RATE LIMITS (from keys.txt):
- Finnhub: 60 API calls/minute (FREE tier)
- Alpha Vantage: 25 requests/day (FREE tier) 
- Tweepy: Not yet implemented
- Fiscal.ai: 250 calls/day

IMPORTANT: This analyzer makes multiple API calls per stock:
- get_comprehensive_sentiment(): ~5-8 calls (FinBERT news, AV sentiment, Finnhub insider)
- get_comprehensive_data(): ~10-15 calls (company, quote, financials, news, earnings)
- Be mindful of rate limits when analyzing multiple stocks in succession!
"""

import sys
import os
import datetime
import time
from typing import Dict, List, Optional
import json

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.finnhub import (
    get_company_news,
    get_basic_financials,
    get_earnings_surprises,
    get_insider_transactions,
    get_insider_sentiment,
    get_company_profile
)
from data.alphavantage import AlphaVantage
from sentiment.finbert import get_sentiment, analyze_stock_sentiment

av = AlphaVantage()


class StockAnalyzer:
    """
    Comprehensive stock analyzer using multiple data sources
    
    WARNING: Each method makes multiple API calls. Monitor your rate limits!
    - Finnhub: 60/min (this is the bottleneck for real-time analysis)
    - Alpha Vantage: 25/day (very limited, use sparingly)
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.cache = {}
    
    def get_comprehensive_sentiment(self) -> Dict:
        """
        Get sentiment from multiple sources and compare:
        1. FinBERT sentiment on Finnhub news
        2. Alpha Vantage sentiment scores
        3. Finnhub insider sentiment
        4. Aggregated comparison
        """
        results = {
            'symbol': self.symbol,
            'timestamp': datetime.datetime.now().isoformat(),
            'sources': {}
        }
        
        # Source 1: FinBERT on Finnhub News
        print(f"Analyzing FinBERT sentiment for {self.symbol}...")
        print(f"  ⚠️  This uses Finnhub API (60 calls/min limit)")
        try:
            finbert_result = analyze_stock_sentiment(self.symbol, self.symbol, days=7)
            if finbert_result:
                articles_count = finbert_result.get('articles_analyzed', 0)
                print(f"  ✓ Analyzed {articles_count} articles")
                
                results['sources']['finbert_finnhub'] = {
                    'provider': 'FinBERT (AI) + Finnhub News',
                    'overall_sentiment': finbert_result.get('overall_sentiment', 'neutral'),
                    'confidence': finbert_result.get('confidence', 0),
                    'positive_ratio': finbert_result.get('positive_ratio', 0),
                    'negative_ratio': finbert_result.get('negative_ratio', 0),
                    'neutral_ratio': finbert_result.get('neutral_ratio', 0),
                    'articles_analyzed': articles_count,
                    'summary': finbert_result.get('summary', ''),
                    'score': self._calculate_sentiment_score(
                        finbert_result.get('positive_ratio', 0),
                        finbert_result.get('negative_ratio', 0)
                    )
                }
        except Exception as e:
            print(f"FinBERT analysis error: {e}")
            results['sources']['finbert_finnhub'] = {'error': str(e)}
        
        # Source 2: Alpha Vantage News Sentiment
        print(f"Getting Alpha Vantage sentiment for {self.symbol}...")
        print(f"  ⚠️  Alpha Vantage: 25 requests/day limit (use sparingly!)")
        try:
            av_news = av.get_market_news_sentiment(tickers=self.symbol, limit=50)
            if av_news and 'feed' in av_news:
                sentiments = []
                relevance_scores = []
                
                for article in av_news['feed']:
                    for ticker_data in article.get('ticker_sentiment', []):
                        if ticker_data.get('ticker') == self.symbol:
                            score = float(ticker_data.get('ticker_sentiment_score', 0))
                            sentiments.append(score)
                            relevance_scores.append(float(ticker_data.get('relevance_score', 0)))
                
                if sentiments:
                    print(f"  ✓ Found {len(sentiments)} sentiment scores")
                    avg_sentiment = sum(sentiments) / len(sentiments)
                    avg_relevance = sum(relevance_scores) / len(relevance_scores)
                    
                    # Convert to categorical
                    if avg_sentiment > 0.15:
                        category = 'positive'
                    elif avg_sentiment < -0.15:
                        category = 'negative'
                    else:
                        category = 'neutral'
                    
                    results['sources']['alphavantage'] = {
                        'provider': 'Alpha Vantage News Sentiment',
                        'overall_sentiment': category,
                        'sentiment_score': avg_sentiment,  # Range: -1 to 1
                        'relevance_score': avg_relevance,
                        'articles_analyzed': len(sentiments),
                        'score': (avg_sentiment + 1) * 50  # Convert to 0-100
                    }
        except Exception as e:
            print(f"Alpha Vantage sentiment error: {e}")
            results['sources']['alphavantage'] = {'error': str(e)}
        
        # Source 3: Finnhub Insider Sentiment
        print(f"Getting Finnhub insider sentiment for {self.symbol}...")
        try:
            # Get insider sentiment for last 3 months
            from_date = (datetime.datetime.now() - datetime.timedelta(days=90)).strftime('%Y-%m-%d')
            to_date = datetime.datetime.now().strftime('%Y-%m-%d')
            
            insider_data = get_insider_sentiment(self.symbol, from_date, to_date)
            if insider_data and 'data' in insider_data:
                recent = insider_data['data'][:3]  # Last 3 months
                if recent:
                    total_change = sum(d.get('change', 0) for d in recent)
                    total_mspr = sum(d.get('mspr', 0) for d in recent) / len(recent)
                    
                    # MSPR > 0 is bullish, < 0 is bearish
                    if total_mspr > 0.5:
                        category = 'positive'
                    elif total_mspr < -0.5:
                        category = 'negative'
                    else:
                        category = 'neutral'
                    
                    results['sources']['finnhub_insider'] = {
                        'provider': 'Finnhub Insider Trading Sentiment',
                        'overall_sentiment': category,
                        'mspr': total_mspr,  # Monthly Share Purchase Ratio
                        'net_shares_change': total_change,
                        'months_analyzed': len(recent),
                        'score': max(0, min(100, 50 + (total_mspr * 50)))
                    }
        except Exception as e:
            print(f"Insider sentiment error: {e}")
            results['sources']['finnhub_insider'] = {'error': str(e)}
        
        # Aggregate and compare
        results['comparison'] = self._compare_sentiments(results['sources'])
        results['consensus'] = self._get_consensus(results['sources'])
        
        return results
    
    def get_comprehensive_data(self) -> Dict:
        """Get all data for a stock from all sources"""
        data = {
            'symbol': self.symbol,
            'timestamp': datetime.datetime.now().isoformat(),
            'company': {},
            'quote': {},
            'financials': {},
            'news': {},
            'sentiment': {},
            'insider_activity': {},
            'earnings': {}
        }
        
        # Company Profile
        try:
            profile = get_company_profile(self.symbol)
            if profile:
                data['company'] = profile
        except Exception as e:
            print(f"Company profile error: {e}")
        
        # Quote Data
        try:
            quote = av.get_global_quote(self.symbol)
            if quote:
                data['quote'] = {
                    'price': float(quote.get('05. price', 0)),
                    'change': float(quote.get('09. change', 0)),
                    'change_percent': quote.get('10. change percent', '0%'),
                    'volume': int(quote.get('06. volume', 0)),
                    'latest_trading_day': quote.get('07. latest trading day', '')
                }
        except Exception as e:
            print(f"Quote error: {e}")
        
        # Financials
        try:
            finnhub_financials = get_basic_financials(self.symbol)
            if finnhub_financials:
                metrics = finnhub_financials.get('metric', {})
                data['financials'] = {
                    'pe_ratio': metrics.get('peBasicExclExtraTTM'),
                    'eps': metrics.get('epsBasicExclExtraItemsTTM'),
                    'eps_growth': metrics.get('epsGrowthTTMYoy'),
                    'revenue_growth': metrics.get('revenueGrowthTTMYoy'),
                    'profit_margin': metrics.get('netProfitMarginTTM'),
                    'roe': metrics.get('roeTTM'),
                    'debt_to_equity': metrics.get('totalDebt/totalEquityQuarterly'),
                    'market_cap': metrics.get('marketCapitalization'),
                    '52w_high': metrics.get('52WeekHigh'),
                    '52w_low': metrics.get('52WeekLow')
                }
        except Exception as e:
            print(f"Financials error: {e}")
        
        # News from multiple sources
        try:
            finnhub_news = get_company_news(
                self.symbol,
                (datetime.date.today() - datetime.timedelta(days=7)).strftime('%Y-%m-%d'),
                datetime.date.today().strftime('%Y-%m-%d')
            )
            
            av_news = av.get_market_news_sentiment(tickers=self.symbol, limit=20)
            
            data['news'] = {
                'finnhub': finnhub_news[:10] if finnhub_news else [],
                'alphavantage': av_news.get('feed', [])[:10] if av_news else [],
                'total_articles': len(finnhub_news or []) + len(av_news.get('feed', []) if av_news else [])
            }
        except Exception as e:
            print(f"News error: {e}")
        
        # Comprehensive Sentiment
        data['sentiment'] = self.get_comprehensive_sentiment()
        
        # Insider Activity
        try:
            insider_trans = get_insider_transactions(self.symbol)
            insider_sent = get_insider_sentiment(self.symbol)
            
            data['insider_activity'] = {
                'transactions': insider_trans.get('data', [])[:10] if insider_trans else [],
                'sentiment': insider_sent.get('data', [])[:3] if insider_sent else []
            }
        except Exception as e:
            print(f"Insider activity error: {e}")
        
        # Earnings
        try:
            earnings = get_earnings_surprises(self.symbol)
            if earnings:
                data['earnings'] = earnings[:4]  # Last 4 quarters
        except Exception as e:
            print(f"Earnings error: {e}")
        
        return data
    
    def _calculate_sentiment_score(self, positive: float, negative: float) -> float:
        """Convert sentiment ratios to 0-100 score"""
        return (positive - negative + 1) * 50
    
    def _compare_sentiments(self, sources: Dict) -> Dict:
        """Compare sentiments across sources"""
        comparison = {
            'agreement_level': 'unknown',
            'scores': {},
            'sentiments': {},
            'average_score': 0,
            'variance': 0
        }
        
        scores = []
        sentiments = []
        
        for source_name, source_data in sources.items():
            if 'error' not in source_data:
                if 'score' in source_data:
                    scores.append(source_data['score'])
                    comparison['scores'][source_name] = source_data['score']
                
                if 'overall_sentiment' in source_data:
                    sentiments.append(source_data['overall_sentiment'])
                    comparison['sentiments'][source_name] = source_data['overall_sentiment']
        
        if scores:
            comparison['average_score'] = sum(scores) / len(scores)
            comparison['variance'] = sum((s - comparison['average_score']) ** 2 for s in scores) / len(scores)
            
            # Determine agreement level
            if comparison['variance'] < 100:
                comparison['agreement_level'] = 'strong'
            elif comparison['variance'] < 300:
                comparison['agreement_level'] = 'moderate'
            else:
                comparison['agreement_level'] = 'weak'
        
        # Check if sentiments agree
        if sentiments:
            unique_sentiments = set(sentiments)
            if len(unique_sentiments) == 1:
                comparison['sentiment_consensus'] = 'unanimous'
            elif len(unique_sentiments) == 2:
                comparison['sentiment_consensus'] = 'mixed'
            else:
                comparison['sentiment_consensus'] = 'divided'
        
        return comparison
    
    def _get_consensus(self, sources: Dict) -> Dict:
        """Get overall consensus from all sources"""
        sentiments = []
        scores = []
        
        for source_data in sources.values():
            if 'error' not in source_data:
                if 'overall_sentiment' in source_data:
                    sentiments.append(source_data['overall_sentiment'])
                if 'score' in source_data:
                    scores.append(source_data['score'])
        
        if not sentiments:
            return {'sentiment': 'unknown', 'confidence': 0, 'score': 50}
        
        # Count sentiment types
        positive_count = sentiments.count('positive')
        negative_count = sentiments.count('negative')
        neutral_count = sentiments.count('neutral')
        
        total = len(sentiments)
        
        if positive_count > negative_count and positive_count > neutral_count:
            overall = 'positive'
            confidence = positive_count / total
        elif negative_count > positive_count and negative_count > neutral_count:
            overall = 'negative'
            confidence = negative_count / total
        else:
            overall = 'neutral'
            confidence = max(positive_count, negative_count, neutral_count) / total
        
        avg_score = sum(scores) / len(scores) if scores else 50
        
        return {
            'sentiment': overall,
            'confidence': confidence,
            'score': avg_score,
            'breakdown': {
                'positive': positive_count,
                'neutral': neutral_count,
                'negative': negative_count
            }
        }
    
    def get_enhanced_scenarios(self, timeframe: str = '1M') -> Dict:
        """Generate scenarios using comprehensive data"""
        data = self.get_comprehensive_data()
        
        financials = data.get('financials', {})
        sentiment = data.get('sentiment', {})
        quote = data.get('quote', {})
        
        current_price = quote.get('price', 0)
        eps_growth = financials.get('eps_growth', 10) or 10
        sentiment_score = sentiment.get('consensus', {}).get('score', 50)
        
        # Sentiment adjustment factor
        sentiment_factor = (sentiment_score - 50) / 100  # -0.5 to +0.5
        
        # Timeframe multiplier
        timeframe_days = {
            '1W': 7, '1M': 30, '3M': 90, '6M': 180, '1Y': 365
        }.get(timeframe, 30)
        
        multiplier = timeframe_days / 365
        
        # Calculate scenarios
        base_growth = eps_growth / 100 * multiplier
        
        scenarios = {
            'bull_case': {
                'price_target': current_price * (1 + base_growth * 1.5 + sentiment_factor * 0.2),
                'probability': 25,
                'return': 0,
                'factors': [
                    f'EPS growth exceeds {eps_growth}% forecast',
                    'Strong positive sentiment across all sources',
                    'Sector momentum remains strong',
                    'Market conditions favorable'
                ],
                'rationale': f'Optimistic scenario assuming {int(eps_growth * 1.5)}% EPS growth with positive market sentiment.'
            },
            'base_case': {
                'price_target': current_price * (1 + base_growth + sentiment_factor * 0.1),
                'probability': 50,
                'return': 0,
                'factors': [
                    f'EPS growth meets {eps_growth}% expectations',
                    'Sentiment remains stable',
                    'Normal market conditions',
                    'Company executes on plan'
                ],
                'rationale': f'Most likely scenario with {eps_growth}% EPS growth and stable market conditions.'
            },
            'bear_case': {
                'price_target': current_price * (1 + base_growth * 0.3 - abs(sentiment_factor) * 0.2),
                'probability': 25,
                'return': 0,
                'factors': [
                    'EPS growth disappoints',
                    'Negative sentiment develops',
                    'Market correction or sector weakness',
                    'Execution challenges emerge'
                ],
                'rationale': f'Conservative scenario with only {int(eps_growth * 0.3)}% growth and market headwinds.'
            }
        }
        
        # Calculate returns
        for scenario in scenarios.values():
            scenario['return'] = ((scenario['price_target'] - current_price) / current_price) * 100
        
        return {
            'symbol': self.symbol,
            'current_price': current_price,
            'timeframe': timeframe,
            'sentiment_score': sentiment_score,
            'eps_growth': eps_growth,
            'scenarios': scenarios,
            'data_sources': list(sentiment.get('sources', {}).keys())
        }
    
    def get_detailed_metrics(self) -> Dict:
        """Get detailed metrics with comparisons"""
        data = self.get_comprehensive_data()
        financials = data.get('financials', {})
        
        # Extract metrics
        pe_ratio = financials.get('pe_ratio', 20) or 20
        roe = financials.get('roe', 15) or 15
        profit_margin = financials.get('profit_margin', 10) or 10
        eps_growth = financials.get('eps_growth', 10) or 10
        revenue_growth = financials.get('revenue_growth', 10) or 10
        debt_to_equity = financials.get('debt_to_equity', 0.5) or 0.5
        
        # Scoring functions
        def grade_valuation(pe: float) -> tuple:
            if pe < 15:
                return 'A', 95, 'Undervalued - PE ratio significantly below market average'
            elif pe < 20:
                return 'B', 85, 'Fair value - PE ratio near market average'
            elif pe < 25:
                return 'C', 75, 'Market value - PE ratio at market average'
            elif pe < 35:
                return 'D', 60, 'Overvalued - PE ratio above market average'
            else:
                return 'F', 40, 'Significantly overvalued'
        
        def grade_profitability(roe: float, margin: float) -> tuple:
            avg = (roe + margin) / 2
            if avg > 20:
                return 'A', 95, 'Excellent profitability metrics'
            elif avg > 15:
                return 'B', 85, 'Strong profitability'
            elif avg > 10:
                return 'C', 75, 'Adequate profitability'
            elif avg > 5:
                return 'D', 60, 'Below average profitability'
            else:
                return 'F', 40, 'Weak profitability'
        
        def grade_growth(eps: float, revenue: float) -> tuple:
            avg = (eps + revenue) / 2
            if avg > 20:
                return 'A', 95, 'High growth company'
            elif avg > 15:
                return 'B', 85, 'Strong growth'
            elif avg > 10:
                return 'C', 75, 'Moderate growth'
            elif avg > 5:
                return 'D', 60, 'Slow growth'
            else:
                return 'F', 40, 'Declining or no growth'
        
        def grade_health(debt: float) -> tuple:
            if debt < 0.3:
                return 'A', 95, 'Excellent financial health - Low debt'
            elif debt < 0.5:
                return 'B', 85, 'Good financial health'
            elif debt < 1.0:
                return 'C', 75, 'Moderate debt levels'
            elif debt < 2.0:
                return 'D', 60, 'High debt levels'
            else:
                return 'F', 40, 'Very high debt - Financial risk'
        
        # Grade each category
        val_grade, val_score, val_desc = grade_valuation(pe_ratio)
        prof_grade, prof_score, prof_desc = grade_profitability(roe, profit_margin)
        growth_grade, growth_score, growth_desc = grade_growth(eps_growth, revenue_growth)
        health_grade, health_score, health_desc = grade_health(debt_to_equity)
        
        # Overall grade
        avg_score = (val_score + prof_score + growth_score + health_score) / 4
        if avg_score >= 90:
            overall_grade = 'A'
        elif avg_score >= 80:
            overall_grade = 'B'
        elif avg_score >= 70:
            overall_grade = 'C'
        elif avg_score >= 60:
            overall_grade = 'D'
        else:
            overall_grade = 'F'
        
        return {
            'symbol': self.symbol,
            'overall_grade': overall_grade,
            'average_score': avg_score,
            'metrics': {
                'valuation': {
                    'grade': val_grade,
                    'score': val_score,
                    'description': val_desc,
                    'pe_ratio': pe_ratio,
                    'factors': ['P/E Ratio', 'Price to Book', 'Price to Sales']
                },
                'profitability': {
                    'grade': prof_grade,
                    'score': prof_score,
                    'description': prof_desc,
                    'roe': roe,
                    'profit_margin': profit_margin,
                    'factors': ['ROE', 'Profit Margin', 'ROA']
                },
                'growth': {
                    'grade': growth_grade,
                    'score': growth_score,
                    'description': growth_desc,
                    'eps_growth': eps_growth,
                    'revenue_growth': revenue_growth,
                    'factors': ['EPS Growth', 'Revenue Growth', 'Market Share']
                },
                'financial_health': {
                    'grade': health_grade,
                    'score': health_score,
                    'description': health_desc,
                    'debt_to_equity': debt_to_equity,
                    'factors': ['Debt/Equity', 'Current Ratio', 'Quick Ratio']
                }
            }
        }


# Utility functions for direct use
def analyze_stock(symbol: str) -> Dict:
    """Quick function to analyze a stock comprehensively"""
    analyzer = StockAnalyzer(symbol)
    return analyzer.get_comprehensive_data()


def compare_sentiment_sources(symbol: str) -> Dict:
    """Quick function to compare sentiment from all sources"""
    analyzer = StockAnalyzer(symbol)
    return analyzer.get_comprehensive_sentiment()


if __name__ == '__main__':
    # Test the analyzer
    import sys
    
    test_symbol = sys.argv[1] if len(sys.argv) > 1 else 'NVDA'
    
    print(f"\n{'='*60}")
    print(f"Comprehensive Analysis for {test_symbol}")
    print(f"{'='*60}\n")
    
    analyzer = StockAnalyzer(test_symbol)
    
    # Test sentiment comparison
    print("1. SENTIMENT COMPARISON")
    print("-" * 60)
    sentiment = analyzer.get_comprehensive_sentiment()
    print(json.dumps(sentiment, indent=2))
    
    print("\n2. COMPREHENSIVE DATA")
    print("-" * 60)
    data = analyzer.get_comprehensive_data()
    print(f"Company: {data.get('company', {}).get('name', test_symbol)}")
    print(f"Price: ${data.get('quote', {}).get('price', 0):.2f}")
    print(f"Total News Articles: {data.get('news', {}).get('total_articles', 0)}")
    print(f"Consensus Sentiment: {data.get('sentiment', {}).get('consensus', {}).get('sentiment', 'unknown')}")
    
    print("\n3. ENHANCED SCENARIOS")
    print("-" * 60)
    scenarios = analyzer.get_enhanced_scenarios('1M')
    print(json.dumps(scenarios, indent=2))
    
    print("\n4. DETAILED METRICS")
    print("-" * 60)
    metrics = analyzer.get_detailed_metrics()
    print(json.dumps(metrics, indent=2))
