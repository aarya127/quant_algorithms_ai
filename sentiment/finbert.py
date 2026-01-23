from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import datetime
import sys
import os

# Add parent directory to path to import finnhub functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.finnhub import get_company_news, get_basic_financials, get_earnings_surprises

# Initialize FinBERT
print("Loading FinBERT model...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
print("✓ Model loaded\n")

# Stock list - US stocks only (for Finnhub)
# For Canadian stocks, use finbert_canadian.py with Alpha Vantage
stocks = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp.",
    "TSLA": "Tesla Inc.",
    "NVDA": "NVIDIA Corp.",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon",
    "META": "Meta Platforms",
}


def get_sentiment(text):
    """Analyze sentiment of financial text using FinBERT"""
    if not text or len(text.strip()) < 10:
        return None, None
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    sentiment_score = predictions[0].tolist()
    
    if hasattr(model.config, 'id2label'):
        labels = [model.config.id2label[i] for i in range(len(sentiment_score))]
    else:
        labels = ['positive', 'negative', 'neutral']
    
    result = {labels[i]: sentiment_score[i] for i in range(len(labels))}
    dominant_sentiment = labels[torch.argmax(predictions[0]).item()]
    
    return dominant_sentiment, result


def analyze_stock_sentiment(symbol, company_name, days=7):
    """Comprehensive sentiment analysis for a stock using Finnhub data"""
    print(f"\n{'='*70}")
    print(f"📊 ANALYZING: {company_name} ({symbol})")
    print(f"{'='*70}")
    
    today = datetime.date.today()
    from_date = (today - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
    to_date = today.strftime('%Y-%m-%d')
    
    sentiment_scores = []
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    
    # 1. Analyze Company News
    print(f"\n📰 Fetching news from {from_date} to {to_date}...")
    try:
        news_articles = get_company_news(symbol, from_date, to_date)
        
        if news_articles and len(news_articles) > 0:
            print(f"   Found {len(news_articles)} news articles")
            
            for i, article in enumerate(news_articles[:10]):  # Analyze top 10 articles
                headline = article.get('headline', '')
                summary = article.get('summary', '')
                
                # Combine headline and summary for better context
                text = f"{headline}. {summary}"
                
                sentiment, scores = get_sentiment(text)
                
                if sentiment and scores:
                    sentiment_scores.append(scores)
                    
                    if sentiment == 'positive':
                        positive_count += 1
                    elif sentiment == 'negative':
                        negative_count += 1
                    else:
                        neutral_count += 1
                    
                    # Print first 3 articles with details
                    if i < 3:
                        print(f"\n   Article {i+1}: {headline[:80]}...")
                        print(f"   Sentiment: {sentiment.upper()}")
                        print(f"   Scores: Pos={scores['positive']:.2f}, Neg={scores['negative']:.2f}, Neu={scores['neutral']:.2f}")
        else:
            print("   ⚠️  No news articles found")
    except Exception as e:
        print(f"   ❌ Error fetching news: {str(e)}")
    
    # 2. Analyze Earnings Data
    print(f"\n💰 Fetching earnings data...")
    try:
        earnings = get_earnings_surprises(symbol)
        
        if earnings and len(earnings) > 0:
            latest_earnings = earnings[0]
            actual = latest_earnings.get('actual')
            estimate = latest_earnings.get('estimate')
            
            if actual and estimate:
                surprise = actual - estimate
                surprise_pct = (surprise / estimate * 100) if estimate != 0 else 0
                
                # Create earnings text
                if surprise_pct > 5:
                    earnings_text = f"{company_name} significantly beat earnings expectations by {surprise_pct:.1f}%, reporting actual EPS of ${actual} versus estimated ${estimate}."
                elif surprise_pct < -5:
                    earnings_text = f"{company_name} missed earnings expectations by {abs(surprise_pct):.1f}%, reporting actual EPS of ${actual} versus estimated ${estimate}."
                else:
                    earnings_text = f"{company_name} met earnings expectations, reporting actual EPS of ${actual} versus estimated ${estimate}."
                
                print(f"   Latest earnings: Actual=${actual}, Estimate=${estimate}, Surprise={surprise_pct:.1f}%")
                
                sentiment, scores = get_sentiment(earnings_text)
                if sentiment and scores:
                    sentiment_scores.append(scores)
                    print(f"   Earnings sentiment: {sentiment.upper()}")
                    
                    if sentiment == 'positive':
                        positive_count += 1
                    elif sentiment == 'negative':
                        negative_count += 1
                    else:
                        neutral_count += 1
        else:
            print("   ℹ️  No recent earnings data available")
    except Exception as e:
        print(f"   ⚠️  Could not fetch earnings: {str(e)}")
    
    # 3. Calculate Overall Sentiment
    print(f"\n📈 OVERALL SENTIMENT ANALYSIS")
    print(f"{'─'*70}")
    
    if sentiment_scores:
        # Average all sentiment scores
        avg_positive = sum(s['positive'] for s in sentiment_scores) / len(sentiment_scores)
        avg_negative = sum(s['negative'] for s in sentiment_scores) / len(sentiment_scores)
        avg_neutral = sum(s['neutral'] for s in sentiment_scores) / len(sentiment_scores)
        
        # Determine overall sentiment
        max_score = max(avg_positive, avg_negative, avg_neutral)
        if max_score == avg_positive:
            overall = "POSITIVE 📈"
        elif max_score == avg_negative:
            overall = "NEGATIVE 📉"
        else:
            overall = "NEUTRAL ➡️"
        
        print(f"   Sources analyzed: {len(sentiment_scores)}")
        print(f"   Distribution: {positive_count} positive, {negative_count} negative, {neutral_count} neutral")
        print(f"\n   Average Scores:")
        print(f"   • Positive: {avg_positive:.1%}")
        print(f"   • Negative: {avg_negative:.1%}")
        print(f"   • Neutral:  {avg_neutral:.1%}")
        print(f"\n   🎯 OVERALL SENTIMENT: {overall}")
        
        # Sentiment strength
        confidence = max_score
        if confidence > 0.7:
            strength = "STRONG"
        elif confidence > 0.5:
            strength = "MODERATE"
        else:
            strength = "WEAK"
        print(f"   📊 Confidence: {strength} ({confidence:.1%})")
        
        total_count = positive_count + negative_count + neutral_count
        
        return {
            'symbol': symbol,
            'company': company_name,
            'overall_sentiment': overall.split()[0].lower(),  # lowercase for consistency
            'positive': avg_positive,
            'negative': avg_negative,
            'neutral': avg_neutral,
            'confidence': confidence,
            'sources_count': len(sentiment_scores),
            'articles_analyzed': len(sentiment_scores),  # Added for stock_analyzer.py compatibility
            'positive_ratio': positive_count / total_count if total_count > 0 else 0,
            'negative_ratio': negative_count / total_count if total_count > 0 else 0,
            'neutral_ratio': neutral_count / total_count if total_count > 0 else 0,
            'distribution': {'positive': positive_count, 'negative': negative_count, 'neutral': neutral_count},
            'summary': f"Analyzed {len(sentiment_scores)} sources: {positive_count} positive, {negative_count} negative, {neutral_count} neutral"
        }
    else:
        print("   ⚠️  No data available for sentiment analysis")
        return None


def main():
    """Run sentiment analysis on US stocks"""
    print("\n" + "="*70)
    print("🚀 US STOCK SENTIMENT ANALYSIS")
    print("   Using FinBERT + Finnhub Real-Time Data")
    print("   For Canadian stocks, use finbert_canadian.py")
    print("="*70)
    
    results = []
    
    for symbol, company_name in stocks.items():
        try:
            result = analyze_stock_sentiment(symbol, company_name, days=7)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\n❌ Error analyzing {symbol}: {str(e)}")
    
    # Summary table
    if results:
        print(f"\n\n{'='*70}")
        print("📊 SUMMARY TABLE")
        print(f"{'='*70}")
        print(f"{'Symbol':<12} {'Company':<20} {'Sentiment':<12} {'Pos':<8} {'Neg':<8} {'Sources':<8}")
        print(f"{'-'*70}")
        
        for r in results:
            print(f"{r['symbol']:<12} {r['company'][:20]:<20} {r['overall_sentiment']:<12} "
                  f"{r['positive']:<8.1%} {r['negative']:<8.1%} {r['sources_count']:<8}")
        
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()