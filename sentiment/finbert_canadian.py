from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import sys
import os

# Add parent directory to path to import alphavantage functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Data.alphavantage import AlphaVantage

# Initialize FinBERT
print("Loading FinBERT model...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
print("✓ Model loaded\n")

# Canadian stocks on TSX
canadian_stocks = {
    "AC.TO": "Air Canada",
    "TD.TO": "TD Bank",
    "ENB.TO": "Enbridge",
    "RCI-B.TO": "Rogers Communications",
    "SHOP.TO": "Shopify",
    "BMO.TO": "Bank of Montreal",
    "RY.TO": "Royal Bank of Canada",
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


def analyze_canadian_stock_sentiment(symbol, company_name):
    """Comprehensive sentiment analysis for Canadian stocks using Alpha Vantage"""
    print(f"\n{'='*70}")
    print(f"📊 ANALYZING: {company_name} ({symbol})")
    print(f"{'='*70}")
    
    sentiment_scores = []
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    
    # Initialize Alpha Vantage
    av = AlphaVantage()
    
    # 1. Analyze Company News using Alpha Vantage
    print(f"\n📰 Fetching news from Alpha Vantage...")
    try:
        news_data = av.get_market_news_sentiment(tickers=symbol, limit=50)
        
        if 'feed' in news_data and len(news_data['feed']) > 0:
            articles = news_data['feed']
            print(f"   Found {len(articles)} news articles")
            
            for i, article in enumerate(articles[:10]):  # Analyze top 10 articles
                headline = article.get('title', '')
                summary = article.get('summary', '')
                
                # Alpha Vantage also provides its own sentiment
                av_sentiment = article.get('overall_sentiment_label', 'Neutral')
                av_score = float(article.get('overall_sentiment_score', 0))
                
                # Combine headline and summary for FinBERT analysis
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
                        print(f"   FinBERT Sentiment: {sentiment.upper()}")
                        print(f"   FinBERT Scores: Pos={scores['positive']:.2f}, Neg={scores['negative']:.2f}, Neu={scores['neutral']:.2f}")
                        print(f"   Alpha Vantage: {av_sentiment} (score: {av_score:.2f})")
        else:
            print("   ⚠️  No news articles found")
            
            # Check if we hit API limit
            if 'Note' in news_data:
                print(f"   ℹ️  {news_data['Note']}")
    except Exception as e:
        print(f"   ❌ Error fetching news: {str(e)}")
    
    # 2. Get Company Overview
    print(f"\n🏢 Fetching company overview...")
    try:
        overview = av.get_company_overview(symbol)
        
        if overview and 'Symbol' in overview:
            sector = overview.get('Sector', 'N/A')
            industry = overview.get('Industry', 'N/A')
            market_cap = overview.get('MarketCapitalization', 'N/A')
            pe_ratio = overview.get('PERatio', 'N/A')
            
            print(f"   Sector: {sector}")
            print(f"   Industry: {industry}")
            print(f"   Market Cap: ${float(market_cap)/1e9:.2f}B" if market_cap != 'N/A' else f"   Market Cap: {market_cap}")
            print(f"   P/E Ratio: {pe_ratio}")
            
            # Analyze company description
            description = overview.get('Description', '')
            if description:
                desc_sentiment, desc_scores = get_sentiment(description[:500])  # First 500 chars
                if desc_sentiment and desc_scores:
                    print(f"   Company Description Sentiment: {desc_sentiment.upper()}")
        else:
            print("   ℹ️  No company overview available")
    except Exception as e:
        print(f"   ⚠️  Could not fetch company overview: {str(e)}")
    
    # 3. Get Quote Data
    print(f"\n💹 Fetching current quote...")
    try:
        quote = av.get_global_quote(symbol)
        
        if quote and '05. price' in quote:
            price = float(quote.get('05. price', 0))
            change = float(quote.get('09. change', 0))
            change_pct = float(quote.get('10. change percent', '0').replace('%', ''))
            
            print(f"   Current Price: ${price:.2f}")
            print(f"   Change: ${change:.2f} ({change_pct:+.2f}%)")
            
            # Create sentiment text based on price movement
            if change_pct > 2:
                price_text = f"{company_name} stock is up {change_pct:.1f}% today, showing strong positive momentum."
            elif change_pct < -2:
                price_text = f"{company_name} stock is down {abs(change_pct):.1f}% today, indicating market concerns."
            else:
                price_text = f"{company_name} stock is relatively stable today with a {change_pct:+.1f}% change."
            
            price_sentiment, price_scores = get_sentiment(price_text)
            if price_sentiment and price_scores:
                sentiment_scores.append(price_scores)
                print(f"   Price Movement Sentiment: {price_sentiment.upper()}")
                
                if price_sentiment == 'positive':
                    positive_count += 1
                elif price_sentiment == 'negative':
                    negative_count += 1
                else:
                    neutral_count += 1
        else:
            print("   ℹ️  No quote data available")
    except Exception as e:
        print(f"   ⚠️  Could not fetch quote: {str(e)}")
    
    # 4. Calculate Overall Sentiment
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
        
        return {
            'symbol': symbol,
            'company': company_name,
            'overall_sentiment': overall.split()[0],
            'positive': avg_positive,
            'negative': avg_negative,
            'neutral': avg_neutral,
            'confidence': confidence,
            'sources_count': len(sentiment_scores),
            'distribution': {'positive': positive_count, 'negative': negative_count, 'neutral': neutral_count}
        }
    else:
        print("   ⚠️  No data available for sentiment analysis")
        return None


def main():
    """Run sentiment analysis on Canadian stocks"""
    print("\n" + "="*70)
    print("🍁 CANADIAN STOCK SENTIMENT ANALYSIS")
    print("   Using FinBERT + Alpha Vantage")
    print("   Note: Alpha Vantage free tier = 25 requests/day")
    print("="*70)
    
    results = []
    
    # Test with Air Canada first (to conserve API calls)
    test_stocks = {"AC.TO": "Air Canada"}
    
    for symbol, company_name in test_stocks.items():
        try:
            result = analyze_canadian_stock_sentiment(symbol, company_name)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\n❌ Error analyzing {symbol}: {str(e)}")
    
    # Summary table
    if results:
        print(f"\n\n{'='*70}")
        print("📊 SUMMARY TABLE - CANADIAN STOCKS")
        print(f"{'='*70}")
        print(f"{'Symbol':<12} {'Company':<20} {'Sentiment':<12} {'Pos':<8} {'Neg':<8} {'Sources':<8}")
        print(f"{'-'*70}")
        
        for r in results:
            print(f"{r['symbol']:<12} {r['company'][:20]:<20} {r['overall_sentiment']:<12} "
                  f"{r['positive']:<8.1%} {r['negative']:<8.1%} {r['sources_count']:<8}")
        
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
