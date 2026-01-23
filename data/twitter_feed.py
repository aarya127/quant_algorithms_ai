"""
Twitter Feed Integration using Tweepy
Fetches latest tweets from financial news accounts and market-related hashtags
"""

import tweepy
import logging
from datetime import datetime
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Twitter API credentials
API_KEY = "47a9sg2P51QWZ5dyTPKtrt5bw"
API_SECRET = "JxTLFVGQ5dFcuUFZMpwGNPP9iHwZRLBbRUdpcrNnxw8klPjpGW"
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAFJd5gEAAAAAHDlDO2S7ydz3eJ6oMxxkHtYAAsI%3DDF5b4CUyYTslNFqLj83BFaPBEES9coi5NczEXoMXvS9EKhDJi7"

# Market-influencing accounts to follow (reduces API calls significantly)
INFLUENTIAL_ACCOUNTS = [
    "realDonaldTrump", # Former President - major market influence
    "elonmusk",      # Tesla/SpaceX CEO - major market mover
    "CathieDWood",   # ARK Invest CEO
    "jimcramer",     # CNBC Mad Money
    "carl_c_icahn",  # Activist investor
    "BillAckman",    # Pershing Square Capital
    "michaeljburry", # The Big Short investor
    "chamath",       # Social Capital founder
    "raydalio",      # Bridgewater Associates founder
    "WarrenBuffett", # Berkshire Hathaway CEO
    "elonmusk",      # Tesla/SpaceX CEO
    "BarryRitholtz", # Ritholtz Wealth Management
    "ScottWapnerCNBC", # CNBC Fast Money host
    "StanleyDruckenmiller", # Duquesne Family Office
    "howardmarks",   # Oaktree Capital co-founder
    "DavidEinhorn",  # Greenlight Capital
    "KirklandLake",  # Activist investor
    "StephenCushing", # Infrastructure investor
]

# Financial news accounts to follow
FINANCIAL_ACCOUNTS = [
    "WSJ",           # Wall Street Journal
    "markets",       # Bloomberg Markets
    "FT",            # Financial Times
    "Reuters",       # Reuters
    "CNBC",          # CNBC
    "YahooFinance",  # Yahoo Finance
    "MarketWatch",   # MarketWatch
    "BarronOnline",  # Barrons
    "business",      # Bloomberg Business
]

def get_twitter_client():
    """Initialize and return Twitter API v2 client"""
    try:
        client = tweepy.Client(
            bearer_token=BEARER_TOKEN,
            consumer_key=API_KEY,
            consumer_secret=API_SECRET,
            wait_on_rate_limit=True
        )
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Twitter client: {str(e)}")
        return None

def get_market_tweets(symbol: str = None, count: int = 20) -> List[Dict]:
    """
    Fetch latest tweets from influential market figures and financial news accounts only
    (Optimized to reduce API calls by focusing on high-impact accounts)
    
    Args:
        symbol: Optional stock symbol to filter tweets
        count: Number of tweets to fetch (max 100)
    
    Returns:
        List of tweet dictionaries with text, author, timestamp, metrics
    """
    client = get_twitter_client()
    if not client:
        return []
    
    try:
        # Build targeted query focusing on influential accounts only
        accounts_query = " OR ".join([f"from:{acc}" for acc in INFLUENTIAL_ACCOUNTS + FINANCIAL_ACCOUNTS])
        
        if symbol:
            # Search for tweets from influential accounts mentioning the stock symbol
            query = f"({accounts_query}) (${symbol} OR #{symbol}) -is:retweet lang:en"
        else:
            # Get tweets from influential accounts about markets
            query = f"({accounts_query}) (stocks OR market OR trading OR Fed OR earnings) -is:retweet lang:en"
        
        # Fetch tweets
        tweets = client.search_recent_tweets(
            query=query,
            max_results=min(count, 100),
            tweet_fields=['created_at', 'public_metrics', 'entities', 'author_id'],
            expansions=['author_id'],
            user_fields=['username', 'name', 'verified', 'profile_image_url']
        )
        
        if not tweets.data:
            return []
        
        # Create user lookup dictionary
        users = {user.id: user for user in tweets.includes.get('users', [])}
        
        # Format tweets
        formatted_tweets = []
        for tweet in tweets.data:
            author = users.get(tweet.author_id)
            
            formatted_tweet = {
                'id': tweet.id,
                'text': tweet.text,
                'created_at': tweet.created_at.isoformat(),
                'author': {
                    'username': author.username if author else 'Unknown',
                    'name': author.name if author else 'Unknown',
                    'verified': author.verified if author else False,
                    'profile_image': author.profile_image_url if author else None
                },
                'metrics': {
                    'likes': tweet.public_metrics['like_count'],
                    'retweets': tweet.public_metrics['retweet_count'],
                    'replies': tweet.public_metrics['reply_count']
                },
                'url': f"https://twitter.com/{author.username}/status/{tweet.id}" if author else None,
                'source': 'twitter'
            }
            
            # Extract symbols from entities if available
            if hasattr(tweet, 'entities') and tweet.entities:
                cashtags = tweet.entities.get('cashtags', [])
                hashtags = tweet.entities.get('hashtags', [])
                symbols = [tag['tag'] for tag in cashtags]
                formatted_tweet['symbols'] = symbols
                formatted_tweet['hashtags'] = [tag['tag'] for tag in hashtags]
            
            formatted_tweets.append(formatted_tweet)
        
        logger.info(f"Fetched {len(formatted_tweets)} tweets for query: {query}")
        return formatted_tweets
        
    except tweepy.errors.TooManyRequests:
        logger.warning("Twitter API rate limit exceeded")
        return []
    except Exception as e:
        logger.error(f"Error fetching tweets: {str(e)}")
        return []

def get_user_timeline(username: str, count: int = 10) -> List[Dict]:
    """
    Fetch latest tweets from a specific user
    
    Args:
        username: Twitter username (without @)
        count: Number of tweets to fetch
    
    Returns:
        List of tweet dictionaries
    """
    client = get_twitter_client()
    if not client:
        return []
    
    try:
        # Get user by username
        user = client.get_user(username=username, user_fields=['profile_image_url', 'verified'])
        
        if not user.data:
            return []
        
        # Get user's tweets
        tweets = client.get_users_tweets(
            id=user.data.id,
            max_results=min(count, 100),
            tweet_fields=['created_at', 'public_metrics', 'entities'],
            exclude=['retweets']
        )
        
        if not tweets.data:
            return []
        
        # Format tweets
        formatted_tweets = []
        for tweet in tweets.data:
            formatted_tweet = {
                'id': tweet.id,
                'text': tweet.text,
                'created_at': tweet.created_at.isoformat(),
                'author': {
                    'username': username,
                    'name': user.data.name,
                    'verified': user.data.verified,
                    'profile_image': user.data.profile_image_url
                },
                'metrics': {
                    'likes': tweet.public_metrics['like_count'],
                    'retweets': tweet.public_metrics['retweet_count'],
                    'replies': tweet.public_metrics['reply_count']
                },
                'url': f"https://twitter.com/{username}/status/{tweet.id}",
                'source': 'twitter'
            }
            
            formatted_tweets.append(formatted_tweet)
        
        return formatted_tweets
        
    except Exception as e:
        logger.error(f"Error fetching user timeline for {username}: {str(e)}")
        return []

def get_financial_news_feed(count: int = 30) -> List[Dict]:
    """
    Fetch latest tweets from major financial news accounts
    
    Args:
        count: Number of tweets to fetch per account
    
    Returns:
        Combined list of tweets from all financial news sources
    """
    all_tweets = []
    
    for account in FINANCIAL_ACCOUNTS[:5]:  # Limit to first 5 to avoid rate limits
        tweets = get_user_timeline(account, count=5)
        all_tweets.extend(tweets)
    
    # Sort by creation time (newest first)
    all_tweets.sort(key=lambda x: x['created_at'], reverse=True)
    
    return all_tweets[:count]

if __name__ == "__main__":
    # Test the Twitter feed
    print("Fetching general market tweets...")
    tweets = get_market_tweets(count=5)
    for tweet in tweets:
        print(f"\n@{tweet['author']['username']}: {tweet['text'][:100]}...")
        print(f"Likes: {tweet['metrics']['likes']}, Retweets: {tweet['metrics']['retweets']}")
    
    print("\n\nFetching AAPL tweets...")
    aapl_tweets = get_market_tweets(symbol="AAPL", count=5)
    for tweet in aapl_tweets:
        print(f"\n@{tweet['author']['username']}: {tweet['text'][:100]}...")
