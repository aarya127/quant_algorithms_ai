"""
Alpaca News WebSocket Integration
Real-time stock market news stream from Alpaca
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime
from typing import List, Dict, Callable
import threading
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Alpaca API credentials
ALPACA_API_KEY = "0bbeda09-dbf4-4834-9ec1-f1fe99f1ed73"
ALPACA_SECRET_KEY = "0xdCjEkgJ_g8RMrlnGgoO8bm5PYUpUO6"

# WebSocket URLs
ALPACA_NEWS_URL = "wss://stream.data.alpaca.markets/v1beta1/news"
ALPACA_NEWS_SANDBOX_URL = "wss://stream.data.sandbox.alpaca.markets/v1beta1/news"

# Store recent news in memory (thread-safe deque)
recent_news = deque(maxlen=100)
news_lock = threading.Lock()

class AlpacaNewsStream:
    """WebSocket client for Alpaca real-time news"""
    
    def __init__(self, use_sandbox=False):
        self.url = ALPACA_NEWS_SANDBOX_URL if use_sandbox else ALPACA_NEWS_URL
        self.api_key = ALPACA_API_KEY
        self.secret_key = ALPACA_SECRET_KEY
        self.websocket = None
        self.running = False
        self.subscribed_symbols = set()
        
    async def connect(self):
        """Establish WebSocket connection"""
        try:
            # Add authentication headers
            additional_headers = {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret_key
            }
            
            self.websocket = await websockets.connect(
                self.url,
                additional_headers=additional_headers
            )
            logger.info(f"Connected to Alpaca News WebSocket: {self.url}")
            
            # Wait for connection and authentication messages
            async for message in self.websocket:
                data = json.loads(message)
                if isinstance(data, list):
                    for msg in data:
                        if msg.get('T') == 'success':
                            logger.info(f"Alpaca: {msg.get('msg')}")
                            if msg.get('msg') == 'authenticated':
                                return True
                        elif msg.get('T') == 'error':
                            logger.error(f"Alpaca error: {msg.get('msg')}")
                            return False
                            
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca News: {str(e)}")
            return False
    
    async def subscribe(self, symbols: List[str] = None):
        """
        Subscribe to news for specific symbols or all news
        
        Args:
            symbols: List of symbols to subscribe to, or None for all news (*)
        """
        if not self.websocket:
            logger.error("Not connected to Alpaca")
            return False
        
        try:
            if symbols:
                self.subscribed_symbols.update(symbols)
                subscribe_msg = {"action": "subscribe", "news": symbols}
            else:
                # Subscribe to all news
                subscribe_msg = {"action": "subscribe", "news": ["*"]}
            
            await self.websocket.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to news: {symbols if symbols else 'ALL'}")
            
            # Wait for subscription confirmation
            message = await self.websocket.recv()
            data = json.loads(message)
            if isinstance(data, list):
                for msg in data:
                    if msg.get('T') == 'subscription':
                        logger.info(f"Subscription confirmed: {msg.get('news')}")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to subscribe: {str(e)}")
            return False
    
    async def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from news for specific symbols"""
        if not self.websocket:
            return False
        
        try:
            unsubscribe_msg = {"action": "unsubscribe", "news": symbols}
            await self.websocket.send(json.dumps(unsubscribe_msg))
            self.subscribed_symbols.difference_update(symbols)
            logger.info(f"Unsubscribed from: {symbols}")
            return True
        except Exception as e:
            logger.error(f"Failed to unsubscribe: {str(e)}")
            return False
    
    async def listen(self, callback: Callable = None):
        """
        Listen for incoming news messages
        
        Args:
            callback: Optional callback function to handle news messages
        """
        self.running = True
        
        try:
            async for message in self.websocket:
                if not self.running:
                    break
                
                data = json.loads(message)
                
                if isinstance(data, list):
                    for msg in data:
                        if msg.get('T') == 'n':  # News message
                            news_item = self._format_news(msg)
                            
                            # Store in memory
                            with news_lock:
                                recent_news.append(news_item)
                            
                            # Call callback if provided
                            if callback:
                                callback(news_item)
                            
                            logger.info(f"News: {news_item['headline'][:60]}...")
                            
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error in listen loop: {str(e)}")
        finally:
            self.running = False
    
    def _format_news(self, msg: Dict) -> Dict:
        """Format Alpaca news message to standard format"""
        return {
            'id': msg.get('id'),
            'headline': msg.get('headline'),
            'summary': msg.get('summary'),
            'author': msg.get('author'),
            'created_at': msg.get('created_at'),
            'updated_at': msg.get('updated_at'),
            'url': msg.get('url'),
            'content': msg.get('content'),
            'symbols': msg.get('symbols', []),
            'source': f"Alpaca - {msg.get('source', 'Unknown')}",
            'type': 'realtime'
        }
    
    async def close(self):
        """Close the WebSocket connection"""
        self.running = False
        if self.websocket:
            await self.websocket.close()
            logger.info("Closed Alpaca News WebSocket")

# Global news stream instance
_news_stream = None
_stream_thread = None

def start_news_stream(symbols: List[str] = None, use_sandbox=False):
    """
    Start the Alpaca news stream in a background thread
    
    Args:
        symbols: List of symbols to subscribe to, or None for all news
        use_sandbox: Use sandbox URL instead of production
    """
    global _news_stream, _stream_thread
    
    if _news_stream and _news_stream.running:
        logger.info("News stream already running")
        return
    
    def run_stream():
        async def stream_task():
            global _news_stream
            _news_stream = AlpacaNewsStream(use_sandbox=use_sandbox)
            
            if await _news_stream.connect():
                if await _news_stream.subscribe(symbols):
                    await _news_stream.listen()
        
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(stream_task())
        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
        finally:
            loop.close()
    
    _stream_thread = threading.Thread(target=run_stream, daemon=True)
    _stream_thread.start()
    logger.info("Started Alpaca news stream in background")

def stop_news_stream():
    """Stop the background news stream"""
    global _news_stream
    
    if _news_stream and _news_stream.running:
        _news_stream.running = False
        logger.info("Stopped Alpaca news stream")
    
    _news_stream = None

def get_recent_news(count: int = 20, symbol: str = None) -> List[Dict]:
    """
    Get recent news from memory cache
    
    Args:
        count: Number of news items to return
        symbol: Optional symbol to filter by
    
    Returns:
        List of recent news items
    """
    with news_lock:
        news_list = list(recent_news)
    
    # Filter by symbol if provided
    if symbol:
        news_list = [n for n in news_list if symbol.upper() in n.get('symbols', [])]
    
    # Sort by created_at (newest first)
    news_list.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    
    return news_list[:count]

async def fetch_news_snapshot(symbols: List[str] = None, timeout: int = 5) -> List[Dict]:
    """
    Fetch a quick snapshot of news (connect, get messages, disconnect)
    
    Args:
        symbols: Symbols to get news for
        timeout: How long to listen for messages (seconds)
    
    Returns:
        List of news items received during timeout period
    """
    news_items = []
    
    def collect_news(item):
        news_items.append(item)
    
    stream = AlpacaNewsStream()
    
    try:
        if await stream.connect():
            if await stream.subscribe(symbols):
                # Listen for limited time
                listen_task = asyncio.create_task(stream.listen(callback=collect_news))
                await asyncio.sleep(timeout)
                await stream.close()
                
    except Exception as e:
        logger.error(f"Error fetching news snapshot: {str(e)}")
    
    return news_items

if __name__ == "__main__":
    # Test the news stream
    print("Starting Alpaca news stream...")
    start_news_stream(symbols=["AAPL", "TSLA", "NVDA"])
    
    # Let it run for 30 seconds
    import time
    time.sleep(30)
    
    # Get recent news
    print("\n\nRecent news:")
    news = get_recent_news(count=5)
    for item in news:
        print(f"\n{item['headline']}")
        print(f"Symbols: {', '.join(item['symbols'])}")
        print(f"Source: {item['source']}")
    
    stop_news_stream()
