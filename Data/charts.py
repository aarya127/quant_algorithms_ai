"""
Stock Charts Data Module
Uses yfinance to fetch historical price data for charting
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def get_chart_data(symbol: str, period: str = "1y", interval: str = "1d") -> dict:
    """
    Get historical price data for charting
    
    Args:
        symbol: Stock ticker symbol (e.g., "AAPL")
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    
    Returns:
        Dictionary with chart data and metadata
    """
    
    try:
        print(f"📊 Fetching {period} chart data for {symbol} with {interval} interval...")
        
        # Create ticker object
        ticker = yf.Ticker(symbol)
        
        # Fetch historical data
        hist = ticker.history(period=period, interval=interval)
        
        if hist.empty:
            print(f"❌ No chart data found for {symbol}")
            return {
                "success": False,
                "error": "No data available"
            }
        
        # Prepare data for Chart.js
        dates = hist.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
        
        chart_data = {
            "success": True,
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "dates": dates,
            "open": hist['Open'].round(2).tolist(),
            "high": hist['High'].round(2).tolist(),
            "low": hist['Low'].round(2).tolist(),
            "close": hist['Close'].round(2).tolist(),
            "volume": hist['Volume'].astype(int).tolist(),
            "data_points": len(dates)
        }
        
        print(f"✓ Retrieved {len(dates)} data points for {symbol}")
        
        return chart_data
        
    except Exception as e:
        print(f"❌ Error fetching chart data for {symbol}: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def get_multiple_timeframes(symbol: str) -> dict:
    """
    Get chart data for multiple timeframes at once
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        Dictionary with data for 1D, 1W, 1M, 3M, 1Y, 5Y
    """
    
    timeframes = {
        "1D": {"period": "1d", "interval": "5m"},
        "1W": {"period": "5d", "interval": "30m"},
        "1M": {"period": "1mo", "interval": "1d"},
        "3M": {"period": "3mo", "interval": "1d"},
        "1Y": {"period": "1y", "interval": "1d"},
        "5Y": {"period": "5y", "interval": "1wk"}
    }
    
    result = {}
    
    for name, params in timeframes.items():
        data = get_chart_data(symbol, params["period"], params["interval"])
        result[name] = data
    
    return result


def get_comparison_data(symbols: list, period: str = "1y", interval: str = "1d") -> dict:
    """
    Get chart data for multiple symbols for comparison
    
    Args:
        symbols: List of stock ticker symbols
        period: Time period
        interval: Data interval
    
    Returns:
        Dictionary with normalized comparison data
    """
    
    try:
        print(f"📊 Fetching comparison data for {', '.join(symbols)}...")
        
        comparison_data = {
            "success": True,
            "symbols": symbols,
            "period": period,
            "interval": interval,
            "datasets": []
        }
        
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            if not hist.empty:
                # Normalize to percentage change from start
                start_price = hist['Close'].iloc[0]
                normalized = ((hist['Close'] - start_price) / start_price * 100).round(2)
                
                comparison_data["datasets"].append({
                    "symbol": symbol,
                    "dates": hist.index.strftime('%Y-%m-%d').tolist(),
                    "values": normalized.tolist(),
                    "start_price": round(start_price, 2),
                    "end_price": round(hist['Close'].iloc[-1], 2),
                    "change_percent": round(((hist['Close'].iloc[-1] - start_price) / start_price * 100), 2)
                })
        
        print(f"✓ Retrieved comparison data for {len(comparison_data['datasets'])} symbols")
        
        return comparison_data
        
    except Exception as e:
        print(f"❌ Error fetching comparison data: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    # Test the module
    import sys
    
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
        period = sys.argv[2] if len(sys.argv) > 2 else "1y"
        
        print(f"\n{'='*50}")
        print(f"Testing Chart Data for {symbol}")
        print(f"{'='*50}\n")
        
        data = get_chart_data(symbol, period)
        
        if data["success"]:
            print(f"\n✓ Successfully retrieved {data['data_points']} data points")
            print(f"  Date range: {data['dates'][0]} to {data['dates'][-1]}")
            print(f"  Price range: ${min(data['low']):.2f} - ${max(data['high']):.2f}")
        else:
            print(f"\n❌ Failed: {data['error']}")
    else:
        print("Usage: python charts.py SYMBOL [PERIOD]")
        print("Example: python charts.py AAPL 1y")
