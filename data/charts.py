"""
Stock Charts Data Module
Uses yfinance to fetch historical price data for charting
"""

import yfinance as yf
import pandas as pd
import numpy as np
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


def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return data.rolling(window=period).mean()


def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()


def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }


def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: int = 2) -> dict:
    """Calculate Bollinger Bands"""
    sma = calculate_sma(data, period)
    std = data.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    return {
        'middle': sma,
        'upper': upper_band,
        'lower': lower_band
    }


def get_technical_indicators(symbol: str, period: str = "1y", interval: str = "1d") -> dict:
    """
    Get chart data with technical indicators
    
    Args:
        symbol: Stock ticker symbol
        period: Time period
        interval: Data interval
    
    Returns:
        Dictionary with price data and technical indicators
    """
    
    try:
        print(f"📊 Fetching technical indicators for {symbol}...")
        
        # Get base chart data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        
        if hist.empty:
            return {
                "success": False,
                "error": "No data available"
            }
        
        close_prices = hist['Close']
        dates = hist.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
        
        # Calculate indicators
        sma_20 = calculate_sma(close_prices, 20)
        sma_50 = calculate_sma(close_prices, 50)
        sma_200 = calculate_sma(close_prices, 200)
        ema_12 = calculate_ema(close_prices, 12)
        ema_26 = calculate_ema(close_prices, 26)
        rsi = calculate_rsi(close_prices, 14)
        macd_data = calculate_macd(close_prices)
        bollinger = calculate_bollinger_bands(close_prices, 20, 2)
        
        result = {
            "success": True,
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "dates": dates,
            "price": {
                "open": hist['Open'].round(2).tolist(),
                "high": hist['High'].round(2).tolist(),
                "low": hist['Low'].round(2).tolist(),
                "close": hist['Close'].round(2).tolist(),
                "volume": hist['Volume'].astype(int).tolist()
            },
            "indicators": {
                "sma_20": [None if pd.isna(x) else round(x, 2) for x in sma_20],
                "sma_50": [None if pd.isna(x) else round(x, 2) for x in sma_50],
                "sma_200": [None if pd.isna(x) else round(x, 2) for x in sma_200],
                "ema_12": [None if pd.isna(x) else round(x, 2) for x in ema_12],
                "ema_26": [None if pd.isna(x) else round(x, 2) for x in ema_26],
                "rsi": [None if pd.isna(x) else round(x, 2) for x in rsi],
                "macd": {
                    "macd": [None if pd.isna(x) else round(x, 2) for x in macd_data['macd']],
                    "signal": [None if pd.isna(x) else round(x, 2) for x in macd_data['signal']],
                    "histogram": [None if pd.isna(x) else round(x, 2) for x in macd_data['histogram']]
                },
                "bollinger": {
                    "upper": [None if pd.isna(x) else round(x, 2) for x in bollinger['upper']],
                    "middle": [None if pd.isna(x) else round(x, 2) for x in bollinger['middle']],
                    "lower": [None if pd.isna(x) else round(x, 2) for x in bollinger['lower']]
                }
            },
            "data_points": len(dates)
        }
        
        print(f"✓ Calculated technical indicators for {symbol}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error calculating technical indicators for {symbol}: {str(e)}")
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
