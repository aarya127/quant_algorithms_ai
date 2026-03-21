"""
MACD Crossover Trading Strategy

Momentum-based signal generation using Moving Average Convergence Divergence.

Strategy Logic:
  BUY  signal: MACD line crosses above signal line (bullish crossover)
  SELL signal: MACD line crosses below signal line (bearish crossover)
  Position sizing: Fixed-fractional based on ATR volatility
  Stop-loss: N x ATR from entry price
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MACDParams:
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    atr_period: int = 14
    risk_per_trade: float = 0.02      # 2% of capital per trade
    stop_loss_atr_mult: float = 2.0   # stop is N * ATR from entry


@dataclass
class TradeSignal:
    timestamp: pd.Timestamp
    direction: str        # 'long' | 'short' | 'close'
    price: float
    stop_loss: float
    position_size: float  # shares / units
    macd_value: float
    signal_value: float
    histogram: float
    reason: str


class MACDStrategy:
    """
    MACD crossover strategy with ATR-based position sizing and stop-loss.

    Parameters
    ----------
    params : MACDParams
        Strategy configuration.
    capital : float
        Starting capital used for fractional sizing.
    """

    def __init__(self, params: MACDParams = None, capital: float = 100_000.0):
        self.params = params or MACDParams()
        self.capital = capital

    # ------------------------------------------------------------------
    # Indicator helpers
    # ------------------------------------------------------------------

    def _ema(self, series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    def compute_macd(self, close: pd.Series) -> pd.DataFrame:
        fast = self._ema(close, self.params.fast_period)
        slow = self._ema(close, self.params.slow_period)
        macd_line = fast - slow
        signal_line = self._ema(macd_line, self.params.signal_period)
        histogram = macd_line - signal_line
        return pd.DataFrame(
            {'macd': macd_line, 'signal': signal_line, 'histogram': histogram},
            index=close.index,
        )

    def compute_atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(span=self.params.atr_period, adjust=False).mean()

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        """
        Generate MACD crossover signals from OHLCV data.

        data must have columns: open, high, low, close, volume (case-insensitive).
        """
        data.columns = [c.lower() for c in data.columns]
        indicators = self.compute_macd(data['close'])
        atr = self.compute_atr(data['high'], data['low'], data['close'])

        signals: List[TradeSignal] = []
        in_position: Optional[str] = None
        warmup = max(self.params.slow_period + self.params.signal_period,
                     self.params.atr_period) + 5

        macd = indicators['macd']
        sig  = indicators['signal']
        hist = indicators['histogram']

        for i in range(1, len(data)):
            if i < warmup:
                continue

            prev_hist = hist.iloc[i - 1]
            curr_hist = hist.iloc[i]
            curr_price = data['close'].iloc[i]
            curr_atr   = atr.iloc[i]
            ts = data.index[i]

            # --- BULL CROSSOVER ---
            if prev_hist <= 0 < curr_hist and in_position != 'long':
                if in_position == 'short':
                    signals.append(TradeSignal(
                        timestamp=ts, direction='close', price=curr_price,
                        stop_loss=0, position_size=0,
                        macd_value=macd.iloc[i], signal_value=sig.iloc[i],
                        histogram=curr_hist, reason='Close short on bullish crossover'))
                    in_position = None

                stop = curr_price - self.params.stop_loss_atr_mult * curr_atr
                risk_per_share = curr_price - stop
                size = (self.capital * self.params.risk_per_trade) / max(risk_per_share, 0.01)
                signals.append(TradeSignal(
                    timestamp=ts, direction='long', price=curr_price,
                    stop_loss=round(stop, 2), position_size=round(size, 2),
                    macd_value=round(macd.iloc[i], 4), signal_value=round(sig.iloc[i], 4),
                    histogram=round(curr_hist, 4), reason='Bullish MACD crossover'))
                in_position = 'long'

            # --- BEAR CROSSOVER ---
            elif prev_hist >= 0 > curr_hist and in_position != 'short':
                if in_position == 'long':
                    signals.append(TradeSignal(
                        timestamp=ts, direction='close', price=curr_price,
                        stop_loss=0, position_size=0,
                        macd_value=macd.iloc[i], signal_value=sig.iloc[i],
                        histogram=curr_hist, reason='Close long on bearish crossover'))
                    in_position = None

                stop = curr_price + self.params.stop_loss_atr_mult * curr_atr
                risk_per_share = stop - curr_price
                size = (self.capital * self.params.risk_per_trade) / max(risk_per_share, 0.01)
                signals.append(TradeSignal(
                    timestamp=ts, direction='short', price=curr_price,
                    stop_loss=round(stop, 2), position_size=round(size, 2),
                    macd_value=round(macd.iloc[i], 4), signal_value=round(sig.iloc[i], 4),
                    histogram=round(curr_hist, 4), reason='Bearish MACD crossover'))
                in_position = 'short'

        return signals

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------

    def run(self, ticker: str, period: str = '1y') -> None:
        """Fetch data and print the most recent signals."""
        import yfinance as yf
        raw = yf.download(ticker, period=period, progress=False)
        signals = self.generate_signals(raw)
        print(f"\n{ticker}  MACD Strategy  ({len(signals)} signals)")
        print("-" * 70)
        for s in signals[-20:]:
            print(f"  {s.timestamp.date()}  {s.direction.upper():6s}  "
                  f"@{s.price:.2f}  SL={s.stop_loss:.2f}  "
                  f"Size={s.position_size:.0f}  | {s.reason}")


if __name__ == '__main__':
    strategy = MACDStrategy()
    strategy.run('NVDA')
