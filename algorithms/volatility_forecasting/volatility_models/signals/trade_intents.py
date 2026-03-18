"""
Trade Intent Layer - Convert model signals into explicit trade intents.

This module bridges research signals and execution-ready intent definitions.
Each intent explicitly defines:
- instrument universe
- trade direction and structure
- entry condition
- exit condition
- holding horizon
- invalidation condition
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from signal_generator import SignalType, TradingSignal


@dataclass
class TradeIntent:
    """Execution-facing intent derived from a validated research signal."""

    created_at: datetime
    intent_id: str
    ticker: str
    source_signal_type: str
    instrument_universe: str
    trade_direction: str
    structure: str
    action_summary: str
    entry_condition: str
    exit_condition: str
    holding_horizon_days: int
    invalidation_condition: str
    expected_net_edge_bps: float
    confidence: float
    strike: float
    maturity_days: float
    hedge_policy: str


class TradeIntentBuilder:
    """Convert filtered signals into concrete trade intents."""

    def __init__(self, min_edge_bps: float = 50.0):
        self.min_edge_bps = min_edge_bps

    def build_from_single_leg_signals(
        self,
        signals: List[TradingSignal],
        ticker: str,
        spot: Optional[float] = None,
    ) -> List[TradeIntent]:
        intents: List[TradeIntent] = []
        for idx, signal in enumerate(signals, start=1):
            intent = self._single_leg_signal_to_intent(signal, ticker=ticker, ordinal=idx, spot=spot)
            if intent is not None:
                intents.append(intent)
        return intents

    def to_records(self, intents: List[TradeIntent]) -> List[Dict]:
        records: List[Dict] = []
        for intent in intents:
            records.append(
                {
                    "created_at": intent.created_at,
                    "intent_id": intent.intent_id,
                    "ticker": intent.ticker,
                    "source_signal_type": intent.source_signal_type,
                    "instrument_universe": intent.instrument_universe,
                    "trade_direction": intent.trade_direction,
                    "structure": intent.structure,
                    "action_summary": intent.action_summary,
                    "entry_condition": intent.entry_condition,
                    "exit_condition": intent.exit_condition,
                    "holding_horizon_days": intent.holding_horizon_days,
                    "invalidation_condition": intent.invalidation_condition,
                    "expected_net_edge_bps": intent.expected_net_edge_bps,
                    "confidence": intent.confidence,
                    "strike": intent.strike,
                    "maturity_days": intent.maturity_days,
                    "hedge_policy": intent.hedge_policy,
                }
            )
        return records

    def _single_leg_signal_to_intent(
        self,
        signal: TradingSignal,
        ticker: str,
        ordinal: int,
        spot: Optional[float],
    ) -> Optional[TradeIntent]:
        if signal.signal_type == SignalType.NO_TRADE:
            return None

        strike = float(signal.strikes[0])
        maturity_days = max(signal.maturities[0] * 365.0, 1.0)
        horizon_days = self._default_holding_horizon_days(maturity_days)

        direction, structure, action = self._map_signal_structure(signal.signal_type, strike)

        moneyness_text = ""
        if spot is not None and spot > 0:
            m = strike / spot
            moneyness_text = f" | moneyness={m:.3f}"

        entry_condition = (
            f"Enter only if net_edge_bps >= {self.min_edge_bps:.0f}, liquidity/spread checks remain true, "
            f"and modeled mispricing persists (model_iv - market_iv keeps sign){moneyness_text}."
        )

        exit_condition = (
            "Exit when edge closes (|model_iv - market_iv| < 0.5 vol points), "
            "or target capture reaches ~70% of expected edge, "
            f"or holding horizon reaches {horizon_days} calendar days, whichever comes first."
        )

        invalidation_condition = (
            "Invalidate immediately on regime flip or jump alert from diagnostics, "
            "net_edge_bps falls below threshold, or market quality degrades "
            "(liquidity/spread/strike availability fails)."
        )

        hedge_policy = self._hedge_policy_for_signal(signal.signal_type)

        return TradeIntent(
            created_at=signal.timestamp,
            intent_id=f"{ticker.upper()}-{signal.signal_type.value}-{ordinal:03d}",
            ticker=ticker.upper(),
            source_signal_type=signal.signal_type.value,
            instrument_universe=f"{ticker.upper()} listed vanilla options (single-stock/ETF options)",
            trade_direction=direction,
            structure=structure,
            action_summary=action,
            entry_condition=entry_condition,
            exit_condition=exit_condition,
            holding_horizon_days=horizon_days,
            invalidation_condition=invalidation_condition,
            expected_net_edge_bps=float(signal.net_edge_bps),
            confidence=float(signal.confidence),
            strike=strike,
            maturity_days=float(maturity_days),
            hedge_policy=hedge_policy,
        )

    @staticmethod
    def _default_holding_horizon_days(maturity_days: float) -> int:
        # Keep horizon materially shorter than option expiry to avoid terminal effects.
        horizon = int(max(3, min(30, maturity_days * 0.35)))
        return horizon

    @staticmethod
    def _hedge_policy_for_signal(signal_type: SignalType) -> str:
        if signal_type in {SignalType.LONG_CALL, SignalType.LONG_PUT, SignalType.SHORT_CALL, SignalType.SHORT_PUT}:
            return (
                "Delta-aware: rebalance delta hedge only when |portfolio_delta| breaches desk threshold "
                "(e.g., 0.10 notional delta) to control transaction churn."
            )
        if signal_type in {SignalType.LONG_STRADDLE, SignalType.SHORT_STRADDLE}:
            return "Delta-hedged options structure: monitor and rebalance delta daily or on threshold breach."
        return "No explicit hedge policy encoded."

    @staticmethod
    def _map_signal_structure(signal_type: SignalType, strike: float):
        if signal_type == SignalType.LONG_CALL:
            return (
                "Long volatility, bullish convexity",
                "Single-leg long call",
                f"BUY 1 call @ K={strike:.2f}",
            )
        if signal_type == SignalType.SHORT_CALL:
            return (
                "Short volatility, bearish/upside-capped",
                "Single-leg short call",
                f"SELL 1 call @ K={strike:.2f}",
            )
        if signal_type == SignalType.LONG_PUT:
            return (
                "Long volatility, bearish convexity",
                "Single-leg long put",
                f"BUY 1 put @ K={strike:.2f}",
            )
        if signal_type == SignalType.SHORT_PUT:
            return (
                "Short volatility, bullish/downside-risk",
                "Single-leg short put",
                f"SELL 1 put @ K={strike:.2f}",
            )
        if signal_type == SignalType.LONG_STRADDLE:
            return (
                "Long volatility wings/ATM expansion",
                "Delta-hedged long straddle",
                f"BUY 1 call + BUY 1 put @ K={strike:.2f}",
            )
        if signal_type == SignalType.SHORT_STRADDLE:
            return (
                "Short volatility mean-reversion",
                "Delta-hedged short straddle",
                f"SELL 1 call + SELL 1 put @ K={strike:.2f}",
            )
        return ("No trade", "No structure", "NO ACTION")
