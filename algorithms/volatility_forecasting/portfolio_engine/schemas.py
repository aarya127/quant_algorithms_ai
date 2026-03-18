from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class SizingBasis(str, Enum):
    """Primary risk unit used for sizing."""

    VEGA = "vega"
    GAMMA = "gamma"
    THETA = "theta"
    PREMIUM = "premium"
    STRESS_LOSS = "stress_loss"


@dataclass
class OptionPositionCandidate:
    """Single candidate position produced from a signal/intent."""

    intent_id: str
    ticker: str
    option_type: str  # call/put
    side: str  # buy/sell
    strike: float
    maturity_days: float
    confidence: float
    expected_net_edge_bps: float

    bid: float
    ask: float
    mid: float
    volume: int
    open_interest: int

    # Per-contract greek risk units (research estimates are acceptable).
    delta: float
    gamma: float
    vega: float
    theta: float

    # 1-day adverse scenario loss per contract (USD, positive magnitude).
    stress_loss_per_contract: float

    quote_timestamp: Optional[datetime] = None


@dataclass
class TargetPosition:
    """Target sized position before execution."""

    intent_id: str
    ticker: str
    option_type: str
    side: str
    strike: float
    maturity_days: float
    contracts: int

    premium_notional: float
    expected_net_edge_bps: float
    confidence: float

    delta: float
    gamma: float
    vega: float
    theta: float

    stress_loss: float


@dataclass
class PortfolioConstraints:
    """Sizing and concentration controls."""

    capital_usd: float
    max_capital_alloc_pct: float = 0.35

    # Book-level greek and stress controls.
    max_abs_vega: float = 8000.0
    max_abs_gamma: float = 80.0
    max_abs_theta: float = 5000.0
    max_total_stress_loss_usd: float = 25000.0

    # Concentration controls.
    max_underlying_alloc_pct: float = 0.20
    max_maturity_alloc_pct: float = 0.15
    max_strike_alloc_pct: float = 0.10

    # Per-intent hard cap.
    max_contracts_per_intent: int = 50


@dataclass
class ExecutionAssumptions:
    """Execution model assumptions for options fills."""

    fill_model: str = "mid_minus_spread_fraction"  # mid, mid_minus_spread_fraction, full_bid_ask, probabilistic
    spread_capture_fraction: float = 0.25
    slippage_bps: float = 10.0
    commission_per_contract: float = 0.65

    min_open_interest: int = 100
    min_volume: int = 20
    max_participation_rate: float = 0.10
    max_quote_age_seconds: int = 120


@dataclass
class FillResult:
    """Execution simulation output for one target position."""

    intent_id: str
    accepted: bool
    reason: str
    filled_contracts: int
    fill_price: float
    estimated_fees: float
    total_cash_impact: float
