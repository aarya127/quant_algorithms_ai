from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Dict, List


class HedgeMode(str, Enum):
    CONTINUOUS = "continuous"
    DISCRETE_DAILY = "discrete_daily"
    THRESHOLD = "threshold"


@dataclass
class BacktestConfig:
    initial_capital_usd: float = 250000.0
    max_holding_days: int = 20

    # Exit assumptions.
    edge_close_vol_points: float = 0.50

    # Hedge assumptions.
    hedge_mode: HedgeMode = HedgeMode.THRESHOLD
    delta_hedge_threshold: float = 25.0
    hedge_slippage_bps: float = 5.0
    hedge_commission_per_share: float = 0.005

    # Execution/transaction assumptions.
    option_commission_per_contract: float = 0.65
    spread_cross_fraction: float = 0.50


@dataclass
class MarketSnapshot:
    date: date
    spot: float
    dspot: float
    iv_change_points: float
    skew_shift_points: float = 0.0
    term_twist_points: float = 0.0
    vol_of_vol_shock_points: float = 0.0
    regime: str = "normal"


@dataclass
class DailyBacktestMetrics:
    date: date

    mtm_pnl: float
    realized_pnl: float
    total_pnl: float

    delta_pnl: float
    gamma_pnl: float
    vega_pnl: float
    theta_pnl: float

    hedge_pnl: float
    hedge_costs: float
    transaction_costs: float

    turnover_usd: float
    capital_used_usd: float
    capital_used_pct: float

    net_delta: float
    net_gamma: float
    net_vega: float
    net_theta: float

    equity: float
    drawdown_pct: float
    regime: str


@dataclass
class AttributionResult:
    model_edge_capture: float
    delta_hedge_contribution: float
    vega_exposure: float
    theta_carry: float
    execution_cost: float
    spread_crossing: float
    regime_performance: Dict[str, float]


@dataclass
class BacktestResult:
    daily: List[DailyBacktestMetrics]
    attribution: AttributionResult
    summary: Dict[str, float]


@dataclass
class ScenarioShock:
    name: str
    spot_shock_pct: float
    skew_shift_points: float
    term_twist_points: float
    vol_of_vol_shock_points: float

    spread_multiplier: float = 1.0
    liquidity_multiplier: float = 1.0
    hedge_lag_factor: float = 1.0
    wrong_way_parameter_jump: bool = False
    calibration_instability: bool = False


@dataclass
class ScenarioResult:
    name: str
    pnl: float
    pnl_breakdown: Dict[str, float]
