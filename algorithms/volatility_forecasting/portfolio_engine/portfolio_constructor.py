from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

from .schemas import TargetPosition


class HedgePolicy(str, Enum):
    CONTINUOUS = "continuous"
    DISCRETE_DAILY = "discrete_daily"
    THRESHOLD = "threshold"


@dataclass
class PortfolioBook:
    positions: List[TargetPosition]
    net_delta: float
    net_gamma: float
    net_vega: float
    net_theta: float
    gross_premium: float
    total_stress_loss: float
    delta_hedge_shares: int


class PortfolioConstructor:
    """Net overlapping trades and apply portfolio-level hedge policy."""

    def __init__(
        self,
        hedge_policy: HedgePolicy = HedgePolicy.THRESHOLD,
        delta_threshold: float = 25.0,
        allow_net_vega_accumulation: bool = True,
    ):
        self.hedge_policy = hedge_policy
        self.delta_threshold = abs(delta_threshold)
        self.allow_net_vega_accumulation = allow_net_vega_accumulation

    def construct(self, positions: List[TargetPosition]) -> PortfolioBook:
        netted = self._net_overlapping_positions(positions)

        net_delta = sum(p.delta for p in netted)
        net_gamma = sum(p.gamma for p in netted)
        net_vega = sum(p.vega for p in netted)
        net_theta = sum(p.theta for p in netted)
        gross_premium = sum(abs(p.premium_notional) for p in netted)
        total_stress = sum(p.stress_loss for p in netted)

        if not self.allow_net_vega_accumulation:
            net_vega = 0.0

        delta_hedge_shares = self._delta_hedge_shares(net_delta)

        return PortfolioBook(
            positions=netted,
            net_delta=net_delta,
            net_gamma=net_gamma,
            net_vega=net_vega,
            net_theta=net_theta,
            gross_premium=gross_premium,
            total_stress_loss=total_stress,
            delta_hedge_shares=delta_hedge_shares,
        )

    def _net_overlapping_positions(self, positions: List[TargetPosition]) -> List[TargetPosition]:
        merged: Dict[Tuple[str, str, float, float], TargetPosition] = {}
        for p in positions:
            key = (p.ticker, p.option_type, round(p.strike, 4), round(p.maturity_days, 2))
            signed_contracts = p.contracts if p.side == "buy" else -p.contracts

            if key not in merged:
                merged[key] = p
                continue

            base = merged[key]
            base_signed = base.contracts if base.side == "buy" else -base.contracts
            net_signed = base_signed + signed_contracts

            if net_signed == 0:
                del merged[key]
                continue

            scale = abs(net_signed) / max(abs(base_signed), 1)
            base.side = "buy" if net_signed > 0 else "sell"
            base.contracts = abs(net_signed)
            base.premium_notional *= scale
            base.delta *= scale
            base.gamma *= scale
            base.vega *= scale
            base.theta *= scale
            base.stress_loss *= scale

        return list(merged.values())

    def _delta_hedge_shares(self, net_delta: float) -> int:
        if self.hedge_policy == HedgePolicy.CONTINUOUS:
            return int(round(-net_delta))

        if self.hedge_policy == HedgePolicy.DISCRETE_DAILY:
            return int(round(-net_delta))

        if abs(net_delta) >= self.delta_threshold:
            return int(round(-net_delta))
        return 0
