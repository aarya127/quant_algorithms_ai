from collections import defaultdict
from typing import Dict, List, Tuple

from .schemas import OptionPositionCandidate, PortfolioConstraints, SizingBasis, TargetPosition


class GreekAwarePositionSizer:
    """Convert signal candidates into target positions under risk-unit constraints."""

    def __init__(self, constraints: PortfolioConstraints, basis: SizingBasis = SizingBasis.VEGA):
        self.constraints = constraints
        self.basis = basis

    def size(self, candidates: List[OptionPositionCandidate]) -> List[TargetPosition]:
        if not candidates:
            return []

        ranked = sorted(candidates, key=self._score_candidate, reverse=True)
        max_alloc_cash = self.constraints.capital_usd * self.constraints.max_capital_alloc_pct

        positions: List[TargetPosition] = []
        used_cash = 0.0

        cash_by_underlying: Dict[str, float] = defaultdict(float)
        cash_by_maturity: Dict[Tuple[str, int], float] = defaultdict(float)
        cash_by_strike: Dict[Tuple[str, int], float] = defaultdict(float)

        total_vega = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_stress = 0.0

        for c in ranked:
            per_contract_cash = max(c.mid, 0.01) * 100.0
            unit_risk = self._unit_risk(c)

            if per_contract_cash <= 0 or unit_risk <= 0:
                continue

            risk_cap_contracts = int(self._risk_capacity_remaining(total_vega, total_gamma, total_theta, total_stress) / unit_risk)
            cash_cap_contracts = int((max_alloc_cash - used_cash) / per_contract_cash)
            per_intent_cap = self.constraints.max_contracts_per_intent

            k_maturity = (c.ticker, int(round(c.maturity_days)))
            k_strike = (c.ticker, int(round(c.strike)))
            undy_remaining = self.constraints.capital_usd * self.constraints.max_underlying_alloc_pct - cash_by_underlying[c.ticker]
            maturity_remaining = self.constraints.capital_usd * self.constraints.max_maturity_alloc_pct - cash_by_maturity[k_maturity]
            strike_remaining = self.constraints.capital_usd * self.constraints.max_strike_alloc_pct - cash_by_strike[k_strike]

            conc_cap_contracts = int(max(0.0, min(undy_remaining, maturity_remaining, strike_remaining)) / per_contract_cash)
            contracts = max(0, min(risk_cap_contracts, cash_cap_contracts, per_intent_cap, conc_cap_contracts))

            if contracts <= 0:
                continue

            signed = 1.0 if c.side == "buy" else -1.0
            pos = TargetPosition(
                intent_id=c.intent_id,
                ticker=c.ticker,
                option_type=c.option_type,
                side=c.side,
                strike=c.strike,
                maturity_days=c.maturity_days,
                contracts=contracts,
                premium_notional=contracts * per_contract_cash,
                expected_net_edge_bps=c.expected_net_edge_bps,
                confidence=c.confidence,
                delta=signed * c.delta * contracts,
                gamma=signed * c.gamma * contracts,
                vega=signed * c.vega * contracts,
                theta=signed * c.theta * contracts,
                stress_loss=c.stress_loss_per_contract * contracts,
            )

            positions.append(pos)

            used_cash += pos.premium_notional
            cash_by_underlying[c.ticker] += pos.premium_notional
            cash_by_maturity[k_maturity] += pos.premium_notional
            cash_by_strike[k_strike] += pos.premium_notional

            total_vega += pos.vega
            total_gamma += pos.gamma
            total_theta += pos.theta
            total_stress += pos.stress_loss

        return positions

    def _score_candidate(self, c: OptionPositionCandidate) -> float:
        risk = max(self._unit_risk(c), 1e-9)
        edge = max(c.expected_net_edge_bps, 0.0)
        return (edge * max(c.confidence, 0.01)) / risk

    def _unit_risk(self, c: OptionPositionCandidate) -> float:
        if self.basis == SizingBasis.VEGA:
            return max(abs(c.vega), 1e-9)
        if self.basis == SizingBasis.GAMMA:
            return max(abs(c.gamma), 1e-9)
        if self.basis == SizingBasis.THETA:
            return max(abs(c.theta), 1e-9)
        if self.basis == SizingBasis.PREMIUM:
            return max(c.mid * 100.0, 1e-9)
        return max(c.stress_loss_per_contract, 1e-9)

    def _risk_capacity_remaining(self, total_vega: float, total_gamma: float, total_theta: float, total_stress: float) -> float:
        if self.basis == SizingBasis.VEGA:
            return max(self.constraints.max_abs_vega - abs(total_vega), 0.0)
        if self.basis == SizingBasis.GAMMA:
            return max(self.constraints.max_abs_gamma - abs(total_gamma), 0.0)
        if self.basis == SizingBasis.THETA:
            return max(self.constraints.max_abs_theta - abs(total_theta), 0.0)
        if self.basis == SizingBasis.PREMIUM:
            return max(self.constraints.capital_usd * self.constraints.max_capital_alloc_pct, 0.0)
        return max(self.constraints.max_total_stress_loss_usd - total_stress, 0.0)
