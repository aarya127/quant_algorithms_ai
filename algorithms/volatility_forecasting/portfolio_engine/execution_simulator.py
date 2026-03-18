import random
from datetime import datetime
from typing import Dict, List

from .schemas import ExecutionAssumptions, FillResult, OptionPositionCandidate, TargetPosition


class OptionFillSimulator:
    """Simulate option fills under spread, liquidity, and fee assumptions."""

    def __init__(self, assumptions: ExecutionAssumptions):
        self.assumptions = assumptions

    def simulate(
        self,
        targets: List[TargetPosition],
        candidate_lookup: Dict[str, OptionPositionCandidate],
    ) -> List[FillResult]:
        results: List[FillResult] = []
        for t in targets:
            c = candidate_lookup.get(t.intent_id)
            if c is None:
                results.append(FillResult(t.intent_id, False, "missing_candidate", 0, 0.0, 0.0, 0.0))
                continue

            valid, reason = self._passes_filters(c, t.contracts)
            if not valid:
                results.append(FillResult(t.intent_id, False, reason, 0, 0.0, 0.0, 0.0))
                continue

            fill_price = self._fill_price(c, t.side)
            filled_contracts = t.contracts
            fees = filled_contracts * self.assumptions.commission_per_contract
            gross = fill_price * 100.0 * filled_contracts
            cash_impact = gross + fees if t.side == "buy" else -gross + fees

            results.append(
                FillResult(
                    intent_id=t.intent_id,
                    accepted=True,
                    reason="filled",
                    filled_contracts=filled_contracts,
                    fill_price=fill_price,
                    estimated_fees=fees,
                    total_cash_impact=cash_impact,
                )
            )
        return results

    def _passes_filters(self, c: OptionPositionCandidate, contracts: int):
        if c.open_interest < self.assumptions.min_open_interest:
            return False, "min_open_interest_fail"
        if c.volume < self.assumptions.min_volume:
            return False, "min_volume_fail"

        max_contracts = int(max(c.volume, 1) * self.assumptions.max_participation_rate)
        if contracts > max_contracts:
            return False, "max_participation_fail"

        if c.quote_timestamp is not None:
            age = (datetime.utcnow() - c.quote_timestamp).total_seconds()
            if age > self.assumptions.max_quote_age_seconds:
                return False, "stale_quote"

        if c.ask <= 0 or c.bid < 0 or c.ask < c.bid:
            return False, "bad_quote"

        return True, "ok"

    def _fill_price(self, c: OptionPositionCandidate, side: str) -> float:
        spread = max(c.ask - c.bid, 0.0)
        model = self.assumptions.fill_model

        if model == "mid":
            px = c.mid
        elif model == "full_bid_ask":
            px = c.ask if side == "buy" else c.bid
        elif model == "probabilistic":
            # 50% mid, 35% mid +/- 25% spread, 15% touch.
            u = random.random()
            if u < 0.50:
                px = c.mid
            elif u < 0.85:
                frac = 0.25
                px = c.mid + frac * spread if side == "buy" else c.mid - frac * spread
            else:
                px = c.ask if side == "buy" else c.bid
        else:
            # mid_minus_spread_fraction (default)
            frac = max(min(self.assumptions.spread_capture_fraction, 1.0), 0.0)
            px = c.mid + frac * spread if side == "buy" else c.mid - frac * spread

        slip = px * (self.assumptions.slippage_bps / 10000.0)
        px = px + slip if side == "buy" else max(px - slip, 0.0)
        return max(px, 0.0)
