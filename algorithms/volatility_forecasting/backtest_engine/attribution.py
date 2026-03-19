from collections import defaultdict
from typing import Dict, List

from .schemas import AttributionResult, DailyBacktestMetrics


class PnLAttributionEngine:
    """Break strategy PnL into model, hedge, greek, and cost components."""

    def attribute(self, daily: List[DailyBacktestMetrics]) -> AttributionResult:
        model_edge_capture = 0.0
        delta_hedge_contribution = 0.0
        vega_exposure = 0.0
        theta_carry = 0.0
        execution_cost = 0.0
        spread_crossing = 0.0

        regime_performance: Dict[str, float] = defaultdict(float)

        for d in daily:
            # "Model edge capture" is residual MTM after greek+hedge decomposition.
            explained = d.delta_pnl + d.gamma_pnl + d.vega_pnl + d.theta_pnl + d.hedge_pnl
            model_edge_capture += d.mtm_pnl - explained

            delta_hedge_contribution += d.hedge_pnl - d.hedge_costs
            vega_exposure += d.vega_pnl
            theta_carry += d.theta_pnl
            execution_cost += d.transaction_costs + d.hedge_costs

            # Approximate spread crossing as a fraction of transaction costs.
            spread_crossing += d.transaction_costs * 0.50

            regime_performance[d.regime] += d.total_pnl

        return AttributionResult(
            model_edge_capture=model_edge_capture,
            delta_hedge_contribution=delta_hedge_contribution,
            vega_exposure=vega_exposure,
            theta_carry=theta_carry,
            execution_cost=execution_cost,
            spread_crossing=spread_crossing,
            regime_performance=dict(regime_performance),
        )
