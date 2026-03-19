from typing import Dict, List

from algorithms.volatility_forecasting.portfolio_engine.schemas import OptionPositionCandidate, TargetPosition

from .schemas import ScenarioResult, ScenarioShock


class PortfolioStressTester:
    """Stress-test the traded system (positions + execution + hedge assumptions)."""

    def default_scenarios(self) -> List[ScenarioShock]:
        return [
            ScenarioShock(
                name="spot_down_5pct_skew_widen",
                spot_shock_pct=-0.05,
                skew_shift_points=2.5,
                term_twist_points=0.5,
                vol_of_vol_shock_points=1.5,
            ),
            ScenarioShock(
                name="spot_up_4pct_term_twist",
                spot_shock_pct=0.04,
                skew_shift_points=-1.0,
                term_twist_points=2.0,
                vol_of_vol_shock_points=1.0,
            ),
            ScenarioShock(
                name="liquidity_collapse_spread_widen",
                spot_shock_pct=-0.01,
                skew_shift_points=1.0,
                term_twist_points=0.0,
                vol_of_vol_shock_points=1.0,
                spread_multiplier=3.0,
                liquidity_multiplier=0.25,
                hedge_lag_factor=1.5,
            ),
            ScenarioShock(
                name="overnight_gap_wrong_way_jump",
                spot_shock_pct=-0.08,
                skew_shift_points=3.0,
                term_twist_points=1.0,
                vol_of_vol_shock_points=2.5,
                spread_multiplier=2.0,
                hedge_lag_factor=2.0,
                wrong_way_parameter_jump=True,
            ),
            ScenarioShock(
                name="calibration_instability_event",
                spot_shock_pct=-0.03,
                skew_shift_points=2.0,
                term_twist_points=2.0,
                vol_of_vol_shock_points=3.0,
                calibration_instability=True,
                spread_multiplier=2.5,
                liquidity_multiplier=0.40,
            ),
        ]

    def run(
        self,
        positions: List[TargetPosition],
        candidate_lookup: Dict[str, OptionPositionCandidate],
        spot: float,
        scenarios: List[ScenarioShock],
    ) -> List[ScenarioResult]:
        results: List[ScenarioResult] = []
        for s in scenarios:
            pnl_delta = 0.0
            pnl_gamma = 0.0
            pnl_vega = 0.0
            pnl_theta = 0.0
            execution_drag = 0.0
            hedge_lag_penalty = 0.0
            wrong_way_penalty = 0.0
            calibration_penalty = 0.0

            dspot = spot * s.spot_shock_pct

            for p in positions:
                c = candidate_lookup.get(p.intent_id)
                spread = 0.0 if c is None else max(c.ask - c.bid, 0.0)
                liquidity_factor = max(s.liquidity_multiplier, 1e-6)

                pnl_delta += p.delta * dspot
                pnl_gamma += 0.5 * p.gamma * (dspot ** 2)

                # Skew and term shifts proxy through vega bucketed by option type/maturity.
                skew_beta = 1.2 if p.option_type == "put" else 0.8
                term_beta = max(p.maturity_days / 30.0, 0.5)
                vol_shift = (
                    s.vol_of_vol_shock_points
                    + skew_beta * s.skew_shift_points
                    + term_beta * s.term_twist_points
                )
                pnl_vega += p.vega * vol_shift

                # One-day carry under stressed conditions.
                pnl_theta += p.theta / 252.0

                # Wider spreads and lower liquidity worsen execution drag.
                execution_drag += abs(p.contracts) * spread * 100.0 * s.spread_multiplier / liquidity_factor

                # Hedge lag amplifies unhedged delta losses.
                hedge_lag_penalty += abs(p.delta * dspot) * max(s.hedge_lag_factor - 1.0, 0.0)

                if s.wrong_way_parameter_jump:
                    wrong_way_penalty += 0.20 * (abs(p.vega) + abs(p.gamma) * max(abs(dspot), 1.0))

                if s.calibration_instability:
                    calibration_penalty += 0.10 * (abs(p.vega) + abs(p.theta) + abs(p.gamma) * 100.0)

            pnl = (
                pnl_delta + pnl_gamma + pnl_vega + pnl_theta
                - execution_drag - hedge_lag_penalty - wrong_way_penalty - calibration_penalty
            )

            results.append(
                ScenarioResult(
                    name=s.name,
                    pnl=pnl,
                    pnl_breakdown={
                        "delta": pnl_delta,
                        "gamma": pnl_gamma,
                        "vega": pnl_vega,
                        "theta": pnl_theta,
                        "execution_drag": -execution_drag,
                        "hedge_lag": -hedge_lag_penalty,
                        "wrong_way_parameter_jump": -wrong_way_penalty,
                        "calibration_instability": -calibration_penalty,
                    },
                )
            )

        return results
