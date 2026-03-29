from dataclasses import dataclass
from typing import Dict, List, Tuple

from algorithms.volatility_forecasting.portfolio_engine.schemas import FillResult, OptionPositionCandidate, TargetPosition

from .attribution import PnLAttributionEngine
from .schemas import BacktestConfig, BacktestResult, DailyBacktestMetrics, MarketSnapshot


@dataclass
class LivePositionState:
    target: TargetPosition
    filled_contracts: int
    avg_entry_price: float
    days_held: int = 0


class VolatilityStrategyBacktestEngine:
    """Backtest traded system with MTM, realized PnL, greeks, costs, turnover, and drawdowns."""

    def __init__(self, config: BacktestConfig):
        self.config = config

    def run(
        self,
        targets: List[TargetPosition],
        fills: List[FillResult],
        candidate_lookup: Dict[str, OptionPositionCandidate],
        path: List[MarketSnapshot],
    ) -> BacktestResult:
        fill_map = {f.intent_id: f for f in fills if f.accepted and f.filled_contracts > 0}

        live_positions: Dict[str, LivePositionState] = {}
        for t in targets:
            fill = fill_map.get(t.intent_id)
            if fill is None:
                continue
            live_positions[t.intent_id] = LivePositionState(
                target=t,
                filled_contracts=fill.filled_contracts,
                avg_entry_price=fill.fill_price,
            )

        if not path:
            return BacktestResult(daily=[], attribution=PnLAttributionEngine().attribute([]), summary={})

        equity = self.config.initial_capital_usd
        peak = equity
        prev_spot = path[0].spot

        daily: List[DailyBacktestMetrics] = []

        for snap in path:
            dspot = snap.spot - prev_spot
            prev_spot = snap.spot

            mtm_pnl = 0.0
            realized_pnl = 0.0
            delta_pnl = 0.0
            gamma_pnl = 0.0
            vega_pnl = 0.0
            theta_pnl = 0.0
            hedge_pnl = 0.0
            hedge_costs = 0.0
            transaction_costs = 0.0
            turnover = 0.0
            capital_used = 0.0

            net_delta = 0.0
            net_gamma = 0.0
            net_vega = 0.0
            net_theta = 0.0

            to_close: List[str] = []

            for intent_id, lp in live_positions.items():
                t = lp.target
                c = candidate_lookup.get(intent_id)
                if c is None:
                    continue

                contracts = lp.filled_contracts
                if contracts <= 0:
                    continue

                signed = 1.0 if t.side == "buy" else -1.0
                d = signed * t.delta / max(abs(t.contracts), 1)
                g = signed * t.gamma / max(abs(t.contracts), 1)
                v = signed * t.vega / max(abs(t.contracts), 1)
                th = signed * t.theta / max(abs(t.contracts), 1)

                # Greeks PnL decomposition (research approximation).
                # 100 = shares per equity option contract (standard multiplier).
                # theta is already quoted in dollars/day so no /252 needed.
                p_delta = d * contracts * dspot * 100.0
                p_gamma = 0.5 * g * contracts * (dspot ** 2) * 100.0
                p_vega = v * contracts * snap.iv_change_points
                p_theta = th * contracts
                pos_mtm = p_delta + p_gamma + p_vega + p_theta

                delta_pnl += p_delta
                gamma_pnl += p_gamma
                vega_pnl += p_vega
                theta_pnl += p_theta
                mtm_pnl += pos_mtm

                # Track risk usage.
                net_delta += d * contracts
                net_gamma += g * contracts
                net_vega += v * contracts
                net_theta += th * contracts
                capital_used += abs(lp.avg_entry_price * 100.0 * contracts)

                # Hedge logic + cost/slippage.
                hedge_shares, hedge_day_pnl, hedge_day_cost = self._hedge_day(net_delta, dspot)
                hedge_pnl += hedge_day_pnl
                hedge_costs += hedge_day_cost

                # Exit conditions: horizon or adverse regime flip.
                # Keep edge-close optional by requiring near-zero IV move persistence.
                lp.days_held += 1
                regime_flip = (
                    self.config.exit_on_regime_flip
                    and snap.regime.lower() in {"stress", "crisis", "jump"}
                )
                edge_closed = (
                    abs(snap.iv_change_points) < self.config.edge_close_vol_points
                    and lp.days_held >= max(3, int(self.config.max_holding_days * 0.30))
                )
                if lp.days_held >= self.config.max_holding_days or edge_closed or regime_flip:
                    # Fair-value exit at mid (no spread) to capture only entry slippage in pnl_real.
                    # Exit spread crossing and commissions are tracked separately in transaction_costs.
                    gross_entry = lp.avg_entry_price * 100.0 * contracts
                    gross_exit_mid = c.mid * 100.0 * contracts
                    pnl_real = (gross_exit_mid - gross_entry) if t.side == "buy" else (gross_entry - gross_exit_mid)

                    exit_spread_cost = contracts * max(c.ask - c.bid, 0.0) * 100.0 * self.config.spread_cross_fraction
                    fees = contracts * self.config.option_commission_per_contract * 2.0
                    tc = exit_spread_cost + fees

                    realized_pnl += pnl_real   # entry slippage only (no tc double-count)
                    transaction_costs += tc    # exit spread + commissions
                    turnover += gross_entry + gross_exit_mid
                    to_close.append(intent_id)

            for intent_id in to_close:
                live_positions.pop(intent_id, None)

            total_pnl = mtm_pnl + realized_pnl + hedge_pnl - hedge_costs - transaction_costs
            equity += total_pnl
            peak = max(peak, equity)
            dd = 0.0 if peak <= 0 else (peak - equity) / peak

            daily.append(
                DailyBacktestMetrics(
                    date=snap.date,
                    mtm_pnl=mtm_pnl,
                    realized_pnl=realized_pnl,
                    total_pnl=total_pnl,
                    delta_pnl=delta_pnl,
                    gamma_pnl=gamma_pnl,
                    vega_pnl=vega_pnl,
                    theta_pnl=theta_pnl,
                    hedge_pnl=hedge_pnl,
                    hedge_costs=hedge_costs,
                    transaction_costs=transaction_costs,
                    turnover_usd=turnover,
                    capital_used_usd=capital_used,
                    capital_used_pct=(capital_used / self.config.initial_capital_usd) if self.config.initial_capital_usd > 0 else 0.0,
                    net_delta=net_delta,
                    net_gamma=net_gamma,
                    net_vega=net_vega,
                    net_theta=net_theta,
                    equity=equity,
                    drawdown_pct=dd,
                    regime=snap.regime,
                )
            )

        attribution = PnLAttributionEngine().attribute(daily)
        summary = self._summary(daily)
        return BacktestResult(daily=daily, attribution=attribution, summary=summary)

    def _hedge_day(self, net_delta: float, dspot: float) -> Tuple[int, float, float]:
        shares = 0
        if self.config.hedge_mode.value == "continuous":
            shares = int(round(-net_delta))
        elif self.config.hedge_mode.value == "discrete_daily":
            shares = int(round(-net_delta))
        else:
            if abs(net_delta) >= self.config.delta_hedge_threshold:
                shares = int(round(-net_delta))

        hedge_pnl = shares * dspot
        slippage = abs(shares * dspot) * (self.config.hedge_slippage_bps / 10000.0)
        fees = abs(shares) * self.config.hedge_commission_per_share
        hedge_cost = slippage + fees

        return shares, hedge_pnl, hedge_cost

    def _summary(self, daily: List[DailyBacktestMetrics]) -> Dict[str, float]:
        if not daily:
            return {}

        total_pnl = sum(d.total_pnl for d in daily)
        total_costs = sum(d.transaction_costs + d.hedge_costs for d in daily)
        gross_before_costs = sum(d.mtm_pnl + d.realized_pnl + d.hedge_pnl for d in daily)
        edge_survives_costs = 1.0 if total_pnl > 0 else 0.0

        max_drawdown = max(d.drawdown_pct for d in daily)
        avg_cap = sum(d.capital_used_pct for d in daily) / len(daily)
        turnover = sum(d.turnover_usd for d in daily)

        return {
            "total_pnl": total_pnl,
            "gross_before_costs": gross_before_costs,
            "total_costs": total_costs,
            "edge_survives_costs": edge_survives_costs,
            "max_drawdown_pct": max_drawdown,
            "average_capital_used_pct": avg_cap,
            "turnover_usd": turnover,
        }
