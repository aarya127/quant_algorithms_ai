"""
Run backtest + attribution + traded-system stress test for volatility strategies.

This runner is designed to consume outputs from the portfolio engine.
It supports:
- realized and MTM PnL tracking
- Greeks over time
- hedge and transaction costs
- turnover, capital usage, drawdowns
- PnL attribution
- system-level stress scenarios
"""

import argparse
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from algorithms.volatility_forecasting.backtest_engine.engine import VolatilityStrategyBacktestEngine
from algorithms.volatility_forecasting.backtest_engine.schemas import BacktestConfig, HedgeMode, MarketSnapshot
from algorithms.volatility_forecasting.backtest_engine.stress import PortfolioStressTester
from algorithms.volatility_forecasting.portfolio_engine.schemas import FillResult, OptionPositionCandidate, TargetPosition


def synthetic_market_path(start_spot: float, n_days: int) -> list:
    out = []
    spot = start_spot
    regimes = ["normal", "normal", "normal", "stress", "normal", "jump"]

    for i in range(n_days):
        reg = regimes[i % len(regimes)]
        if reg == "normal":
            shock = random.gauss(0.0, 0.6)
            iv = random.gauss(0.0, 0.15)
        elif reg == "stress":
            shock = random.gauss(-1.2, 1.0)
            iv = random.gauss(1.2, 0.5)
        else:
            shock = random.gauss(-2.0, 1.8)
            iv = random.gauss(1.8, 0.7)

        spot = max(1.0, spot + shock)
        out.append(
            MarketSnapshot(
                date=date.today() + timedelta(days=i),
                spot=spot,
                dspot=shock,
                iv_change_points=iv,
                skew_shift_points=0.5 * iv,
                term_twist_points=0.3 * iv,
                vol_of_vol_shock_points=0.8 * iv,
                regime=reg,
            )
        )
    return out


def sample_positions() -> tuple:
    candidates = [
        OptionPositionCandidate(
            intent_id="SPY-long_call-001",
            ticker="SPY",
            option_type="call",
            side="buy",
            strike=590.0,
            maturity_days=30.0,
            confidence=0.72,
            expected_net_edge_bps=140.0,
            bid=4.90,
            ask=5.15,
            mid=5.025,
            volume=520,
            open_interest=2200,
            delta=0.52,
            gamma=0.018,
            vega=13.5,
            theta=-8.2,
            stress_loss_per_contract=625.0,
            quote_timestamp=None,
        ),
        OptionPositionCandidate(
            intent_id="SPY-short_put-002",
            ticker="SPY",
            option_type="put",
            side="sell",
            strike=560.0,
            maturity_days=25.0,
            confidence=0.66,
            expected_net_edge_bps=120.0,
            bid=3.60,
            ask=3.90,
            mid=3.75,
            volume=410,
            open_interest=1800,
            delta=0.35,
            gamma=0.015,
            vega=11.8,
            theta=7.0,
            stress_loss_per_contract=520.0,
            quote_timestamp=None,
        ),
    ]

    targets = [
        TargetPosition(
            intent_id="SPY-long_call-001",
            ticker="SPY",
            option_type="call",
            side="buy",
            strike=590.0,
            maturity_days=30.0,
            contracts=8,
            premium_notional=8 * 5.025 * 100,
            expected_net_edge_bps=140.0,
            confidence=0.72,
            delta=8 * 0.52,
            gamma=8 * 0.018,
            vega=8 * 13.5,
            theta=8 * -8.2,
            stress_loss=8 * 625.0,
        ),
        TargetPosition(
            intent_id="SPY-short_put-002",
            ticker="SPY",
            option_type="put",
            side="sell",
            strike=560.0,
            maturity_days=25.0,
            contracts=6,
            premium_notional=6 * 3.75 * 100,
            expected_net_edge_bps=120.0,
            confidence=0.66,
            delta=-6 * 0.35,
            gamma=-6 * 0.015,
            vega=-6 * 11.8,
            theta=6 * 7.0,
            stress_loss=6 * 520.0,
        ),
    ]

    fills = [
        FillResult("SPY-long_call-001", True, "filled", 8, 5.09, 8 * 0.65, 8 * 5.09 * 100 + 8 * 0.65),
        FillResult("SPY-short_put-002", True, "filled", 6, 3.68, 6 * 0.65, -(6 * 3.68 * 100) + 6 * 0.65),
    ]

    c_lookup = {c.intent_id: c for c in candidates}
    return targets, fills, c_lookup


def main():
    parser = argparse.ArgumentParser(description="Run volatility strategy backtest + attribution + stress")
    parser.add_argument("--days", type=int, default=45, help="Number of synthetic backtest days")
    parser.add_argument("--spot", type=float, default=585.0, help="Starting spot")
    parser.add_argument("--capital", type=float, default=250000.0, help="Initial capital")
    parser.add_argument("--hedge-mode", type=str, default="threshold", choices=["continuous", "discrete_daily", "threshold"])
    parser.add_argument("--export-csv", action="store_true", help="Export backtest and scenario outputs")
    args = parser.parse_args()

    targets, fills, c_lookup = sample_positions()
    path = synthetic_market_path(args.spot, args.days)

    cfg = BacktestConfig(
        initial_capital_usd=args.capital,
        hedge_mode=HedgeMode(args.hedge_mode),
    )

    engine = VolatilityStrategyBacktestEngine(cfg)
    result = engine.run(targets=targets, fills=fills, candidate_lookup=c_lookup, path=path)

    print("=" * 80)
    print("BACKTEST SUMMARY")
    print("=" * 80)
    for k, v in result.summary.items():
        if "pct" in k:
            print(f"{k:28s}: {v:.2%}")
        else:
            print(f"{k:28s}: {v:,.2f}")

    print("\n" + "=" * 80)
    print("ATTRIBUTION")
    print("=" * 80)
    attr = result.attribution
    print(f"model_edge_capture          : {attr.model_edge_capture:,.2f}")
    print(f"delta_hedge_contribution    : {attr.delta_hedge_contribution:,.2f}")
    print(f"vega_exposure               : {attr.vega_exposure:,.2f}")
    print(f"theta_carry                 : {attr.theta_carry:,.2f}")
    print(f"execution_cost              : {attr.execution_cost:,.2f}")
    print(f"spread_crossing             : {attr.spread_crossing:,.2f}")
    print("regime_performance:")
    for reg, pnl in attr.regime_performance.items():
        print(f"  - {reg:12s}: {pnl:,.2f}")

    stress = PortfolioStressTester()
    scenarios = stress.default_scenarios()
    scenario_results = stress.run(
        positions=targets,
        candidate_lookup=c_lookup,
        spot=args.spot,
        scenarios=scenarios,
    )

    print("\n" + "=" * 80)
    print("SYSTEM STRESS SCENARIOS")
    print("=" * 80)
    for sr in scenario_results:
        print(f"{sr.name:35s} pnl={sr.pnl:>12,.2f}")

    if args.export_csv:
        out_dir = Path(__file__).parent / "output"
        out_dir.mkdir(parents=True, exist_ok=True)

        daily_df = pd.DataFrame([
            {
                "date": d.date,
                "mtm_pnl": d.mtm_pnl,
                "realized_pnl": d.realized_pnl,
                "total_pnl": d.total_pnl,
                "delta_pnl": d.delta_pnl,
                "gamma_pnl": d.gamma_pnl,
                "vega_pnl": d.vega_pnl,
                "theta_pnl": d.theta_pnl,
                "hedge_pnl": d.hedge_pnl,
                "hedge_costs": d.hedge_costs,
                "transaction_costs": d.transaction_costs,
                "turnover_usd": d.turnover_usd,
                "capital_used_usd": d.capital_used_usd,
                "capital_used_pct": d.capital_used_pct,
                "net_delta": d.net_delta,
                "net_gamma": d.net_gamma,
                "net_vega": d.net_vega,
                "net_theta": d.net_theta,
                "equity": d.equity,
                "drawdown_pct": d.drawdown_pct,
                "regime": d.regime,
            }
            for d in result.daily
        ])

        scenario_df = pd.DataFrame([
            {"name": s.name, "pnl": s.pnl, **s.pnl_breakdown}
            for s in scenario_results
        ])

        daily_path = out_dir / "backtest_daily_metrics.csv"
        scenario_path = out_dir / "stress_scenarios.csv"
        daily_df.to_csv(daily_path, index=False)
        scenario_df.to_csv(scenario_path, index=False)

        print(f"\nExported daily metrics: {daily_path}")
        print(f"Exported scenario results: {scenario_path}")


if __name__ == "__main__":
    main()
