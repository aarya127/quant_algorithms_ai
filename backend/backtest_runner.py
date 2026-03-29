"""
backtest_runner.py — bridges the Flask app to VolatilityStrategyBacktestEngine.

Responsibilities:
  1. Fetch real historical price data for `asset` via yfinance.
  2. Build a MarketSnapshot path (spot, dspot, iv_change derived from realised vol).
  3. Construct strategy-specific option positions anchored to the starting spot price.
  4. Run VolatilityStrategyBacktestEngine and return a JSON-serialisable result dict.

Three strategies:
  - Volatility Mean Reversion : short OTM strangle, threshold delta hedge
  - Delta-Neutral Short Vol   : short ATM straddle, continuous delta hedge
  - Long Gamma Momentum       : long ATM / short OTM call spread, daily hedge
"""

import math
import statistics
import sys
import os

import yfinance as yf

# Ensure workspace root is on the import path when running from backend/.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from algorithms.volatility_forecasting.backtest_engine.attribution import PnLAttributionEngine
from algorithms.volatility_forecasting.backtest_engine.engine import VolatilityStrategyBacktestEngine
from algorithms.volatility_forecasting.backtest_engine.schemas import (
    BacktestConfig,
    HedgeMode,
    MarketSnapshot,
)
from algorithms.volatility_forecasting.portfolio_engine.schemas import (
    FillResult,
    OptionPositionCandidate,
    TargetPosition,
)

# ---------------------------------------------------------------------------
# Strategy configuration presets
# ---------------------------------------------------------------------------

STRATEGIES = {
    "Volatility Mean Reversion": dict(
        hedge_mode=HedgeMode.THRESHOLD,
        max_holding_days=20,
        delta_hedge_threshold=25.0,
        edge_close_vol_points=0.0,   # disable early exit; hold through full window
    ),
    "Delta-Neutral Short Vol": dict(
        hedge_mode=HedgeMode.CONTINUOUS,
        max_holding_days=15,
        delta_hedge_threshold=10.0,
        edge_close_vol_points=0.0,
    ),
    "Long Gamma Momentum": dict(
        hedge_mode=HedgeMode.DISCRETE_DAILY,
        max_holding_days=10,
        delta_hedge_threshold=15.0,
        edge_close_vol_points=0.0,
    ),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _approx_option_price(S: float, K: float, call: bool = True, days: int = 30, vol: float = 0.20) -> float:
    """Black-Scholes approximation (no external lib required)."""
    T = days / 252.0
    if T <= 0 or vol <= 0 or S <= 0:
        return 0.01
    d1 = (math.log(S / K) + 0.5 * vol ** 2 * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)

    def ncdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

    if call:
        return max(0.01, S * ncdf(d1) - K * ncdf(d2))
    return max(0.01, K * ncdf(-d2) - S * ncdf(-d1))


def _build_market_path(ticker: str, start: str, end: str):
    """
    Fetch OHLCV data from yfinance and produce a list of MarketSnapshot objects.
    Also returns (start_spot, end_spot) as floats.
    """
    df = yf.Ticker(ticker).history(start=start, end=end, interval="1d")
    if df.empty or len(df) < 2:
        raise ValueError(f"No price data found for {ticker} between {start} and {end}. "
                         "Check the ticker symbol and date range.")

    df["returns"] = df["Close"].pct_change().fillna(0)
    # 5-day rolling realised vol in vol-point units (×100 so ~0.1–2.0 range).
    # Use min_periods=2 + fillna(0) so the result doesn't depend on the
    # full-series std(), which changes if yfinance returns a different row
    # count between calls (boundary-date inconsistency).
    df["rv5"] = df["returns"].rolling(5, min_periods=2).std().fillna(0.0) * 100
    df["iv_chg"] = df["rv5"].diff().fillna(0.0)

    snapshots = []
    prev_close = float(df["Close"].iloc[0])

    for ts, row in df.iterrows():
        close = float(row["Close"])
        dspot = close - prev_close
        prev_close = close

        abs_ret = abs(float(row["returns"]))
        if abs_ret > 0.03:
            regime = "crisis"
        elif abs_ret > 0.015:
            regime = "stress"
        else:
            regime = "normal"

        iv_chg = float(row["iv_chg"])
        snapshots.append(
            MarketSnapshot(
                date=ts.date() if hasattr(ts, "date") else ts,
                spot=round(close, 2),
                dspot=round(dspot, 4),
                iv_change_points=round(iv_chg, 4),
                skew_shift_points=round(iv_chg * 0.5, 4),
                term_twist_points=round(iv_chg * 0.3, 4),
                vol_of_vol_shock_points=round(iv_chg * 0.8, 4),
                regime=regime,
            )
        )

    start_spot = float(df["Close"].iloc[0])
    end_spot = float(df["Close"].iloc[-1])
    return snapshots, start_spot, end_spot


def _build_positions(strategy: str, ticker: str, spot: float, capital: float = 250_000.0):
    """Return (targets, fills, candidate_lookup) anchored to `spot`, scaled to `capital`.

    Greek sign convention — values represent the OPTION'S own properties; the engine
    applies the buy/sell sign on top (signed = +1 buy, -1 sell):
      delta : +value for calls (positive delta),  -value for puts (negative delta)
      gamma : always +value  (gamma is positive for both calls and puts)
      vega  : always +value  (positive vega for both; engine negates for short)
      theta : always -value  (options always decay; engine negates for short → income)
    """
    atm = round(spot, 0)
    otm_call = round(spot * 1.05, 0)
    otm_put = round(spot * 0.95, 0)

    if strategy == "Volatility Mean Reversion":
        # Short OTM strangle — profits from low realised vol / positive theta
        c_mid = _approx_option_price(spot, otm_call, call=True)
        p_mid = _approx_option_price(spot, otm_put,  call=False)
        # Contract sizing: short strangle margin ≈ 13% of underlying notional per leg.
        # Deploy 30% of capital as margin → n ≈ 0.30 / 0.13 ≈ 2.3× contracts vs underlying.
        n = max(1, int(capital * 0.30 / (spot * 0.13 * 100)))

        candidates = [
            OptionPositionCandidate(
                intent_id=f"{ticker}-short_call-001", ticker=ticker,
                option_type="call", side="sell", strike=otm_call, maturity_days=30,
                confidence=0.68, expected_net_edge_bps=120.0,
                bid=round(c_mid * 0.97, 2), ask=round(c_mid * 1.03, 2), mid=round(c_mid, 2),
                volume=400, open_interest=2000,
                delta=0.30, gamma=0.012, vega=10.5, theta=-6.5,
                stress_loss_per_contract=spot * 0.05 * 100, quote_timestamp=None,
            ),
            OptionPositionCandidate(
                intent_id=f"{ticker}-short_put-002", ticker=ticker,
                option_type="put", side="sell", strike=otm_put, maturity_days=30,
                confidence=0.65, expected_net_edge_bps=110.0,
                bid=round(p_mid * 0.97, 2), ask=round(p_mid * 1.03, 2), mid=round(p_mid, 2),
                volume=380, open_interest=1800,
                delta=-0.30, gamma=0.012, vega=10.0, theta=-6.0,
                stress_loss_per_contract=spot * 0.05 * 100, quote_timestamp=None,
            ),
        ]
        targets = [
            TargetPosition(
                intent_id=f"{ticker}-short_call-001", ticker=ticker,
                option_type="call", side="sell", strike=otm_call, maturity_days=30,
                contracts=n, premium_notional=n * round(c_mid, 2) * 100,
                expected_net_edge_bps=120.0, confidence=0.68,
                delta=n * 0.30, gamma=n * 0.012, vega=n * 10.5, theta=-n * 6.5,
                stress_loss=n * spot * 0.05 * 100,
            ),
            TargetPosition(
                intent_id=f"{ticker}-short_put-002", ticker=ticker,
                option_type="put", side="sell", strike=otm_put, maturity_days=30,
                contracts=n, premium_notional=n * round(p_mid, 2) * 100,
                expected_net_edge_bps=110.0, confidence=0.65,
                delta=-n * 0.30, gamma=n * 0.012, vega=n * 10.0, theta=-n * 6.0,
                stress_loss=n * spot * 0.05 * 100,
            ),
        ]
        fills = [
            FillResult(f"{ticker}-short_call-001", True, "filled", n,
                       round(c_mid * 1.01, 2), n * 0.65, -(n * round(c_mid * 1.01, 2) * 100) + n * 0.65),
            FillResult(f"{ticker}-short_put-002",  True, "filled", n,
                       round(p_mid * 1.01, 2), n * 0.65, -(n * round(p_mid * 1.01, 2) * 100) + n * 0.65),
        ]

    elif strategy == "Delta-Neutral Short Vol":
        # Short ATM straddle with continuous delta hedging
        c_mid = _approx_option_price(spot, atm, call=True)
        p_mid = _approx_option_price(spot, atm, call=False)
        # Contract sizing: short ATM straddle margin ≈ 20% of underlying notional.
        # Deploy 30% of capital as margin.
        n = max(1, int(capital * 0.30 / (spot * 0.20 * 100)))

        candidates = [
            OptionPositionCandidate(
                intent_id=f"{ticker}-short_atm_call-001", ticker=ticker,
                option_type="call", side="sell", strike=atm, maturity_days=25,
                confidence=0.70, expected_net_edge_bps=130.0,
                bid=round(c_mid * 0.97, 2), ask=round(c_mid * 1.03, 2), mid=round(c_mid, 2),
                volume=600, open_interest=3000,
                delta=0.50, gamma=0.020, vega=14.0, theta=-8.0,
                stress_loss_per_contract=spot * 0.04 * 100, quote_timestamp=None,
            ),
            OptionPositionCandidate(
                intent_id=f"{ticker}-short_atm_put-002", ticker=ticker,
                option_type="put", side="sell", strike=atm, maturity_days=25,
                confidence=0.68, expected_net_edge_bps=125.0,
                bid=round(p_mid * 0.97, 2), ask=round(p_mid * 1.03, 2), mid=round(p_mid, 2),
                volume=580, open_interest=2800,
                delta=-0.50, gamma=0.020, vega=13.5, theta=-7.5,
                stress_loss_per_contract=spot * 0.04 * 100, quote_timestamp=None,
            ),
        ]
        targets = [
            TargetPosition(
                intent_id=f"{ticker}-short_atm_call-001", ticker=ticker,
                option_type="call", side="sell", strike=atm, maturity_days=25,
                contracts=n, premium_notional=n * round(c_mid, 2) * 100,
                expected_net_edge_bps=130.0, confidence=0.70,
                delta=n * 0.50, gamma=n * 0.020, vega=n * 14.0, theta=-n * 8.0,
                stress_loss=n * spot * 0.04 * 100,
            ),
            TargetPosition(
                intent_id=f"{ticker}-short_atm_put-002", ticker=ticker,
                option_type="put", side="sell", strike=atm, maturity_days=25,
                contracts=n, premium_notional=n * round(p_mid, 2) * 100,
                expected_net_edge_bps=125.0, confidence=0.68,
                delta=-n * 0.50, gamma=n * 0.020, vega=n * 13.5, theta=-n * 7.5,
                stress_loss=n * spot * 0.04 * 100,
            ),
        ]
        fills = [
            FillResult(f"{ticker}-short_atm_call-001", True, "filled", n,
                       round(c_mid * 1.01, 2), n * 0.65, -(n * round(c_mid * 1.01, 2) * 100) + n * 0.65),
            FillResult(f"{ticker}-short_atm_put-002",  True, "filled", n,
                       round(p_mid * 1.01, 2), n * 0.65, -(n * round(p_mid * 1.01, 2) * 100) + n * 0.65),
        ]

    else:  # Long Gamma Momentum — long ATM / short OTM call spread
        c_mid     = _approx_option_price(spot, atm,      call=True, vol=0.20)
        otm_c_mid = _approx_option_price(spot, otm_call, call=True, vol=0.22)
        # Contract sizing: size by net debit (long premium minus short credit received).
        # Deploy 10% of capital in net premium cost — keeps capital-at-risk bounded.
        net_debit = max(c_mid - otm_c_mid, 0.50)
        n = max(1, int(capital * 0.10 / (net_debit * 100)))

        candidates = [
            OptionPositionCandidate(
                intent_id=f"{ticker}-long_atm_call-001", ticker=ticker,
                option_type="call", side="buy", strike=atm, maturity_days=20,
                confidence=0.62, expected_net_edge_bps=90.0,
                bid=round(c_mid * 0.97, 2), ask=round(c_mid * 1.03, 2), mid=round(c_mid, 2),
                volume=500, open_interest=2500,
                delta=0.52, gamma=0.022, vega=13.0, theta=-7.5,
                stress_loss_per_contract=round(c_mid, 2) * 100, quote_timestamp=None,
            ),
            OptionPositionCandidate(
                intent_id=f"{ticker}-short_otm_call-002", ticker=ticker,
                option_type="call", side="sell", strike=otm_call, maturity_days=20,
                confidence=0.60, expected_net_edge_bps=80.0,
                bid=round(otm_c_mid * 0.97, 2), ask=round(otm_c_mid * 1.03, 2), mid=round(otm_c_mid, 2),
                volume=450, open_interest=2200,
                delta=0.28, gamma=0.014, vega=9.5, theta=-5.0,
                stress_loss_per_contract=round(otm_c_mid, 2) * 100, quote_timestamp=None,
            ),
        ]
        targets = [
            TargetPosition(
                intent_id=f"{ticker}-long_atm_call-001", ticker=ticker,
                option_type="call", side="buy", strike=atm, maturity_days=20,
                contracts=n, premium_notional=n * round(c_mid, 2) * 100,
                expected_net_edge_bps=90.0, confidence=0.62,
                delta=n * 0.52, gamma=n * 0.022, vega=n * 13.0, theta=-n * 7.5,
                stress_loss=n * round(c_mid, 2) * 100,
            ),
            TargetPosition(
                intent_id=f"{ticker}-short_otm_call-002", ticker=ticker,
                option_type="call", side="sell", strike=otm_call, maturity_days=20,
                contracts=n, premium_notional=n * round(otm_c_mid, 2) * 100,
                expected_net_edge_bps=80.0, confidence=0.60,
                delta=n * 0.28, gamma=n * 0.014, vega=n * 9.5, theta=-n * 5.0,
                stress_loss=n * round(otm_c_mid, 2) * 100,
            ),
        ]
        fills = [
            FillResult(f"{ticker}-long_atm_call-001", True, "filled", n,
                       round(c_mid * 1.01, 2), n * 0.65, n * round(c_mid * 1.01, 2) * 100 + n * 0.65),
            FillResult(f"{ticker}-short_otm_call-002", True, "filled", n,
                       round(otm_c_mid * 1.01, 2), n * 0.65,
                       -(n * round(otm_c_mid * 1.01, 2) * 100) + n * 0.65),
        ]

    c_lookup = {c.intent_id: c for c in candidates}
    return targets, fills, c_lookup


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_backtest(strategy: str, asset: str, start_date: str, end_date: str,
                 capital: float = 250_000.0) -> dict:
    """
    Run a real backtest and return a JSON-serialisable result dict.

    Parameters
    ----------
    strategy   : one of STRATEGIES keys
    asset      : ticker symbol recognised by yfinance (e.g. "SPY", "AAPL")
    start_date : ISO date string "YYYY-MM-DD"
    end_date   : ISO date string "YYYY-MM-DD"
    capital    : initial capital in USD
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Valid options: {list(STRATEGIES.keys())}"
        )

    ticker = asset.upper().strip()
    path, start_spot, end_spot = _build_market_path(ticker, start_date, end_date)

    strat_kwargs = STRATEGIES[strategy]
    max_holding = strat_kwargs.get("max_holding_days", 20)

    # -----------------------------------------------------------------------
    # Rolling re-entry: split the full path into windows of max_holding_days.
    # At the start of each window we re-enter the strategy anchored to the
    # current spot price and current portfolio equity.  This ensures the chart
    # shows real activity every day rather than flatling after the first exit.
    # -----------------------------------------------------------------------
    all_daily: list = []
    current_capital = capital

    for i in range(0, len(path), max_holding):
        window_path = path[i: i + max_holding]
        if not window_path:
            break
        window_spot = window_path[0].spot
        targets, fills, c_lookup = _build_positions(
            strategy, ticker, window_spot, capital=current_capital
        )
        cfg = BacktestConfig(initial_capital_usd=current_capital, **strat_kwargs)
        engine = VolatilityStrategyBacktestEngine(cfg)
        result = engine.run(
            targets=targets, fills=fills, candidate_lookup=c_lookup, path=window_path
        )
        all_daily.extend(result.daily)
        if result.daily:
            current_capital = result.daily[-1].equity

    if not all_daily:
        return {"success": False, "error": "No trading data produced. Try a wider date range."}

    # Recompute global peak-to-trough drawdown across all windows.
    peak = capital
    for d in all_daily:
        peak = max(peak, d.equity)
        d.drawdown_pct = 0.0 if peak <= 0 else (peak - d.equity) / peak

    # Aggregate summary.
    total_pnl        = sum(d.total_pnl for d in all_daily)
    total_costs      = sum(d.transaction_costs + d.hedge_costs for d in all_daily)
    gross_b4_costs   = sum(d.mtm_pnl + d.realized_pnl + d.hedge_pnl for d in all_daily)
    max_drawdown     = max(d.drawdown_pct for d in all_daily)
    avg_cap          = sum(d.capital_used_pct for d in all_daily) / len(all_daily)
    total_return_pct = round((total_pnl / capital) * 100, 2) if capital > 0 else 0.0

    daily_pnls = [d.total_pnl for d in all_daily]
    if len(daily_pnls) >= 2:
        std = statistics.stdev(daily_pnls)
        sharpe = round((statistics.mean(daily_pnls) / std) * (252 ** 0.5), 2) if std > 0 else 0.0
    else:
        sharpe = 0.0

    attr = PnLAttributionEngine().attribute(all_daily)

    return {
        "success": True,
        "strategy": strategy,
        "asset": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": capital,
        "summary": {
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": total_return_pct,
            "gross_before_costs": round(gross_b4_costs, 2),
            "total_costs": round(total_costs, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "avg_capital_used_pct": round(avg_cap * 100, 2),
            "sharpe_ratio": sharpe,
            "days": len(all_daily),
            "start_spot": round(start_spot, 2),
            "end_spot": round(end_spot, 2),
        },
        "attribution": {
            "model_edge_capture": round(attr.model_edge_capture, 2),
            "delta_hedge_contribution": round(attr.delta_hedge_contribution, 2),
            "vega_exposure": round(attr.vega_exposure, 2),
            "theta_carry": round(attr.theta_carry, 2),
            "execution_cost": round(attr.execution_cost, 2),
            "regime_performance": {k: round(v, 2) for k, v in attr.regime_performance.items()},
        },
        "equity_curve": [
            {
                "date": d.date.isoformat(),
                "equity": round(d.equity, 2),
                "total_pnl": round(d.total_pnl, 2),
                "drawdown_pct": round(d.drawdown_pct * 100, 2),
                "regime": d.regime,
            }
            for d in all_daily
        ],
    }
