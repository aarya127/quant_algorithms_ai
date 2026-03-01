"""
Trading Signal Orchestration - Full Pipeline

Integrates calibration → signal generation → reality checks → stress testing.

⚠️ RESEARCH ONLY - Not a deployable trading system.

Usage:
    python run_signals.py --ticker SPY --model sabr
    python run_signals.py --ticker AAPL --model heston --export-csv
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from signal_generator import (
    VolatilitySignalGenerator, SignalStressTester, 
    RealityCheckConfig, TradingSignal
)
from strategy_signals import StrategySignalGenerator, StrategySignal

# Import calibration modules
from calibration.data_aquisition import VolatilityDataFetcher
from calibration.sabr_pricer import SABRParams, price_sabr_option


def load_calibrated_parameters(ticker: str, model: str) -> dict:
    """
    Load most recent calibration parameters.
    
    In production, this would query a database or parameter store.
    For demo, loads from calibration output files.
    """
    calib_dir = Path(__file__).parent.parent / "calibration" / "output"
    
    if not calib_dir.exists():
        print(f"⚠️ Calibration directory not found: {calib_dir}")
        print("Run calibration first: python run_calibration.py --ticker {ticker} --model {model}")
        return None
    
    # Find most recent calibration file
    pattern = f"{ticker}_{model}_params_*.json"
    files = sorted(calib_dir.glob(pattern), reverse=True)
    
    if not files:
        print(f"⚠️ No calibration found for {ticker} {model}")
        print(f"Run: python run_calibration.py --ticker {ticker} --model {model}")
        return None
    
    latest_file = files[0]
    print(f"Loading calibration: {latest_file.name}")
    
    with open(latest_file, 'r') as f:
        params = json.load(f)
    
    return params


def fetch_current_market_data(ticker: str) -> dict:
    """
    Fetch current market data for signal generation.
    
    Returns dict with:
        - spot: Current spot price
        - options: Dict[strike, option_data]
        - timestamp: Market data timestamp
    """
    print(f"\nFetching market data for {ticker}...")
    
    fetcher = VolatilityDataFetcher()
    
    # Fetch options data
    result = fetcher.fetch_option_data(
        ticker=ticker,
        min_volume=5,
        min_open_interest=1,
        max_bid_ask_spread_pct=0.50,
        min_moneyness=0.7,
        max_moneyness=1.3
    )
    
    if result is None:
        print("❌ Failed to fetch market data")
        return None
    
    options_df, spot, risk_free = result
    
    if options_df.empty:
        print("❌ No options data available")
        return None
    
    print(f"✓ Fetched {len(options_df)} options")
    print(f"  Spot: ${spot:.2f}")
    print(f"  Risk-free rate: {risk_free:.4f}")
    
    # Convert to dict format for signal generator
    options_dict = {}
    for _, row in options_df.iterrows():
        strike = row['strike']
        options_dict[strike] = {
            'bid': row['bid'],
            'ask': row['ask'],
            'mid': (row['bid'] + row['ask']) / 2,
            'volume': row['volume'],
            'open_interest': row['open_interest'],
            'iv': row['implied_volatility'],
            'maturity': row['time_to_maturity'],
            'option_type': row['option_type'],
            'strike_exists': True
        }
    
    return {
        'spot': spot,
        'risk_free': risk_free,
        'options': options_dict,
        'timestamp': datetime.now()
    }


def compute_model_ivs(spot: float, strikes: list, maturities: list,
                     params: dict, model: str, risk_free: float) -> dict:
    """
    Compute model-implied IVs for all strikes.
    
    Returns:
        Dict[strike, model_iv]
    """
    print(f"\nComputing model IVs using {model.upper()}...")
    
    model_ivs = {}
    
    if model == 'sabr':
        # Extract SABR parameters
        alpha = params['alpha']
        nu = params['nu']
        rho = params['rho']
        beta = params.get('beta', 0.5)
        
        sabr_params = SABRParams(alpha=alpha, beta=beta, rho=rho, nu=nu)
        
        print(f"  SABR params: α={alpha:.4f}, ν={nu:.4f}, ρ={rho:.4f}, β={beta:.4f}")
        
        # Compute IV for each strike using Hagan formula
        from sabr_pricer import hagan_implied_volatility
        
        for strike, maturity in zip(strikes, maturities):
            try:
                model_iv = hagan_implied_volatility(
                    F=spot,  # Forward = spot for simplicity
                    K=strike,
                    T=maturity,
                    params=sabr_params
                )
                model_ivs[strike] = model_iv
            except Exception as e:
                print(f"  ⚠️ Failed to compute IV for K={strike}: {e}")
                model_ivs[strike] = None
        
        valid_ivs = sum(1 for iv in model_ivs.values() if iv is not None)
        print(f"✓ Computed {valid_ivs}/{len(strikes)} model IVs")
    
    elif model == 'heston':
        print("⚠️ Heston IV computation not yet implemented")
        print("Using placeholder values for demonstration")
        # Placeholder: Use market IV + noise
        for strike in strikes:
            model_ivs[strike] = None
    
    else:
        print(f"❌ Unknown model: {model}")
        return {}
    
    return model_ivs


def generate_single_leg_signals(spot: float, market_data: dict, 
                                model_ivs: dict, config: RealityCheckConfig) -> list:
    """Generate signals for all single-leg options."""
    print("\n" + "="*70)
    print("GENERATING SINGLE-LEG SIGNALS")
    print("="*70)
    
    generator = VolatilitySignalGenerator(config)
    signals = []
    
    for strike, option_data in market_data['options'].items():
        model_iv = model_ivs.get(strike)
        if model_iv is None:
            continue
        
        market_iv = option_data['iv']
        maturity = option_data['maturity']
        
        # Create market data dict for signal generator
        signal_market_data = {
            'open_interest': option_data['open_interest'],
            'volume': option_data['volume'],
            'bid': option_data['bid'],
            'ask': option_data['ask'],
            'strike_exists': True
        }
        
        signal = generator.generate_signal(
            spot=spot,
            strike=strike,
            maturity=maturity,
            model_iv=model_iv,
            market_iv=market_iv,
            market_data=signal_market_data
        )
        
        signals.append(signal)
    
    print(f"\n✓ Generated {len(signals)} signals")
    return signals


def filter_and_rank_signals(signals: list, config: RealityCheckConfig) -> list:
    """Filter tradeable signals and rank by net edge."""
    print("\n" + "="*70)
    print("FILTERING & RANKING SIGNALS")
    print("="*70)
    
    # Filter by reality checks and minimum net edge
    tradeable = [
        s for s in signals
        if (s.net_edge_bps >= config.min_net_edge_bps and
            s.liquidity_adequate and
            s.spread_reasonable and
            s.strike_available and
            s.signal_type.value != 'no_trade')
    ]
    
    print(f"\nReality Check Results:")
    print(f"  Total signals: {len(signals)}")
    print(f"  Tradeable signals: {len(tradeable)}")
    print(f"  Filtered out: {len(signals) - len(tradeable)}")
    
    if not tradeable:
        print("\n⚠️ No tradeable signals found")
        return []
    
    # Rank by net edge
    tradeable.sort(key=lambda s: s.net_edge_bps, reverse=True)
    
    print(f"\nTop 5 Signals by Net Edge:")
    print("-" * 70)
    for i, sig in enumerate(tradeable[:5], 1):
        print(f"{i}. {sig.signal_type.value:20s} K=${sig.strikes[0]:.2f}  "
              f"Edge: {sig.net_edge_bps:>6.0f} bps  Conf: {sig.confidence:.1%}")
    
    return tradeable


def stress_test_signals(signals: list, market_data: dict):
    """Stress test top signals."""
    print("\n" + "="*70)
    print("STRESS TESTING TOP SIGNALS")
    print("="*70)
    
    generator = VolatilitySignalGenerator()
    stress_tester = SignalStressTester(generator)
    
    # Test top 3 signals
    for i, signal in enumerate(signals[:3], 1):
        print(f"\n--- Signal {i}: {signal.signal_type.value} K=${signal.strikes[0]:.2f} ---")
        
        # Get market data for this strike
        strike = signal.strikes[0]
        signal_market_data = {
            'open_interest': market_data['options'][strike]['open_interest'],
            'volume': market_data['options'][strike]['volume']
        }
        
        stress_tester.print_stress_test_report(signal, signal_market_data)


def export_signals_to_csv(signals: list, ticker: str, model: str, output_dir: Path):
    """Export signals to CSV for analysis."""
    if not signals:
        print("No signals to export")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame
    records = []
    for sig in signals:
        records.append({
            'timestamp': sig.timestamp,
            'signal_type': sig.signal_type.value,
            'strike': sig.strikes[0],
            'maturity_days': sig.maturities[0] * 365,
            'model_iv': sig.model_iv,
            'market_iv': sig.market_iv,
            'iv_diff_pct': (sig.model_iv - sig.market_iv) * 100,
            'edge_bps': sig.edge_bps,
            'spread_cost_bps': sig.estimated_spread_cost_bps,
            'commission_bps': sig.estimated_commission_bps,
            'total_cost_bps': sig.estimated_total_cost_bps,
            'net_edge_bps': sig.net_edge_bps,
            'confidence': sig.confidence,
            'liquidity_ok': sig.liquidity_adequate,
            'spread_ok': sig.spread_reasonable,
            'strike_ok': sig.strike_available
        })
    
    df = pd.DataFrame(records)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"signals_{ticker}_{model}_{timestamp}.csv"
    filepath = output_dir / filename
    
    df.to_csv(filepath, index=False)
    print(f"\n✓ Exported {len(df)} signals to {filepath}")
    
    # Print summary statistics
    print("\nSignal Summary:")
    print("-" * 70)
    print(df[['signal_type', 'net_edge_bps', 'confidence']].describe())


def main():
    """Main orchestration pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate trading signals from calibrated volatility models"
    )
    parser.add_argument('--ticker', type=str, required=True,
                       help='Ticker symbol (e.g., SPY, AAPL)')
    parser.add_argument('--model', type=str, required=True,
                       choices=['sabr', 'heston'],
                       help='Volatility model')
    parser.add_argument('--export-csv', action='store_true',
                       help='Export signals to CSV')
    parser.add_argument('--top-n', type=int, default=5,
                       help='Number of top signals to display (default: 5)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("TRADING SIGNAL GENERATION PIPELINE")
    print("="*70)
    print(f"\nTicker: {args.ticker}")
    print(f"Model: {args.model.upper()}")
    print(f"\n⚠️ DISCLAIMER: This is SIGNAL RESEARCH, not trading advice\n")
    
    # Step 1: Load calibrated parameters
    print("="*70)
    print("STEP 1: LOAD CALIBRATION")
    print("="*70)
    params = load_calibrated_parameters(args.ticker, args.model)
    if params is None:
        return
    
    # Step 2: Fetch market data
    print("\n" + "="*70)
    print("STEP 2: FETCH MARKET DATA")
    print("="*70)
    market_data = fetch_current_market_data(args.ticker)
    if market_data is None:
        return
    
    spot = market_data['spot']
    risk_free = market_data['risk_free']
    
    # Step 3: Compute model IVs
    print("\n" + "="*70)
    print("STEP 3: COMPUTE MODEL IVs")
    print("="*70)
    strikes = list(market_data['options'].keys())
    maturities = [market_data['options'][k]['maturity'] for k in strikes]
    
    model_ivs = compute_model_ivs(spot, strikes, maturities, params, args.model, risk_free)
    
    if not model_ivs:
        print("❌ Failed to compute model IVs")
        return
    
    # Step 4: Generate signals
    config = RealityCheckConfig(
        min_open_interest=100,
        max_bid_ask_spread_pct=0.10,
        min_volume=20,
        min_net_edge_bps=50
    )
    
    signals = generate_single_leg_signals(spot, market_data, model_ivs, config)
    
    if not signals:
        print("❌ No signals generated")
        return
    
    # Step 5: Filter and rank
    tradeable_signals = filter_and_rank_signals(signals, config)
    
    if not tradeable_signals:
        print("\n⚠️ No tradeable signals after filtering")
        print("Consider:")
        print("  - Lowering minimum net edge threshold")
        print("  - Relaxing liquidity requirements")
        print("  - Waiting for better market conditions")
        return
    
    # Step 6: Stress test top signals
    stress_test_signals(tradeable_signals[:3], market_data)
    
    # Step 7: Display top signals
    print("\n" + "="*70)
    print(f"TOP {args.top_n} TRADEABLE SIGNALS")
    print("="*70)
    
    generator = VolatilitySignalGenerator(config)
    for i, signal in enumerate(tradeable_signals[:args.top_n], 1):
        print(f"\n{'='*70}")
        print(f"SIGNAL {i}")
        print("="*70)
        generator.print_signal_report(signal)
    
    # Step 8: Export if requested
    if args.export_csv:
        output_dir = Path(__file__).parent / "output"
        export_signals_to_csv(signals, args.ticker, args.model, output_dir)
    
    # Final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"\nGenerated: {len(signals)} total signals")
    print(f"Tradeable: {len(tradeable_signals)} signals")
    print(f"Pass rate: {len(tradeable_signals)/len(signals)*100:.1f}%")
    
    if tradeable_signals:
        avg_edge = np.mean([s.net_edge_bps for s in tradeable_signals])
        max_edge = max(s.net_edge_bps for s in tradeable_signals)
        print(f"\nNet Edge Statistics:")
        print(f"  Average: {avg_edge:.0f} bps")
        print(f"  Maximum: {max_edge:.0f} bps")
    
    print("\n⚠️ REMEMBER: This is signal research, not trading advice")
    print("Real trading requires live systems, risk management, and compliance")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
