# Volatility Forecasting Backtest Engine

This package implements Phase 5-7 of the strategy stack:

- Backtest engine (realized + MTM PnL, Greeks, hedge costs, transaction costs, turnover, capital usage, drawdowns)
- PnL attribution
- Traded-system stress testing

## Modules

- `schemas.py`: backtest, attribution, and scenario data contracts
- `engine.py`: day-by-day backtest simulation
- `attribution.py`: PnL attribution decomposition
- `stress.py`: portfolio-level stress scenarios
- `run_backtest.py`: runnable demo driver and CSV export

## Run

```bash
cd /Users/aaryas127/quant_algorithms_ai
python3 -m algorithms.volatility_forecasting.backtest_engine.run_backtest --days 45 --spot 585 --export-csv
```

## Stress Scenarios Included

- spot shock + skew widening
- term-structure twist
- liquidity collapse + spread widening
- overnight gap + wrong-way parameter jump
- calibration instability event
