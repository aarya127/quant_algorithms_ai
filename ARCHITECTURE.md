# Polyglot Architecture Overview

## Language Distribution

```
🐍 Python (70-75%)  ⚡ C++ (15-20%)  🔷 Go (10-15%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Strategy Logic      Execution Engine   Risk Engine
ML Models          Order Book         Control Plane
Backtesting        Market Making      Data Ingestion
Research           Hedging            Monitoring
Data Science       < 10μs latency     System Reliability
```

## Integration Flow

```
┌─────────────────────────────────────────────────────┐
│                 Python Strategy Layer                │
│  • Research & Development                            │
│  • Machine Learning Models                           │
│  • Backtesting & Analytics                           │
└──────────────┬──────────────────┬───────────────────┘
               │                  │
               │ pybind11         │ gRPC
               │                  │
    ┌──────────▼──────┐    ┌─────▼────────┐
    │  C++ Execution  │    │  Go Risk     │
    │  Engine         │    │  Engine      │
    │                 │    │              │
    │  • < 10μs order │    │  • < 100μs   │
    │  • < 1μs book   │    │    checks    │
    │  • Deterministic│    │  • Limits    │
    │  • Zero-copy    │    │  • Alerts    │
    └─────────────────┘    └──────────────┘
```

## When to Use Each Language

### 🐍 Python - Strategy & Research (70-75%)
**Use when:**
- Developing trading strategies
- Training ML models
- Analyzing historical data
- Backtesting algorithms
- Building dashboards
- Rapid prototyping

**Don't use when:**
- Need microsecond latency
- Inside the execution loop
- Processing tick-by-tick data
- Real-time order routing

### ⚡ C++ - Ultra-Low-Latency (15-20%)
**Use when:**
- Latency < milliseconds matters
- Order execution and fills
- Order book manipulation
- Market making quotes
- Delta hedging
- Cache effects are critical

**Don't use when:**
- Prototyping new ideas
- Need rapid iteration
- External API calls dominate
- Business logic frequently changes

### 🔷 Go - System Services (10-15%)
**Use when:**
- High concurrency needed
- Network services
- Risk monitoring
- Data ingestion pipelines
- Health checks & alerts
- System orchestration

**Don't use when:**
- Need guaranteed latency < 1ms
- Inside execution critical path
- Heavy numerical computing
- ML model training

## Performance Characteristics

| Operation | Python | Go | C++ |
|-----------|--------|-----|-----|
| Loop (1M iterations) | ~100ms | ~3ms | ~1ms |
| Dict/Map lookup | ~50ns | ~30ns | ~10ns |
| Function call | ~100ns | ~10ns | ~5ns |
| Memory allocation | GC | GC | Manual |
| Concurrency | GIL | Goroutines | Threads |

## File Organization

```
Project Root/
│
├── Python (Strategy Layer)
│   ├── algorithms/          # Trading strategies
│   ├── models/             # Quant models
│   ├── backtesting/        # Historical testing
│   ├── data/               # Data fetching
│   └── sentiment/          # ML sentiment
│
├── C++ (Execution Layer)
│   ├── execution_engine/   # < 10μs orders
│   ├── order_book/         # < 1μs updates
│   ├── market_making/      # Quote generation
│   └── bindings/           # Python interface
│
└── Go (System Layer)
    ├── risk_engine/        # < 100μs checks
    ├── control_plane/      # Orchestration
    ├── data_ingestion/     # Pipelines
    └── monitoring/         # Observability
```

## Communication Patterns

### Python → C++ (pybind11)
```python
from cpp_bindings import ExecutionEngine

engine = ExecutionEngine()
order_id = engine.submit_order(
    symbol="AAPL",
    side=OrderSide.BUY,
    type=OrderType.LIMIT,
    price=150.0,
    quantity=100
)  # Returns in < 10μs
```

### Python → Go (gRPC)
```python
from go.risk_engine.client import RiskEngineClient

risk = RiskEngineClient('localhost', 50051)
result = risk.check_position(
    symbol='AAPL',
    quantity=Decimal('1000'),
    price=Decimal('150.50'),
    side='BUY'
)  # Returns in < 100μs
```

### Integrated Workflow
```python
# 1. Python: Generate signal
signal = strategy.generate_signal()  # ~100ms

# 2. Go: Check risk limits
if risk.check_position(signal):     # ~100μs
    
    # 3. C++: Execute order
    order_id = engine.submit_order(signal)  # ~10μs
    
    # 4. Go: Update risk state
    risk.update_position(order_id)  # ~50μs
```

## Build & Deploy

```bash
# Development
make all          # Build everything
make test         # Run all tests
make clean        # Clean artifacts

# Individual components
make cpp          # Build C++ only
make go           # Build Go only
make python       # Install Python deps

# Running services
make run-risk     # Start Go risk engine
make run-backend  # Start Flask backend
```

## Monitoring

### C++ Metrics
```cpp
auto metrics = engine.get_metrics();
// avg_latency_us: 8.5μs
// p99_latency_us: 15.0μs
// total_orders: 1,000,000
```

### Go Metrics
```go
metrics := engine.GetMetrics()
// TotalExposure: $10,000,000
// Leverage: 2.5x
// LimitBreaches: []
```

### Python Analytics
```python
import pandas as pd
performance = backtest.analyze()
# Sharpe: 2.1
# Max Drawdown: -8.5%
# Win Rate: 58%
```

## Best Practices

### ✅ Do
- Use Python for strategy logic and ML
- Use C++ only in the execution loop
- Use Go for concurrent system services
- Profile before optimizing
- Keep boundaries clean
- Test each layer independently

### ❌ Don't
- Don't write complex strategies in C++
- Don't use Python for tick processing
- Don't use Go for ultra-low-latency
- Don't mix languages unnecessarily
- Don't optimize prematurely
- Don't skip integration tests

## Further Reading

- **C++ Deep Dive**: `cpp/README.md`
- **Go Services**: `go/README.md`
- **Setup Guide**: `SETUP.md`
- **Examples**: `examples/multi_language_demo.py`
- **Build System**: `Makefile`
