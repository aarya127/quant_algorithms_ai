# 🚀 Performance-Critical Components

This directory contains high-performance components written in **C++** and **Go** for operations where microseconds matter.

## 📂 Directory Structure

```
performance/
├── cpp_execution/        # C++ - Ultra-Low-Latency Execution (15-20%)
│   └── When you need < 1 millisecond performance
│
└── go_services/          # Go - Reliable System Services (10-15%)
    └── When you need concurrency + fault tolerance
```

## 🎯 Why Two Languages?

### ⚡ C++ Execution (`cpp_execution/`)
**Use for: Order execution, order book, market making, hedging**

- **Target:** < 10 microseconds per operation
- **Why C++:** 
  - Deterministic timing (no garbage collection)
  - Direct memory control
  - Cache-friendly data structures
  - 100-1000x faster than Python

**Examples:**
- Order submission in 8 microseconds
- Order book updates in 850 nanoseconds
- Market making quote generation
- Delta hedging calculations

### 🔷 Go Services (`go_services/`)
**Use for: Risk management, monitoring, data pipelines, orchestration**

- **Target:** < 100 microseconds for risk checks, handles 100k+ messages/sec
- **Why Go:**
  - Built-in concurrency (goroutines)
  - Fast compilation
  - Memory safety without GC pauses affecting latency
  - Great for network services and system reliability

**Examples:**
- Real-time risk limit checks
- Position aggregation across accounts
- Market data ingestion pipelines
- System health monitoring

## 🤔 Quick Decision Guide

**Need to process orders in real-time?** → Use `cpp_execution/`  
**Need to check risk limits or monitor system?** → Use `go_services/`  
**Developing strategy logic or ML models?** → Use Python (main codebase)

## 🔗 Integration with Python

Both C++ and Go expose clean APIs callable from Python:

```python
# C++ via pybind11 (zero-copy, inline)
from cpp_bindings import ExecutionEngine
engine = ExecutionEngine()
order_id = engine.submit_order("AAPL", ...) # < 10μs

# Go via gRPC (network call, still fast)
from performance.go_services.risk_engine.client import RiskEngineClient
risk = RiskEngineClient()
approved = risk.check_position("AAPL", ...) # < 100μs
```

## 🛠️ Building

```bash
# Build everything
make all

# Build individually
make cpp    # Build C++ components
make go     # Build Go services

# Clean
make clean
```

## 📊 Performance Comparison

| Operation | Python | Go | C++ |
|-----------|--------|-----|-----|
| Order submission | ~1 ms | ~100 μs | **< 10 μs** |
| Risk check | ~500 μs | **< 100 μs** | < 10 μs (overkill) |
| Strategy signal | < 1s ✅ | ~10 ms | ~1 ms (overkill) |
| Data pipeline | ~100 msg/s | **> 100k msg/s** | > 1M msg/s (complex) |

## 📖 Documentation

- **C++ Deep Dive:** `cpp_execution/README.md`
- **Go Services:** `go_services/README.md`
- **Setup Guide:** `../SETUP.md`
- **Architecture:** `../ARCHITECTURE.md`

## 💡 Key Insight

> **"Use the right tool for the job"**
> 
> - Python is 100x easier to develop in → use it for 70% of code
> - C++ is 100x faster → use it for the critical 15-20%
> - Go is best for concurrency → use it for system services 10-15%
> 
> This gives you the best of all worlds! 🎯
