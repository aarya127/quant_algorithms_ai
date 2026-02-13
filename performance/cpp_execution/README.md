# C++ High-Performance Components

This directory contains ultra-low-latency components written in C++ for performance-critical operations.

## 🎯 Purpose

C++ is used **only** where latency < milliseconds matters:
- **Execution Engine**: Order routing and fills
- **Order Book Simulation**: Level 2/3 market data processing
- **Market Making Logic**: Quote generation and inventory management
- **Latency-Sensitive Hedging**: Delta hedging with microsecond precision

## 📁 Structure

```
cpp/
├── execution_engine/     # Order execution and routing
├── order_book/          # Order book simulation and matching
├── market_making/       # Market maker strategies
├── hedging/            # Low-latency hedging algorithms
├── common/             # Shared utilities (threading, memory pools, time)
├── bindings/           # Python bindings via pybind11
├── tests/              # Unit tests (Google Test)
└── benchmarks/         # Performance benchmarks
```

## 🔧 Build System

Uses CMake for cross-platform builds with optimization flags:
- `-O3`: Maximum optimization
- `-march=native`: CPU-specific optimizations
- `-flto`: Link-time optimization
- Cache-friendly data structures
- Lock-free algorithms where possible

## 🔗 Python Integration

Python bindings via **pybind11** allow seamless integration:
```python
from cpp_bindings import ExecutionEngine, OrderBook

engine = ExecutionEngine()
order_id = engine.submit_order(symbol="AAPL", quantity=100, price=150.0)
```

## ⚡ Performance Targets

- Order submission: < 10 microseconds
- Order book update: < 1 microsecond
- Market making quote: < 5 microseconds
- Delta hedge calculation: < 3 microseconds

## 🛠️ Dependencies

- **Compiler**: GCC 11+ or Clang 14+ (C++20 support)
- **CMake**: 3.20+
- **pybind11**: 2.11+ (Python bindings)
- **Google Test**: Unit testing framework
- **Google Benchmark**: Performance benchmarking

## 🚀 Building

```bash
cd cpp
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
make test
```

## 📊 Monitoring

Performance metrics exposed to Python layer:
- Order latency histograms
- Queue depths
- Cache hit rates
- Memory allocation stats
