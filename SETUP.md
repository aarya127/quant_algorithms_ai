# Multi-Language Setup Guide

## Overview
This guide walks through setting up the full polyglot system: Python, C++, and Go.

## Why Multi-Language?

**Python (70-75%)** - Strategy development, research, ML
- ✅ Rich ML/data science ecosystem
- ✅ Rapid prototyping
- ✅ Easy integration with APIs
- ❌ 100-1000x slower for tight loops

**C++ (15-20%)** - Ultra-low-latency execution
- ✅ Microsecond-level performance
- ✅ Deterministic timing
- ✅ Cache-friendly data structures
- ❌ Slower development cycle
- ❌ More complex debugging

**Go (10-15%)** - System services and reliability
- ✅ Built-in concurrency (goroutines)
- ✅ Fast compilation
- ✅ Memory safety without GC pauses
- ✅ Great for network services
- ❌ Not suitable for ultra-low-latency

## Prerequisites

### macOS
```bash
# Install Xcode command line tools
xcode-select --install

# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake go protobuf
```

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install -y build-essential cmake python3-dev golang-go protobuf-compiler
```

## Quick Start

```bash
# Clone repository
git clone https://github.com/aarya127/quant_algorithms_ai.git
cd quant_algorithms_ai

# Build all components
make all

# Run tests
make test
```

## Detailed Setup

### 1. Python Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. C++ Components
```bash
cd cpp
mkdir build && cd build

# Configure with optimization
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-O3 -march=native -flto" \
      ..

# Build (using all CPU cores)
make -j$(nproc)

# Run tests
ctest --output-on-failure

cd ../..
```

**Expected output:** `cpp_bindings.so` in project root

### 3. Go Services
```bash
cd go

# Download dependencies
go mod download

# Generate protobuf code
protoc --go_out=. --go_opt=paths=source_relative \
       --go-grpc_out=. --go-grpc_opt=paths=source_relative \
       proto/risk.proto

# Build risk engine
go build -o bin/risk_engine ./risk_engine/main.go

# Run tests
go test ./... -v

cd ..
```

## Verify Installation

### Test C++ Bindings
```python
python3 << EOF
from cpp_bindings import ExecutionEngine, OrderBook
engine = ExecutionEngine()
print("✅ C++ bindings working!")
EOF
```

### Test Go Service
```bash
# Terminal 1: Start risk engine
cd go && go run risk_engine/main.go --port 50051

# Terminal 2: Test client
python3 << EOF
from go.risk_engine.client import RiskEngineClient
client = RiskEngineClient()
print("✅ Go service working!")
EOF
```

## Running the System

### Option 1: Using Makefile
```bash
# Start risk engine (Terminal 1)
make run-risk

# Start Flask backend (Terminal 2)
make run-backend
```

### Option 2: Manual
```bash
# Terminal 1: Risk Engine
cd go
go run risk_engine/main.go --port 50051

# Terminal 2: Flask Backend
cd backend
source ../.venv/bin/activate
python app.py
```

### Option 3: Demo Script
```bash
python examples/multi_language_demo.py
```

## Troubleshooting

### C++ Build Fails
```bash
# Check CMake version (need 3.20+)
cmake --version

# Check compiler
g++ --version  # or clang++ --version

# Install pybind11
pip install pybind11
```

### Go Build Fails
```bash
# Check Go version (need 1.21+)
go version

# Clean and rebuild
cd go
go clean -modcache
go mod download
go mod tidy
```

### Python Bindings Not Found
```bash
# Check if .so file exists
ls -la cpp_bindings*.so

# If missing, rebuild C++
cd cpp/build
make clean
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### gRPC Connection Errors
```bash
# Make sure risk engine is running
ps aux | grep risk_engine

# Check port availability
lsof -i :50051

# Test connection
grpcurl -plaintext localhost:50051 list
```

## Performance Expectations

After successful build:

| Component | Metric | Target |
|-----------|--------|--------|
| C++ Order Submission | Latency | < 10 μs |
| C++ Order Book Update | Latency | < 1 μs |
| Go Risk Check | Latency | < 100 μs |
| Python Strategy Signal | Latency | < 1 s |

Run benchmarks:
```bash
cd cpp/build/benchmarks
./benchmark_execution_engine
./benchmark_order_book
```

## Next Steps

1. ✅ Complete setup following this guide
2. ✅ Run `make test` to verify all components
3. ✅ Run `python examples/multi_language_demo.py`
4. ✅ Explore C++ code in `cpp/`
5. ✅ Explore Go code in `go/`
6. ✅ Start building strategies in Python that call C++/Go

## Development Workflow

```bash
# Rebuild after C++ changes
cd cpp/build && make -j$(nproc) && cd ../..

# Rebuild after Go changes
cd go && go build ./... && cd ..

# Run all tests
make test

# Clean everything
make clean
```

## Resources

- **C++ Documentation**: `cpp/README.md`
- **Go Documentation**: `go/README.md`
- **Python Examples**: `examples/multi_language_demo.py`
- **Build Scripts**: `scripts/build_*.sh`
