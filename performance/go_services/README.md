# Go System Services

This directory contains Go services for risk management, orchestration, and system reliability.

## 🎯 Purpose

Go provides **10-15% of the codebase** for:
- **Risk Engine**: Real-time risk calculations and limit checks
- **Control Plane**: Strategy orchestration and lifecycle management
- **Data Ingestion**: High-throughput market data pipeline
- **Monitoring & Alerts**: System health and performance tracking

## 📁 Structure

```
go/
├── risk_engine/        # Real-time risk management
├── control_plane/      # Strategy orchestration
├── data_ingestion/     # Market data pipelines
├── monitoring/         # Observability and alerts
├── common/            # Shared utilities
├── proto/             # gRPC protocol definitions
└── tests/             # Unit and integration tests
```

## 🔧 Key Features

### Concurrency
- **Goroutines**: Lightweight threads for parallel processing
- **Channels**: Safe communication between goroutines
- **Context**: Cancellation and timeout propagation

### Reliability
- **Graceful Shutdown**: Clean resource cleanup
- **Circuit Breakers**: Fault isolation
- **Health Checks**: Liveness and readiness probes
- **Retry Logic**: Exponential backoff

### Observability
- **Structured Logging**: JSON logs with context
- **Metrics**: Prometheus-compatible metrics
- **Tracing**: OpenTelemetry integration
- **Profiling**: pprof endpoints

## 🔗 Python Integration

Go services expose **gRPC APIs** for Python integration:

```python
import grpc
from go_proto import risk_pb2, risk_pb2_grpc

channel = grpc.insecure_channel('localhost:50051')
risk_client = risk_pb2_grpc.RiskEngineStub(channel)

response = risk_client.CheckPosition(risk_pb2.PositionRequest(
    symbol="AAPL",
    quantity=1000,
    side="BUY"
))
```

## ⚡ Performance Targets

- Risk calculation: < 100 microseconds
- Position aggregation: < 1 millisecond
- Data ingestion: > 100k msg/sec
- Alert delivery: < 10 milliseconds

## 🛠️ Dependencies

- **Go**: 1.21+ (generics support)
- **gRPC**: High-performance RPC framework
- **Protocol Buffers**: Efficient serialization
- **Prometheus**: Metrics collection
- **Zap**: Fast structured logging

## 🚀 Building

```bash
cd go
go mod download
go build ./...
go test ./...
```

## 🏃 Running Services

```bash
# Risk Engine
go run risk_engine/main.go --port 50051

# Control Plane
go run control_plane/main.go --port 50052

# Data Ingestion
go run data_ingestion/main.go --port 50053
```

## 📊 Health Checks

```bash
# Check service health
curl http://localhost:8080/health

# Metrics endpoint
curl http://localhost:8080/metrics
```
