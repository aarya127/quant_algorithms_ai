# High-Performance Components Reference (`performance/`)

Per-file reference for the C++ execution engine and the Go risk service. See
[ARCHITECTURE.md](../ARCHITECTURE.md) for the polyglot rationale and
[README §14](../README.md#14-high-performance-components) for the summary + design targets.

> **⚠️ Not wired into the deployed app.** Nothing under `backend/` imports
> `cpp_bindings` or the Go client (grep confirms). The only consumer is
> `examples/multi_language_demo.py`, and it guards both imports behind try/except.
> These are demo/reference components today, not part of the production request path.
>
> **⚠️ Won't build as-is.** Several rough edges below (missing CMake subdirs, a
> stubbed Python gRPC client, a Go module-path mismatch) mean `make cpp`/`make go`
> need fixes before they succeed. Documented honestly per component.

---

## C++ — `performance/cpp_execution/` (namespace `quant`, C++20)

### OrderBook (`order_book/order_book.{hpp,cpp}`)
Price-level limit order book.
- Ctor: `OrderBook(const std::string& symbol, double tick_size = 0.01)`
- `add_order(uint64_t id, bool is_bid, double price, double qty)`, `cancel_order(id)`, `modify_order(id, new_price, new_qty)`
- `get_depth(int levels=10) -> MarketDepth`, `get_best_bid()`, `get_best_ask()`, `get_mid_price()`, `get_vwap(int levels=5)`
- `get_stats() -> Stats{total_updates,total_trades,avg_update_time_ns,p99_update_time_ns}`, `reset_stats()`
- **Data structure:** ordered `std::map` for price levels — `bids_` (`std::greater`, descending), `asks_` (ascending), plus `unordered_map<uint64_t, OrderInfo>` for id lookup. Levels aggregate **quantity only** (no per-order FIFO queue).
- **Matching (`match_orders()`):** crosses `best_bid`/`best_ask`, fills `min(qty)`, erases emptied levels, bumps `total_trades_`. Prices rounded to `tick_size_`.
- Structs: `PriceLevel{price,quantity,order_count}`, `MarketDepth{bids,asks,timestamp}` (+ `get_mid_price/get_spread/get_bid_volume(levels=5)/get_ask_volume(levels=5)`).
- ⚠️ Latency stats are hardcoded placeholders (avg 850ns / p99 1500ns), not measured.

### ExecutionEngine (`execution_engine/execution_engine.{hpp,cpp}`)
Order lifecycle + simulated fills.
- Ctor `ExecutionEngine()`
- `submit_order(symbol, OrderSide side, OrderType type, double price, double qty) -> uint64_t`
- `cancel_order(id) -> bool`, `modify_order(id, new_price, new_qty) -> bool`
- `get_order(id) -> Order` (throws `std::runtime_error` if missing), `get_open_orders() -> vector<Order>`, `get_fills(id) -> vector<Fill>`
- `get_metrics() -> Metrics{total_orders,filled_orders,cancelled_orders,rejected_orders,avg_latency_us,p99_latency_us}`, `reset_metrics()`
- Enums: `OrderSide{BUY,SELL}`, `OrderType{MARKET,LIMIT,STOP,STOP_LIMIT}`, `OrderStatus{PENDING,SUBMITTED,PARTIAL_FILL,FILLED,CANCELLED,REJECTED}`.
- ⚠️ Fills are RNG-simulated (MARKET ~80% immediate, LIMIT ~50%); latency metrics are placeholders (avg 8.5µs / p99 15µs).

### Python bindings (`bindings/bindings.cpp`)
`PYBIND11_MODULE(cpp_bindings, m)` — import as **`cpp_bindings`**. Exposes:
`OrderSide`, `OrderType`, `OrderStatus`, `Order` (read-only), `Fill` (read-only),
`ExecutionEngine` (all methods above), `ExecutionMetrics` (= C++ `Metrics`),
`PriceLevel`, `MarketDepth`, `OrderBook` (all methods above), `OrderBookStats` (= C++ `Stats`).

```python
from cpp_bindings import ExecutionEngine, OrderBook, OrderSide, OrderType
eng = ExecutionEngine()
oid = eng.submit_order("NVDA", OrderSide.BUY, OrderType.LIMIT, 137.5, 100)
```

### Build (`scripts/build_cpp.sh` + CMake)
C++20; Release flags `-O3 -march=native -flto -ffast-math -funroll-loops`; `find_package(pybind11 CONFIG REQUIRED)`.
```
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(sysctl -n hw.ncpu)
```
Artifact: **`cpp_bindings*.so`** (emitted to `performance/`). ⚠️ The root `CMakeLists.txt` `add_subdirectory`s `market_making`, `hedging`, `tests`, `benchmarks` — **those dirs don't exist**, so a plain configure fails until they're added or removed. The script's `.so` check `[ -f "cpp_bindings*.so" ]` also doesn't glob correctly.

---

## Go — `performance/go_services/` (module `github.com/aarya127/quant_algorithms_ai`, go 1.21)

### Risk engine (`risk_engine/engine.go`, package `risk`)
- Decimal math via `github.com/shopspring/decimal`; logging via `go.uber.org/zap`; concurrency via `sync.RWMutex` (RLock on reads, Lock on `UpdatePosition`).
- `NewEngine(limits RiskLimits, logger) *Engine`
- `CheckPosition(ctx, symbol, quantity, side, price) error` — rejects if `quantity*price > MaxPositionSize`, or if `totalExposure + positionValue > MaxPortfolioValue`. (Leverage check is stubbed.)
- `UpdatePosition(symbol, quantity, price, side)`, `GetMetrics() RiskMetrics`, `GetPositions() []*Position`, `MonitorLimits(ctx, alertChan)` (1s ticker, non-blocking alert sends).
- `RiskLimits{MaxPositionSize, MaxPortfolioValue, MaxDailyLoss, MaxDrawdown, MaxLeverage}`, `RiskMetrics{TotalExposure, TotalPnL, DailyPnL, Leverage, VaR95, Sharpe, VolatilityAnnual, PositionCount, LimitBreaches, LastUpdated}`.

### gRPC service (`proto/risk.proto`, `server.go`, `main.go`)
- Service `RiskEngine`: `CheckPosition(PositionRequest) -> PositionResponse`, `GetMetrics(MetricsRequest) -> MetricsResponse`, `StreamAlerts(AlertRequest) -> stream AlertResponse` **(declared in proto, not implemented in server.go)**.
- Messages: `PositionRequest{symbol, quantity(str), price(str), side}` → `PositionResponse{approved bool, message}`; `MetricsResponse{total_exposure, total_pnl, daily_pnl, leverage, var_95, sharpe, volatility, position_count, limit_breaches[]}`; `AlertResponse{severity, message, timestamp}`.
- `main.go`: flag `-port` (default **50051**), hardcoded limits (MaxPositionSize $1M, MaxPortfolioValue $10M, MaxDailyLoss $500K, MaxDrawdown 0.20, MaxLeverage 3.0), starts `MonitorLimits` + alert-handler goroutines, graceful shutdown on SIGINT/SIGTERM. Run via `make run-risk`.

### Python client (`risk_engine/client.py`)
`RiskEngineClient(host='localhost', port=50051)` with `check_position(symbol, quantity: Decimal, price: Decimal, side) -> dict`, `get_metrics() -> dict`, `close()`.
> ⚠️ **Stub.** The gRPC calls and `risk_pb2`/`risk_pb2_grpc` imports are commented out; methods return hardcoded placeholders. It only works after `protoc` generates the pb2 files.

### Build (`scripts/build_go.sh` + `go.mod`)
```
cd performance/go_services
go mod download && go mod tidy
# if protoc present: generate proto stubs (--go_out / --go-grpc_out, paths=source_relative)
go build -o bin/risk_engine ./risk_engine/main.go
go test ./...
```
Artifact: **`performance/go_services/bin/risk_engine`**. ⚠️ The Go sources import package paths under `.../go/proto` and `.../go/risk_engine`, which don't match the on-disk `performance/go_services/...` layout — building requires generated proto stubs and reconciling the module path first.

---

## Wiring it in (if you ever do)

`examples/multi_language_demo.py` is the template: `from cpp_bindings import ...`
after `make cpp`, and `sys.path.append('../performance/go_services/risk_engine');
from client import RiskEngineClient` after `make go` (and un-stubbing the client).
Both are optional/guarded there — mirror that pattern rather than hard-importing.
