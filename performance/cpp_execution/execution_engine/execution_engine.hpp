#pragma once

#include <string>
#include <atomic>
#include <memory>
#include <chrono>
#include <vector>
#include <unordered_map>

namespace quant {

enum class OrderSide : uint8_t {
    BUY = 0,
    SELL = 1
};

enum class OrderType : uint8_t {
    MARKET = 0,
    LIMIT = 1,
    STOP = 2,
    STOP_LIMIT = 3
};

enum class OrderStatus : uint8_t {
    PENDING = 0,
    SUBMITTED = 1,
    PARTIAL_FILL = 2,
    FILLED = 3,
    CANCELLED = 4,
    REJECTED = 5
};

struct Order {
    uint64_t order_id;
    std::string symbol;
    OrderSide side;
    OrderType type;
    OrderStatus status;
    double price;
    double quantity;
    double filled_quantity;
    std::chrono::nanoseconds timestamp;
    
    Order(const std::string& sym, OrderSide s, OrderType t, 
          double p, double q)
        : order_id(0), symbol(sym), side(s), type(t), 
          status(OrderStatus::PENDING), price(p), quantity(q), 
          filled_quantity(0.0),
          timestamp(std::chrono::high_resolution_clock::now().time_since_epoch()) {}
};

struct Fill {
    uint64_t order_id;
    double price;
    double quantity;
    std::chrono::nanoseconds timestamp;
};

class ExecutionEngine {
public:
    ExecutionEngine();
    ~ExecutionEngine();

    // Order management
    uint64_t submit_order(const std::string& symbol, OrderSide side, 
                         OrderType type, double price, double quantity);
    
    bool cancel_order(uint64_t order_id);
    
    bool modify_order(uint64_t order_id, double new_price, double new_quantity);
    
    // Order queries
    Order get_order(uint64_t order_id) const;
    
    std::vector<Order> get_open_orders() const;
    
    std::vector<Fill> get_fills(uint64_t order_id) const;
    
    // Performance metrics
    struct Metrics {
        uint64_t total_orders;
        uint64_t filled_orders;
        uint64_t cancelled_orders;
        uint64_t rejected_orders;
        double avg_latency_us;
        double p99_latency_us;
    };
    
    Metrics get_metrics() const;
    
    void reset_metrics();

private:
    std::atomic<uint64_t> next_order_id_;
    
    // Order storage (lock-free when possible)
    std::unordered_map<uint64_t, Order> orders_;
    std::unordered_map<uint64_t, std::vector<Fill>> fills_;
    
    // Metrics
    mutable std::atomic<uint64_t> total_orders_{0};
    mutable std::atomic<uint64_t> filled_orders_{0};
    mutable std::atomic<uint64_t> cancelled_orders_{0};
    mutable std::atomic<uint64_t> rejected_orders_{0};
    
    // Internal methods
    void process_order(Order& order);
    void simulate_fill(Order& order);
    uint64_t get_latency_us() const;
};

} // namespace quant
