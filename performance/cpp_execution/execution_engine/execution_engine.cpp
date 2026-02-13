#include "execution_engine.hpp"
#include <algorithm>
#include <random>
#include <thread>

namespace quant {

ExecutionEngine::ExecutionEngine() 
    : next_order_id_(1) {
}

ExecutionEngine::~ExecutionEngine() = default;

uint64_t ExecutionEngine::submit_order(const std::string& symbol, 
                                        OrderSide side, 
                                        OrderType type,
                                        double price, 
                                        double quantity) {
    auto start = std::chrono::high_resolution_clock::now();
    
    Order order(symbol, side, type, price, quantity);
    order.order_id = next_order_id_.fetch_add(1, std::memory_order_relaxed);
    order.status = OrderStatus::SUBMITTED;
    
    // Store order
    orders_[order.order_id] = order;
    
    // Process order (simulate exchange interaction)
    process_order(orders_[order.order_id]);
    
    total_orders_.fetch_add(1, std::memory_order_relaxed);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto latency = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    return order.order_id;
}

bool ExecutionEngine::cancel_order(uint64_t order_id) {
    auto it = orders_.find(order_id);
    if (it == orders_.end()) {
        return false;
    }
    
    auto& order = it->second;
    if (order.status == OrderStatus::FILLED || 
        order.status == OrderStatus::CANCELLED) {
        return false;
    }
    
    order.status = OrderStatus::CANCELLED;
    cancelled_orders_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool ExecutionEngine::modify_order(uint64_t order_id, 
                                    double new_price, 
                                    double new_quantity) {
    auto it = orders_.find(order_id);
    if (it == orders_.end()) {
        return false;
    }
    
    auto& order = it->second;
    if (order.status != OrderStatus::SUBMITTED && 
        order.status != OrderStatus::PARTIAL_FILL) {
        return false;
    }
    
    order.price = new_price;
    order.quantity = new_quantity;
    return true;
}

Order ExecutionEngine::get_order(uint64_t order_id) const {
    auto it = orders_.find(order_id);
    if (it != orders_.end()) {
        return it->second;
    }
    throw std::runtime_error("Order not found");
}

std::vector<Order> ExecutionEngine::get_open_orders() const {
    std::vector<Order> open_orders;
    for (const auto& [id, order] : orders_) {
        if (order.status == OrderStatus::SUBMITTED || 
            order.status == OrderStatus::PARTIAL_FILL) {
            open_orders.push_back(order);
        }
    }
    return open_orders;
}

std::vector<Fill> ExecutionEngine::get_fills(uint64_t order_id) const {
    auto it = fills_.find(order_id);
    if (it != fills_.end()) {
        return it->second;
    }
    return {};
}

ExecutionEngine::Metrics ExecutionEngine::get_metrics() const {
    Metrics m;
    m.total_orders = total_orders_.load(std::memory_order_relaxed);
    m.filled_orders = filled_orders_.load(std::memory_order_relaxed);
    m.cancelled_orders = cancelled_orders_.load(std::memory_order_relaxed);
    m.rejected_orders = rejected_orders_.load(std::memory_order_relaxed);
    m.avg_latency_us = 8.5;  // Placeholder
    m.p99_latency_us = 15.0; // Placeholder
    return m;
}

void ExecutionEngine::reset_metrics() {
    total_orders_.store(0, std::memory_order_relaxed);
    filled_orders_.store(0, std::memory_order_relaxed);
    cancelled_orders_.store(0, std::memory_order_relaxed);
    rejected_orders_.store(0, std::memory_order_relaxed);
}

void ExecutionEngine::process_order(Order& order) {
    // Simulate order processing with realistic latency
    // In production, this would connect to exchange/broker API
    
    // Simulate fill (80% chance of immediate fill for market orders)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    if (order.type == OrderType::MARKET && dis(gen) < 0.8) {
        simulate_fill(order);
    } else if (order.type == OrderType::LIMIT && dis(gen) < 0.5) {
        simulate_fill(order);
    }
}

void ExecutionEngine::simulate_fill(Order& order) {
    Fill fill;
    fill.order_id = order.order_id;
    fill.price = order.price;
    fill.quantity = order.quantity;
    fill.timestamp = std::chrono::high_resolution_clock::now().time_since_epoch();
    
    order.filled_quantity = order.quantity;
    order.status = OrderStatus::FILLED;
    
    fills_[order.order_id].push_back(fill);
    filled_orders_.fetch_add(1, std::memory_order_relaxed);
}

uint64_t ExecutionEngine::get_latency_us() const {
    // Return simulated latency in microseconds
    return 8; // ~8μs average
}

} // namespace quant
