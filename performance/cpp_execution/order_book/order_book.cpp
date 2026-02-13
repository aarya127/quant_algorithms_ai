#include "order_book.hpp"
#include <algorithm>
#include <stdexcept>

namespace quant {

OrderBook::OrderBook(const std::string& symbol, double tick_size)
    : symbol_(symbol), tick_size_(tick_size) {
}

OrderBook::~OrderBook() = default;

void OrderBook::add_order(uint64_t order_id, bool is_bid, 
                          double price, double quantity) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Round price to tick size
    price = std::round(price / tick_size_) * tick_size_;
    
    // Add to order tracking
    orders_[order_id] = OrderInfo{is_bid, price, quantity};
    
    // Add to book
    if (is_bid) {
        bids_[price] += quantity;
    } else {
        asks_[price] += quantity;
    }
    
    // Attempt matching
    match_orders();
    
    total_updates_.fetch_add(1, std::memory_order_relaxed);
    
    auto end = std::chrono::high_resolution_clock::now();
    // Track latency here if needed
}

void OrderBook::cancel_order(uint64_t order_id) {
    auto it = orders_.find(order_id);
    if (it == orders_.end()) {
        return; // Order not found
    }
    
    remove_order_from_book(it->second);
    orders_.erase(it);
    
    total_updates_.fetch_add(1, std::memory_order_relaxed);
}

void OrderBook::modify_order(uint64_t order_id, 
                              double new_price, 
                              double new_quantity) {
    auto it = orders_.find(order_id);
    if (it == orders_.end()) {
        return;
    }
    
    // Remove old order
    remove_order_from_book(it->second);
    
    // Add new order
    new_price = std::round(new_price / tick_size_) * tick_size_;
    it->second.price = new_price;
    it->second.quantity = new_quantity;
    
    if (it->second.is_bid) {
        bids_[new_price] += new_quantity;
    } else {
        asks_[new_price] += new_quantity;
    }
    
    match_orders();
    total_updates_.fetch_add(1, std::memory_order_relaxed);
}

MarketDepth OrderBook::get_depth(int levels) const {
    MarketDepth depth;
    depth.timestamp = std::chrono::high_resolution_clock::now().time_since_epoch();
    
    // Bids (descending)
    int count = 0;
    for (const auto& [price, qty] : bids_) {
        if (count++ >= levels) break;
        depth.bids.emplace_back(price, qty, 1);
    }
    
    // Asks (ascending)
    count = 0;
    for (const auto& [price, qty] : asks_) {
        if (count++ >= levels) break;
        depth.asks.emplace_back(price, qty, 1);
    }
    
    return depth;
}

double OrderBook::get_best_bid() const {
    if (bids_.empty()) return 0.0;
    return bids_.begin()->first;
}

double OrderBook::get_best_ask() const {
    if (asks_.empty()) return 0.0;
    return asks_.begin()->first;
}

double OrderBook::get_mid_price() const {
    if (bids_.empty() || asks_.empty()) return 0.0;
    return (get_best_bid() + get_best_ask()) / 2.0;
}

double OrderBook::get_vwap(int levels) const {
    double total_value = 0.0;
    double total_volume = 0.0;
    
    // VWAP from bids
    int count = 0;
    for (const auto& [price, qty] : bids_) {
        if (count++ >= levels) break;
        total_value += price * qty;
        total_volume += qty;
    }
    
    // VWAP from asks
    count = 0;
    for (const auto& [price, qty] : asks_) {
        if (count++ >= levels) break;
        total_value += price * qty;
        total_volume += qty;
    }
    
    return total_volume > 0.0 ? total_value / total_volume : 0.0;
}

OrderBook::Stats OrderBook::get_stats() const {
    Stats s;
    s.total_updates = total_updates_.load(std::memory_order_relaxed);
    s.total_trades = total_trades_.load(std::memory_order_relaxed);
    s.avg_update_time_ns = 850.0;  // Placeholder: ~850ns
    s.p99_update_time_ns = 1500.0; // Placeholder: ~1.5μs
    return s;
}

void OrderBook::reset_stats() {
    total_updates_.store(0, std::memory_order_relaxed);
    total_trades_.store(0, std::memory_order_relaxed);
}

void OrderBook::match_orders() {
    // Simple matching logic: cross the spread
    while (!bids_.empty() && !asks_.empty()) {
        double best_bid = bids_.begin()->first;
        double best_ask = asks_.begin()->first;
        
        if (best_bid < best_ask) {
            break; // No match
        }
        
        // Match available quantity
        double& bid_qty = bids_.begin()->second;
        double& ask_qty = asks_.begin()->second;
        double match_qty = std::min(bid_qty, ask_qty);
        
        bid_qty -= match_qty;
        ask_qty -= match_qty;
        
        // Remove empty levels
        if (bid_qty <= 0.0) bids_.erase(bids_.begin());
        if (ask_qty <= 0.0) asks_.erase(asks_.begin());
        
        total_trades_.fetch_add(1, std::memory_order_relaxed);
    }
}

void OrderBook::remove_order_from_book(const OrderInfo& order) {
    if (order.is_bid) {
        auto it = bids_.find(order.price);
        if (it != bids_.end()) {
            it->second -= order.quantity;
            if (it->second <= 0.0) {
                bids_.erase(it);
            }
        }
    } else {
        auto it = asks_.find(order.price);
        if (it != asks_.end()) {
            it->second -= order.quantity;
            if (it->second <= 0.0) {
                asks_.erase(it);
            }
        }
    }
}

} // namespace quant
