#pragma once

#include <array>
#include <vector>
#include <map>
#include <memory>
#include <atomic>
#include <chrono>

namespace quant {

// Price level in order book
struct PriceLevel {
    double price;
    double quantity;
    uint32_t order_count;
    
    PriceLevel() : price(0.0), quantity(0.0), order_count(0) {}
    PriceLevel(double p, double q, uint32_t c) 
        : price(p), quantity(q), order_count(c) {}
};

// L2 Market depth snapshot
struct MarketDepth {
    std::vector<PriceLevel> bids;
    std::vector<PriceLevel> asks;
    std::chrono::nanoseconds timestamp;
    
    double get_mid_price() const {
        if (!bids.empty() && !asks.empty()) {
            return (bids[0].price + asks[0].price) / 2.0;
        }
        return 0.0;
    }
    
    double get_spread() const {
        if (!bids.empty() && !asks.empty()) {
            return asks[0].price - bids[0].price;
        }
        return 0.0;
    }
    
    double get_bid_volume(int levels = 5) const {
        double total = 0.0;
        for (int i = 0; i < std::min(levels, (int)bids.size()); ++i) {
            total += bids[i].quantity;
        }
        return total;
    }
    
    double get_ask_volume(int levels = 5) const {
        double total = 0.0;
        for (int i = 0; i < std::min(levels, (int)asks.size()); ++i) {
            total += asks[i].quantity;
        }
        return total;
    }
};

// Order book matching engine
class OrderBook {
public:
    explicit OrderBook(const std::string& symbol, double tick_size = 0.01);
    ~OrderBook();
    
    // Order operations (< 1μs target)
    void add_order(uint64_t order_id, bool is_bid, double price, double quantity);
    void cancel_order(uint64_t order_id);
    void modify_order(uint64_t order_id, double new_price, double new_quantity);
    
    // Market data queries
    MarketDepth get_depth(int levels = 10) const;
    double get_best_bid() const;
    double get_best_ask() const;
    double get_mid_price() const;
    double get_vwap(int levels = 5) const;
    
    // Statistics
    struct Stats {
        uint64_t total_updates;
        uint64_t total_trades;
        double avg_update_time_ns;
        double p99_update_time_ns;
    };
    
    Stats get_stats() const;
    void reset_stats();
    
private:
    std::string symbol_;
    double tick_size_;
    
    // Bid and ask books (price -> quantity)
    // Using std::map for ordered price levels
    std::map<double, double, std::greater<double>> bids_; // Descending
    std::map<double, double> asks_;                        // Ascending
    
    // Order tracking
    struct OrderInfo {
        bool is_bid;
        double price;
        double quantity;
    };
    std::unordered_map<uint64_t, OrderInfo> orders_;
    
    // Performance tracking
    mutable std::atomic<uint64_t> total_updates_{0};
    mutable std::atomic<uint64_t> total_trades_{0};
    
    // Internal helpers
    void match_orders();
    void remove_order_from_book(const OrderInfo& order);
};

} // namespace quant
