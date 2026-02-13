#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>
#include "../execution_engine/execution_engine.hpp"
#include "../order_book/order_book.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cpp_bindings, m) {
    m.doc() = "C++ high-performance components for quantitative finance";

    // Enums
    py::enum_<quant::OrderSide>(m, "OrderSide")
        .value("BUY", quant::OrderSide::BUY)
        .value("SELL", quant::OrderSide::SELL);

    py::enum_<quant::OrderType>(m, "OrderType")
        .value("MARKET", quant::OrderType::MARKET)
        .value("LIMIT", quant::OrderType::LIMIT)
        .value("STOP", quant::OrderType::STOP)
        .value("STOP_LIMIT", quant::OrderType::STOP_LIMIT);

    py::enum_<quant::OrderStatus>(m, "OrderStatus")
        .value("PENDING", quant::OrderStatus::PENDING)
        .value("SUBMITTED", quant::OrderStatus::SUBMITTED)
        .value("PARTIAL_FILL", quant::OrderStatus::PARTIAL_FILL)
        .value("FILLED", quant::OrderStatus::FILLED)
        .value("CANCELLED", quant::OrderStatus::CANCELLED)
        .value("REJECTED", quant::OrderStatus::REJECTED);

    // Order struct
    py::class_<quant::Order>(m, "Order")
        .def_readonly("order_id", &quant::Order::order_id)
        .def_readonly("symbol", &quant::Order::symbol)
        .def_readonly("side", &quant::Order::side)
        .def_readonly("type", &quant::Order::type)
        .def_readonly("status", &quant::Order::status)
        .def_readonly("price", &quant::Order::price)
        .def_readonly("quantity", &quant::Order::quantity)
        .def_readonly("filled_quantity", &quant::Order::filled_quantity);

    // Fill struct
    py::class_<quant::Fill>(m, "Fill")
        .def_readonly("order_id", &quant::Fill::order_id)
        .def_readonly("price", &quant::Fill::price)
        .def_readonly("quantity", &quant::Fill::quantity);

    // ExecutionEngine
    py::class_<quant::ExecutionEngine>(m, "ExecutionEngine")
        .def(py::init<>())
        .def("submit_order", &quant::ExecutionEngine::submit_order,
             py::arg("symbol"), py::arg("side"), py::arg("type"),
             py::arg("price"), py::arg("quantity"),
             "Submit an order to the execution engine")
        .def("cancel_order", &quant::ExecutionEngine::cancel_order,
             py::arg("order_id"),
             "Cancel an existing order")
        .def("modify_order", &quant::ExecutionEngine::modify_order,
             py::arg("order_id"), py::arg("new_price"), py::arg("new_quantity"),
             "Modify an existing order")
        .def("get_order", &quant::ExecutionEngine::get_order,
             py::arg("order_id"),
             "Get order details")
        .def("get_open_orders", &quant::ExecutionEngine::get_open_orders,
             "Get all open orders")
        .def("get_fills", &quant::ExecutionEngine::get_fills,
             py::arg("order_id"),
             "Get fills for an order")
        .def("get_metrics", &quant::ExecutionEngine::get_metrics,
             "Get performance metrics")
        .def("reset_metrics", &quant::ExecutionEngine::reset_metrics,
             "Reset performance metrics");

    // ExecutionEngine Metrics
    py::class_<quant::ExecutionEngine::Metrics>(m, "ExecutionMetrics")
        .def_readonly("total_orders", &quant::ExecutionEngine::Metrics::total_orders)
        .def_readonly("filled_orders", &quant::ExecutionEngine::Metrics::filled_orders)
        .def_readonly("cancelled_orders", &quant::ExecutionEngine::Metrics::cancelled_orders)
        .def_readonly("rejected_orders", &quant::ExecutionEngine::Metrics::rejected_orders)
        .def_readonly("avg_latency_us", &quant::ExecutionEngine::Metrics::avg_latency_us)
        .def_readonly("p99_latency_us", &quant::ExecutionEngine::Metrics::p99_latency_us);

    // PriceLevel
    py::class_<quant::PriceLevel>(m, "PriceLevel")
        .def_readonly("price", &quant::PriceLevel::price)
        .def_readonly("quantity", &quant::PriceLevel::quantity)
        .def_readonly("order_count", &quant::PriceLevel::order_count);

    // MarketDepth
    py::class_<quant::MarketDepth>(m, "MarketDepth")
        .def_readonly("bids", &quant::MarketDepth::bids)
        .def_readonly("asks", &quant::MarketDepth::asks)
        .def("get_mid_price", &quant::MarketDepth::get_mid_price)
        .def("get_spread", &quant::MarketDepth::get_spread)
        .def("get_bid_volume", &quant::MarketDepth::get_bid_volume,
             py::arg("levels") = 5)
        .def("get_ask_volume", &quant::MarketDepth::get_ask_volume,
             py::arg("levels") = 5);

    // OrderBook
    py::class_<quant::OrderBook>(m, "OrderBook")
        .def(py::init<const std::string&, double>(),
             py::arg("symbol"), py::arg("tick_size") = 0.01)
        .def("add_order", &quant::OrderBook::add_order,
             py::arg("order_id"), py::arg("is_bid"), 
             py::arg("price"), py::arg("quantity"),
             "Add order to the book")
        .def("cancel_order", &quant::OrderBook::cancel_order,
             py::arg("order_id"),
             "Cancel order from the book")
        .def("modify_order", &quant::OrderBook::modify_order,
             py::arg("order_id"), py::arg("new_price"), py::arg("new_quantity"),
             "Modify order in the book")
        .def("get_depth", &quant::OrderBook::get_depth,
             py::arg("levels") = 10,
             "Get market depth snapshot")
        .def("get_best_bid", &quant::OrderBook::get_best_bid)
        .def("get_best_ask", &quant::OrderBook::get_best_ask)
        .def("get_mid_price", &quant::OrderBook::get_mid_price)
        .def("get_vwap", &quant::OrderBook::get_vwap,
             py::arg("levels") = 5)
        .def("get_stats", &quant::OrderBook::get_stats)
        .def("reset_stats", &quant::OrderBook::reset_stats);

    // OrderBook Stats
    py::class_<quant::OrderBook::Stats>(m, "OrderBookStats")
        .def_readonly("total_updates", &quant::OrderBook::Stats::total_updates)
        .def_readonly("total_trades", &quant::OrderBook::Stats::total_trades)
        .def_readonly("avg_update_time_ns", &quant::OrderBook::Stats::avg_update_time_ns)
        .def_readonly("p99_update_time_ns", &quant::OrderBook::Stats::p99_update_time_ns);
}
