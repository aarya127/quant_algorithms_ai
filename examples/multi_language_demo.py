"""
Example demonstrating C++ and Go integration with Python
"""

import sys
import time
from decimal import Decimal

# Import C++ bindings (after building)
try:
    from cpp_bindings import ExecutionEngine, OrderBook, OrderSide, OrderType
    CPP_AVAILABLE = True
except ImportError:
    print("⚠️  C++ bindings not available. Run 'make cpp' first.")
    CPP_AVAILABLE = False

# Import Go client
sys.path.append('../performance/go_services/risk_engine')
try:
    from client import RiskEngineClient
    GO_AVAILABLE = True
except ImportError:
    print("⚠️  Go client not available. Run 'make go' first.")
    GO_AVAILABLE = False


def demo_cpp_execution():
    """Demonstrate C++ execution engine"""
    print("\n" + "="*50)
    print("🔥 C++ Execution Engine Demo")
    print("="*50)
    
    if not CPP_AVAILABLE:
        return
    
    # Create execution engine
    engine = ExecutionEngine()
    print("✅ Execution engine initialized")
    
    # Submit orders
    print("\n📝 Submitting orders...")
    start = time.perf_counter()
    
    order_ids = []
    for i in range(100):
        order_id = engine.submit_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            price=150.0 + i * 0.1,
            quantity=100
        )
        order_ids.append(order_id)
    
    elapsed = (time.perf_counter() - start) * 1_000_000  # microseconds
    print(f"✅ Submitted 100 orders in {elapsed:.2f}μs")
    print(f"   Average: {elapsed/100:.2f}μs per order")
    
    # Get metrics
    metrics = engine.get_metrics()
    print(f"\n📊 Execution Metrics:")
    print(f"   Total orders: {metrics.total_orders}")
    print(f"   Filled orders: {metrics.filled_orders}")
    print(f"   Avg latency: {metrics.avg_latency_us:.2f}μs")
    print(f"   P99 latency: {metrics.p99_latency_us:.2f}μs")


def demo_cpp_orderbook():
    """Demonstrate C++ order book"""
    print("\n" + "="*50)
    print("📚 C++ Order Book Demo")
    print("="*50)
    
    if not CPP_AVAILABLE:
        return
    
    # Create order book
    book = OrderBook(symbol="AAPL", tick_size=0.01)
    print("✅ Order book initialized")
    
    # Add orders
    print("\n📝 Adding orders to book...")
    start = time.perf_counter()
    
    # Add bids
    for i in range(10):
        book.add_order(
            order_id=i,
            is_bid=True,
            price=150.0 - i * 0.1,
            quantity=100 + i * 10
        )
    
    # Add asks
    for i in range(10):
        book.add_order(
            order_id=100 + i,
            is_bid=False,
            price=150.1 + i * 0.1,
            quantity=100 + i * 10
        )
    
    elapsed = (time.perf_counter() - start) * 1_000_000
    print(f"✅ Added 20 orders in {elapsed:.2f}μs")
    
    # Get market data
    depth = book.get_depth(levels=5)
    print(f"\n📊 Market Depth:")
    print(f"   Best bid: ${depth.bids[0].price:.2f} ({depth.bids[0].quantity} shares)")
    print(f"   Best ask: ${depth.asks[0].price:.2f} ({depth.asks[0].quantity} shares)")
    print(f"   Spread: ${depth.get_spread():.2f}")
    print(f"   Mid price: ${depth.get_mid_price():.2f}")
    
    # Stats
    stats = book.get_stats()
    print(f"\n📈 Order Book Stats:")
    print(f"   Total updates: {stats.total_updates}")
    print(f"   Avg update time: {stats.avg_update_time_ns:.0f}ns")


def demo_go_risk():
    """Demonstrate Go risk engine"""
    print("\n" + "="*50)
    print("🛡️  Go Risk Engine Demo")
    print("="*50)
    
    if not GO_AVAILABLE:
        return
    
    # Create client
    client = RiskEngineClient(host='localhost', port=50051)
    print("✅ Connected to risk engine")
    
    # Check position
    print("\n📝 Checking position...")
    result = client.check_position(
        symbol='AAPL',
        quantity=Decimal('1000'),
        price=Decimal('150.50'),
        side='BUY'
    )
    
    if result['approved']:
        print(f"✅ Position approved: {result['message']}")
    else:
        print(f"❌ Position rejected: {result['message']}")
    
    # Get metrics
    print("\n📊 Risk Metrics:")
    metrics = client.get_metrics()
    print(f"   Total exposure: ${metrics['total_exposure']}")
    print(f"   Position count: {metrics['position_count']}")
    if metrics['limit_breaches']:
        print(f"   ⚠️  Limit breaches: {metrics['limit_breaches']}")
    
    client.close()


def demo_integrated_workflow():
    """Demonstrate full integration: Python → Go → C++"""
    print("\n" + "="*50)
    print("🔄 Integrated Workflow Demo")
    print("="*50)
    
    if not (CPP_AVAILABLE and GO_AVAILABLE):
        print("⚠️  Requires both C++ and Go components")
        return
    
    print("\n1️⃣  Python: Strategy decision")
    symbol = "AAPL"
    quantity = Decimal('500')
    price = Decimal('150.25')
    
    print(f"   Decision: BUY {quantity} shares of {symbol} @ ${price}")
    
    print("\n2️⃣  Go: Risk check")
    risk_client = RiskEngineClient()
    risk_result = risk_client.check_position(symbol, quantity, price, 'BUY')
    
    if not risk_result['approved']:
        print(f"   ❌ Risk rejected: {risk_result['message']}")
        return
    
    print(f"   ✅ Risk approved")
    
    print("\n3️⃣  C++: Order execution")
    engine = ExecutionEngine()
    order_id = engine.submit_order(
        symbol=symbol,
        side=OrderSide.BUY,
        type=OrderType.LIMIT,
        price=float(price),
        quantity=float(quantity)
    )
    
    print(f"   ✅ Order submitted: ID {order_id}")
    
    order = engine.get_order(order_id)
    print(f"   Status: {order.status}")
    print(f"   Filled: {order.filled_quantity} / {order.quantity}")
    
    print("\n✨ Complete workflow executed successfully!")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🤖 Quant Algorithms AI - Multi-Language Demo")
    print("="*60)
    
    # Run demos
    demo_cpp_execution()
    demo_cpp_orderbook()
    demo_go_risk()
    demo_integrated_workflow()
    
    print("\n" + "="*60)
    print("✅ Demo complete!")
    print("="*60)
