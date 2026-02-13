# Makefile for Quant Algorithms AI

.PHONY: all cpp go python clean test help

# Default target
all: cpp go python

help:
	@echo "Quant Algorithms AI - Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Build all components (C++, Go, Python)"
	@echo "  cpp        - Build C++ execution engine and order book"
	@echo "  go         - Build Go risk engine and services"
	@echo "  python     - Install Python dependencies"
	@echo "  test       - Run all tests"
	@echo "  clean      - Remove build artifacts"
	@echo "  run-risk   - Start Go risk engine"
	@echo "  run-backend- Start Flask backend"

cpp:
	@echo "🔨 Building C++ components..."
	@bash scripts/build_cpp.sh

go:
	@echo "🔨 Building Go services..."
	@bash scripts/build_go.sh

python:
	@echo "📦 Installing Python dependencies..."
	@pip3 install -r requirements.txt

test:
	@echo "🧪 Running tests..."
	@cd cpp/build && ctest --output-on-failure
	@cd go && go test ./... -v
	@cd tests && python3 run_all_tests.py

clean:
	@echo "🧹 Cleaning build artifacts..."
	@rm -rf performance/cpp_execution/build
	@rm -rf performance/go_services/bin
	@rm -f cpp_bindings*.so
	@rm -rf **/__pycache__
	@rm -rf **/*.pyc

# Service runners
run-risk:
	@cd performance/go_services && go run risk_engine/main.go --port 50051

run-backend:
	@cd backend && python3 app.py

# Quick rebuild
rebuild: clean all

# Development mode (with file watching)
dev:
	@echo "🔄 Starting development mode..."
	@cd backend && python3 app.py
