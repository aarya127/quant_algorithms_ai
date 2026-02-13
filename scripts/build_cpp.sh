#!/bin/bash
# Build script for C++ components

set -e

echo "🔨 Building C++ components..."

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check for required tools
echo -e "${BLUE}Checking dependencies...${NC}"
if ! command -v cmake &> /dev/null; then
    echo "❌ CMake not found. Install with: brew install cmake"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found."
    exit 1
fi

# Install pybind11 if needed
if ! python3 -c "import pybind11" 2>/dev/null; then
    echo "📦 Installing pybind11..."
    pip3 install pybind11
fi

# Create build directory
BUILD_DIR="performance/cpp_execution/build"
if [ -d "$BUILD_DIR" ]; then
    echo "🧹 Cleaning existing build..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
echo -e "${BLUE}Configuring CMake...${NC}"
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build
echo -e "${BLUE}Compiling (using all cores)...${NC}"
make -j$(sysctl -n hw.ncpu)

# Run tests
echo -e "${BLUE}Running tests...${NC}"
if [ -f "tests/run_tests" ]; then
    ./tests/run_tests
fi

# Install Python module
echo -e "${BLUE}Installing Python bindings...${NC}"
cd ../../..
if [ -f "cpp_bindings*.so" ]; then
    echo -e "${GREEN}✅ Build successful! Python module: cpp_bindings.so${NC}"
else
    echo "⚠️  Warning: Python bindings not found in expected location"
fi

echo -e "${GREEN}✅ C++ build complete!${NC}"
