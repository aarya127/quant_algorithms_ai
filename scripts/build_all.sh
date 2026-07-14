#!/bin/bash
# Master build script for all components

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=================================="
echo "🚀 Quant Algorithms AI - Build All"
echo "=================================="
echo ""

# Track build status
CPP_STATUS="⏩"
GO_STATUS="⏩"
PYTHON_STATUS="⏩"

# Build C++
if [ -d "performance/cpp_execution" ]; then
    echo -e "${BLUE}1/3: Building C++ components...${NC}"
    if bash scripts/build_cpp.sh; then
        CPP_STATUS="${GREEN}✅${NC}"
    else
        CPP_STATUS="${RED}❌${NC}"
        echo -e "${RED}C++ build failed${NC}"
    fi
    echo ""
else
    CPP_STATUS="${YELLOW}⏭️ ${NC}"
fi

# Build Go
if [ -d "performance/go_services" ]; then
    echo -e "${BLUE}2/3: Building Go services...${NC}"
    if bash scripts/build_go.sh; then
        GO_STATUS="${GREEN}✅${NC}"
    else
        GO_STATUS="${RED}❌${NC}"
        echo -e "${RED}Go build failed${NC}"
    fi
    echo ""
else
    GO_STATUS="${YELLOW}⏭️ ${NC}"
fi

# Install Python dependencies
echo -e "${BLUE}3/3: Installing Python dependencies...${NC}"
if pip3 install -r requirements.txt; then
    PYTHON_STATUS="${GREEN}✅${NC}"
else
    PYTHON_STATUS="${RED}❌${NC}"
    echo -e "${RED}Python setup failed${NC}"
fi

# Summary
echo ""
echo "=================================="
echo "📊 Build Summary"
echo "=================================="
echo -e "C++ Components:     $CPP_STATUS"
echo -e "Go Services:        $GO_STATUS"
echo -e "Python Dependencies: $PYTHON_STATUS"
echo "=================================="
echo ""

# Check if all succeeded
if [[ "$CPP_STATUS" == *"✅"* ]] && [[ "$GO_STATUS" == *"✅"* ]] && [[ "$PYTHON_STATUS" == *"✅"* ]]; then
    echo -e "${GREEN}✨ All components built successfully!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️  Some components failed to build${NC}"
    exit 1
fi
