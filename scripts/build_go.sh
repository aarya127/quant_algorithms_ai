#!/bin/bash
# Build script for Go services

set -e

echo "🔨 Building Go services..."

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check for Go
if ! command -v go &> /dev/null; then
    echo "❌ Go not found. Install from: https://go.dev/dl/"
    exit 1
fi

GO_VERSION=$(go version | awk '{print $3}')
echo -e "${BLUE}Using ${GO_VERSION}${NC}"

cd performance/go_services

# Download dependencies
echo -e "${BLUE}Downloading dependencies...${NC}"
go mod download
go mod tidy

# Generate protobuf files
echo -e "${BLUE}Generating protobuf code...${NC}"
if command -v protoc &> /dev/null; then
    protoc --go_out=. --go_opt=paths=source_relative \
           --go-grpc_out=. --go-grpc_opt=paths=source_relative \
           proto/risk.proto
    echo -e "${GREEN}✅ Protobuf generated${NC}"
else
    echo -e "${YELLOW}⚠️  protoc not found. Skipping proto generation.${NC}"
    echo "   Install with: brew install protobuf"
fi

# Build all services
echo -e "${BLUE}Building risk engine...${NC}"
go build -o bin/risk_engine ./risk_engine/main.go

# Run tests
echo -e "${BLUE}Running tests...${NC}"
go test ./... -v

echo -e "${GREEN}✅ Go build complete!${NC}"
echo "Binaries available in go/bin/"
