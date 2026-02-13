package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	
	"github.com/shopspring/decimal"
	"go.uber.org/zap"
	
	"github.com/aarya127/quant_algorithms_ai/go/risk_engine"
)

func main() {
	port := flag.Int("port", 50051, "gRPC server port")
	flag.Parse()
	
	// Initialize logger
	logger, err := zap.NewProduction()
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to initialize logger: %v\n", err)
		os.Exit(1)
	}
	defer logger.Sync()
	
	// Configure risk limits
	limits := risk.RiskLimits{
		MaxPositionSize:   decimal.NewFromInt(1000000),  // $1M per position
		MaxPortfolioValue: decimal.NewFromInt(10000000), // $10M total
		MaxDailyLoss:      decimal.NewFromInt(500000),   // $500K daily loss
		MaxDrawdown:       decimal.NewFromFloat(0.20),   // 20% max drawdown
		MaxLeverage:       decimal.NewFromFloat(3.0),    // 3x leverage
	}
	
	// Create risk engine
	engine := risk.NewEngine(limits, logger)
	
	// Start alert monitor
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	
	alertChan := make(chan string, 100)
	go engine.MonitorLimits(ctx, alertChan)
	
	// Handle alerts
	go func() {
		for alert := range alertChan {
			logger.Warn("RISK ALERT", zap.String("message", alert))
		}
	}()
	
	// Start gRPC server
	go func() {
		if err := risk.Serve(*port, engine, logger); err != nil {
			logger.Fatal("failed to serve", zap.Error(err))
		}
	}()
	
	logger.Info("risk engine started successfully", zap.Int("port", *port))
	
	// Graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	<-sigChan
	
	logger.Info("shutting down risk engine")
	cancel()
	close(alertChan)
}
