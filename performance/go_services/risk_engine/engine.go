package risk

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/shopspring/decimal"
	"go.uber.org/zap"
)

// Side represents the direction of a position
type Side string

const (
	SideBuy  Side = "BUY"
	SideSell Side = "SELL"
)

// Position represents a trading position
type Position struct {
	Symbol    string
	Quantity  decimal.Decimal
	AvgPrice  decimal.Decimal
	Side      Side
	Timestamp time.Time
}

// RiskLimits defines trading risk parameters
type RiskLimits struct {
	MaxPositionSize   decimal.Decimal // Max position per symbol
	MaxPortfolioValue decimal.Decimal // Max total portfolio value
	MaxDailyLoss      decimal.Decimal // Max loss per day
	MaxDrawdown       decimal.Decimal // Max drawdown percentage
	MaxLeverage       decimal.Decimal // Max leverage ratio
}

// RiskMetrics contains current risk measurements
type RiskMetrics struct {
	TotalExposure     decimal.Decimal
	TotalPnL          decimal.Decimal
	DailyPnL          decimal.Decimal
	Leverage          decimal.Decimal
	VaR95             decimal.Decimal // 95% Value at Risk
	Sharpe            float64
	VolatilityAnnual  float64
	PositionCount     int
	LimitBreaches     []string
	LastUpdated       time.Time
}

// Engine manages risk calculations and limit checks
type Engine struct {
	mu        sync.RWMutex
	positions map[string]*Position
	limits    RiskLimits
	metrics   RiskMetrics
	logger    *zap.Logger
	
	// Performance tracking
	checksPerSecond int64
	totalChecks     int64
}

// NewEngine creates a new risk engine
func NewEngine(limits RiskLimits, logger *zap.Logger) *Engine {
	return &Engine{
		positions: make(map[string]*Position),
		limits:    limits,
		logger:    logger,
		metrics: RiskMetrics{
			LastUpdated: time.Now(),
		},
	}
}

// CheckPosition validates if a new position is within risk limits
func (e *Engine) CheckPosition(ctx context.Context, symbol string, quantity decimal.Decimal, side Side, price decimal.Decimal) error {
	e.mu.RLock()
	defer e.mu.RUnlock()
	
	start := time.Now()
	defer func() {
		e.totalChecks++
		latency := time.Since(start)
		e.logger.Debug("risk check completed",
			zap.String("symbol", symbol),
			zap.Duration("latency", latency))
	}()
	
	// Check position size limit
	positionValue := quantity.Mul(price)
	if positionValue.GreaterThan(e.limits.MaxPositionSize) {
		return fmt.Errorf("position size %s exceeds limit %s", 
			positionValue, e.limits.MaxPositionSize)
	}
	
	// Check portfolio value limit
	currentExposure := e.calculateTotalExposure()
	newExposure := currentExposure.Add(positionValue)
	if newExposure.GreaterThan(e.limits.MaxPortfolioValue) {
		return fmt.Errorf("portfolio value %s would exceed limit %s",
			newExposure, e.limits.MaxPortfolioValue)
	}
	
	// Check leverage limit
	// (Simplified: actual implementation would use account equity)
	
	e.logger.Info("risk check passed",
		zap.String("symbol", symbol),
		zap.String("quantity", quantity.String()),
		zap.String("side", string(side)))
	
	return nil
}

// UpdatePosition updates or creates a position
func (e *Engine) UpdatePosition(symbol string, quantity decimal.Decimal, price decimal.Decimal, side Side) {
	e.mu.Lock()
	defer e.mu.Unlock()
	
	pos, exists := e.positions[symbol]
	if !exists {
		e.positions[symbol] = &Position{
			Symbol:    symbol,
			Quantity:  quantity,
			AvgPrice:  price,
			Side:      side,
			Timestamp: time.Now(),
		}
	} else {
		// Update existing position (simplified VWAP calculation)
		totalValue := pos.Quantity.Mul(pos.AvgPrice).Add(quantity.Mul(price))
		totalQuantity := pos.Quantity.Add(quantity)
		
		if !totalQuantity.IsZero() {
			pos.AvgPrice = totalValue.Div(totalQuantity)
			pos.Quantity = totalQuantity
			pos.Timestamp = time.Now()
		} else {
			// Position closed
			delete(e.positions, symbol)
		}
	}
	
	// Recalculate metrics
	e.recalculateMetrics()
}

// GetMetrics returns current risk metrics
func (e *Engine) GetMetrics() RiskMetrics {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.metrics
}

// GetPositions returns all current positions
func (e *Engine) GetPositions() []*Position {
	e.mu.RLock()
	defer e.mu.RUnlock()
	
	positions := make([]*Position, 0, len(e.positions))
	for _, pos := range e.positions {
		positions = append(positions, pos)
	}
	return positions
}

// calculateTotalExposure computes total portfolio exposure
func (e *Engine) calculateTotalExposure() decimal.Decimal {
	total := decimal.Zero
	for _, pos := range e.positions {
		value := pos.Quantity.Mul(pos.AvgPrice)
		total = total.Add(value.Abs())
	}
	return total
}

// recalculateMetrics updates all risk metrics
func (e *Engine) recalculateMetrics() {
	e.metrics.TotalExposure = e.calculateTotalExposure()
	e.metrics.PositionCount = len(e.positions)
	e.metrics.LastUpdated = time.Now()
	
	// Check limit breaches
	e.metrics.LimitBreaches = []string{}
	
	if e.metrics.TotalExposure.GreaterThan(e.limits.MaxPortfolioValue) {
		e.metrics.LimitBreaches = append(e.metrics.LimitBreaches,
			"Portfolio value exceeds limit")
	}
	
	// More sophisticated metrics would be calculated here:
	// - VaR using historical simulation or parametric method
	// - Sharpe ratio from return history
	// - Volatility from price history
}

// MonitorLimits continuously checks for limit breaches
func (e *Engine) MonitorLimits(ctx context.Context, alertChan chan<- string) {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			e.mu.RLock()
			breaches := e.metrics.LimitBreaches
			e.mu.RUnlock()
			
			for _, breach := range breaches {
				select {
				case alertChan <- breach:
				default:
					e.logger.Warn("alert channel full, dropping alert", zap.String("breach", breach))
				}
			}
		}
	}
}
