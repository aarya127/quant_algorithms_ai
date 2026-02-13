package risk

import (
	"context"
	"fmt"
	"net"

	"go.uber.org/zap"
	"google.golang.org/grpc"
	
	pb "github.com/aarya127/quant_algorithms_ai/go/proto"
)

// Server implements the gRPC risk engine service
type Server struct {
	pb.UnimplementedRiskEngineServer
	engine *Engine
	logger *zap.Logger
}

// NewServer creates a new gRPC risk engine server
func NewServer(engine *Engine, logger *zap.Logger) *Server {
	return &Server{
		engine: engine,
		logger: logger,
	}
}

// CheckPosition validates if a position is within risk limits
func (s *Server) CheckPosition(ctx context.Context, req *pb.PositionRequest) (*pb.PositionResponse, error) {
	s.logger.Info("received position check request",
		zap.String("symbol", req.Symbol),
		zap.String("quantity", req.Quantity),
		zap.String("side", req.Side))
	
	quantity, err := decimalFromString(req.Quantity)
	if err != nil {
		return &pb.PositionResponse{
			Approved: false,
			Message:  fmt.Sprintf("invalid quantity: %v", err),
		}, nil
	}
	
	price, err := decimalFromString(req.Price)
	if err != nil {
		return &pb.PositionResponse{
			Approved: false,
			Message:  fmt.Sprintf("invalid price: %v", err),
		}, nil
	}
	
	side := Side(req.Side)
	err = s.engine.CheckPosition(ctx, req.Symbol, quantity, side, price)
	
	if err != nil {
		return &pb.PositionResponse{
			Approved: false,
			Message:  err.Error(),
		}, nil
	}
	
	return &pb.PositionResponse{
		Approved: true,
		Message:  "Position approved",
	}, nil
}

// GetMetrics returns current risk metrics
func (s *Server) GetMetrics(ctx context.Context, req *pb.MetricsRequest) (*pb.MetricsResponse, error) {
	metrics := s.engine.GetMetrics()
	
	return &pb.MetricsResponse{
		TotalExposure:    metrics.TotalExposure.String(),
		TotalPnl:         metrics.TotalPnL.String(),
		DailyPnl:         metrics.DailyPnL.String(),
		Leverage:         metrics.Leverage.String(),
		Var95:            metrics.VaR95.String(),
		Sharpe:           metrics.Sharpe,
		Volatility:       metrics.VolatilityAnnual,
		PositionCount:    int32(metrics.PositionCount),
		LimitBreaches:    metrics.LimitBreaches,
	}, nil
}

// Serve starts the gRPC server
func Serve(port int, engine *Engine, logger *zap.Logger) error {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return fmt.Errorf("failed to listen: %w", err)
	}
	
	grpcServer := grpc.NewServer()
	pb.RegisterRiskEngineServer(grpcServer, NewServer(engine, logger))
	
	logger.Info("risk engine server starting", zap.Int("port", port))
	return grpcServer.Serve(lis)
}

// Helper to convert string to decimal
func decimalFromString(s string) (decimal.Decimal, error) {
	return decimal.NewFromString(s)
}
