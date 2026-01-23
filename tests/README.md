# API Test Suite

Comprehensive test suites for Finnhub and Alpha Vantage APIs.

## Test Files

### `test_finnhub.py`
Tests all Finnhub API endpoints:
- Company News
- Basic Financials
- Earnings Surprises
- Earnings Calendar
- Insider Transactions
- Insider Sentiment
- USA Spending Data

### `test_alphavantage.py`
Tests all Alpha Vantage API endpoints:
- Market News & Sentiment
- Company Overview
- Global Quote
- Income Statement
- Balance Sheet
- Cash Flow
- Earnings History
- Insider Transactions
- Technical Indicators (SMA, RSI)

### `run_all_tests.py`
Master test runner that executes both test suites sequentially.

## Usage

### Run Individual Tests

```bash
# Test Finnhub API (fast - ~10 seconds)
python Tests/test_finnhub.py

# Test Alpha Vantage API (slow - ~2 minutes due to rate limits)
python Tests/test_alphavantage.py
```

### Run All Tests

```bash
# Run both test suites
python Tests/run_all_tests.py
```

## Test Configuration

- **Test Symbol**: NVDA (NVIDIA)
- **Date Range**: Last 30 days
- **Rate Limiting**: 
  - Finnhub: 1 second between calls
  - Alpha Vantage: 12 seconds between calls (5 calls/minute limit)

## Output

All test results are saved to `Tests/results/` directory:

### CSV Files Generated

1. **Detailed Results**: `{api}_test_results_NVDA_{timestamp}.csv`
   - Contains every test execution with:
     - Test number
     - Timestamp
     - Endpoint name
     - Status (PASS/FAIL)
     - Response type
     - Data count
     - Error message (if failed)
     - Sample data

2. **Summary**: `{api}_summary_NVDA_{timestamp}.csv`
   - Test overview with:
     - Total tests
     - Passed/Failed counts
     - Success rate
     - Test symbol
     - Timestamp

## Latest Test Results

**Date**: 2025-11-09

### Finnhub API
- ✅ Total Tests: 7
- ✅ Passed: 7 (100.0%)
- ✅ Failed: 0 (0.0%)

### Alpha Vantage API
- ✅ Total Tests: 10
- ✅ Passed: 10 (100.0%)
- ✅ Failed: 0 (0.0%)

## Test Coverage

### Finnhub (7 endpoints)
1. ✓ Company News
2. ✓ Basic Financials
3. ✓ Earnings Surprises
4. ✓ Earnings Calendar
5. ✓ Insider Transactions
6. ✓ Insider Sentiment
7. ✓ USA Spending

### Alpha Vantage (10 endpoints)
1. ✓ Market News & Sentiment
2. ✓ Company Overview
3. ✓ Global Quote
4. ✓ Income Statement
5. ✓ Balance Sheet
6. ✓ Cash Flow
7. ✓ Earnings History
8. ✓ Insider Transactions
9. ✓ SMA (Technical Indicator)
10. ✓ RSI (Technical Indicator)

## Notes

- **Alpha Vantage Rate Limits**: Free tier allows 25 requests/day. Running the full test suite uses 10 requests.
- **Test Symbol**: Change `TEST_SYMBOL` in each test file to test different stocks
- **Results Directory**: Automatically created if it doesn't exist
- **Timestamps**: Each test run creates unique CSV files with timestamps

## Troubleshooting

### "Rate limit exceeded"
- **Alpha Vantage**: Wait 24 hours or reduce test count
- **Finnhub**: Wait 1 minute

### "API key not found"
- Ensure `keys.txt` exists in project root with valid API keys

### "Module not found"
- Activate virtual environment: `source .venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
