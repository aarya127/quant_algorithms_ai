"""
Test Suite for Alpha Vantage API
Tests all endpoints with NVDA stock and saves results to CSV
"""

import sys
import os
import datetime
import csv
import json
from time import sleep

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Data.alphavantage import AlphaVantage

# Test configuration
TEST_SYMBOL = "NVDA"
TEST_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# Create output directory
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)


class AlphaVantageTestSuite:
    def __init__(self):
        self.av = AlphaVantage()
        self.results = []
        self.test_count = 0
        self.passed = 0
        self.failed = 0
        
    def log_result(self, endpoint, status, response_type, data_count, error=None, sample_data=None):
        """Log test result"""
        self.test_count += 1
        
        result = {
            'test_number': self.test_count,
            'timestamp': datetime.datetime.now().isoformat(),
            'endpoint': endpoint,
            'status': status,
            'response_type': response_type,
            'data_count': data_count,
            'error': error or '',
            'sample_data': json.dumps(sample_data) if sample_data else ''
        }
        
        self.results.append(result)
        
        if status == 'PASS':
            self.passed += 1
            print(f"✓ Test {self.test_count}: {endpoint} - PASSED")
        else:
            self.failed += 1
            print(f"✗ Test {self.test_count}: {endpoint} - FAILED")
            if error:
                print(f"  Error: {error}")
        
        if sample_data:
            print(f"  Sample: {str(sample_data)[:100]}...")
        print()
        
        # Small delay to respect API rate limits
        sleep(12)  # Alpha Vantage: 5 calls per minute = 12 seconds between calls
    
    def test_endpoint(self, name, func, *args, **kwargs):
        """Generic endpoint test wrapper"""
        print(f"Testing: {name}...")
        try:
            response = func(*args, **kwargs)
            
            if not response:
                self.log_result(name, 'FAIL', type(response).__name__, 0, 
                              error="Empty response")
                return
            
            # Check for API errors
            if isinstance(response, dict):
                if 'Error Message' in response:
                    self.log_result(name, 'FAIL', 'dict', 0, 
                                  error=response['Error Message'])
                    return
                if 'Note' in response:
                    self.log_result(name, 'FAIL', 'dict', 0, 
                                  error=f"Rate limit: {response['Note']}")
                    return
            
            # Count data items
            data_count = 0
            sample_data = None
            
            if isinstance(response, dict):
                data_count = len(response)
                # Get sample of first key
                if response:
                    first_key = list(response.keys())[0]
                    sample_data = {first_key: response[first_key]}
            elif isinstance(response, list):
                data_count = len(response)
                sample_data = response[0] if response else None
            
            self.log_result(name, 'PASS', type(response).__name__, data_count, 
                          sample_data=sample_data)
            
        except Exception as e:
            self.log_result(name, 'FAIL', 'exception', 0, error=str(e))
    
    def run_all_tests(self):
        """Run all Alpha Vantage endpoint tests"""
        print("="*70)
        print(f"ALPHA VANTAGE API TEST SUITE - {TEST_SYMBOL}")
        print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        print()
        
        # News & Sentiment
        print("📰 NEWS & SENTIMENT TESTS")
        print("-"*70)
        self.test_endpoint(
            "get_market_news_sentiment",
            self.av.get_market_news_sentiment,
            tickers=TEST_SYMBOL,
            limit=5
        )
        
        # Company Overview & Fundamentals
        print("🏢 COMPANY OVERVIEW & FUNDAMENTALS")
        print("-"*70)
        self.test_endpoint(
            "get_company_overview",
            self.av.get_company_overview,
            TEST_SYMBOL
        )
        
        self.test_endpoint(
            "get_global_quote",
            self.av.get_global_quote,
            TEST_SYMBOL
        )
        
        self.test_endpoint(
            "get_income_statement",
            self.av.get_income_statement,
            TEST_SYMBOL
        )
        
        self.test_endpoint(
            "get_balance_sheet",
            self.av.get_balance_sheet,
            TEST_SYMBOL
        )
        
        self.test_endpoint(
            "get_cash_flow",
            self.av.get_cash_flow,
            TEST_SYMBOL
        )
        
        self.test_endpoint(
            "get_earnings_history",
            self.av.get_earnings_history,
            TEST_SYMBOL
        )
        
        # Insider Data
        print("👔 INSIDER DATA")
        print("-"*70)
        self.test_endpoint(
            "get_insider_transactions",
            self.av.get_insider_transactions,
            TEST_SYMBOL
        )
        
        # Technical Indicators (sample)
        print("📊 TECHNICAL INDICATORS")
        print("-"*70)
        self.test_endpoint(
            "get_sma",
            self.av.get_sma,
            TEST_SYMBOL,
            interval='daily',
            time_period=20
        )
        
        self.test_endpoint(
            "get_rsi",
            self.av.get_rsi,
            TEST_SYMBOL,
            interval='daily',
            time_period=14
        )
        
        # Summary
        print("="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Total Tests: {self.test_count}")
        print(f"Passed: {self.passed} ({self.passed/self.test_count*100:.1f}%)")
        print(f"Failed: {self.failed} ({self.failed/self.test_count*100:.1f}%)")
        print("="*70)
        print()
        
    def save_results(self):
        """Save test results to CSV"""
        output_file = os.path.join(
            TEST_OUTPUT_DIR, 
            f'alphavantage_test_results_{TEST_SYMBOL}_{TIMESTAMP}.csv'
        )
        
        if not self.results:
            print("No results to save")
            return
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)
        
        print(f"✓ Results saved to: {output_file}")
        
        # Also create a summary CSV
        summary_file = os.path.join(
            TEST_OUTPUT_DIR,
            f'alphavantage_summary_{TEST_SYMBOL}_{TIMESTAMP}.csv'
        )
        
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Test Symbol', TEST_SYMBOL])
            writer.writerow(['Total Tests', self.test_count])
            writer.writerow(['Passed', self.passed])
            writer.writerow(['Failed', self.failed])
            writer.writerow(['Success Rate', f"{self.passed/self.test_count*100:.1f}%"])
            writer.writerow(['Timestamp', TIMESTAMP])
        
        print(f"✓ Summary saved to: {summary_file}")
        print()


def main():
    """Run Alpha Vantage test suite"""
    suite = AlphaVantageTestSuite()
    
    try:
        suite.run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test suite error: {e}")
    finally:
        suite.save_results()


if __name__ == "__main__":
    main()
