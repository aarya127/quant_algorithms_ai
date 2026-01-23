"""
Test Suite for Finnhub API
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
from Data.finnhub import (
    get_company_news,
    get_basic_financials,
    get_earnings_surprises,
    get_insider_transactions,
    get_insider_sentiment,
    get_earnings_calendar,
    get_usa_spending
)

# Test configuration
TEST_SYMBOL = "NVDA"
TEST_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# Create output directory
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)


class FinnhubTestSuite:
    def __init__(self):
        self.results = []
        self.test_count = 0
        self.passed = 0
        self.failed = 0
        
        # Date ranges for testing
        self.today = datetime.date.today()
        self.from_date = (self.today - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
        self.to_date = self.today.strftime('%Y-%m-%d')
        
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
        
        # Small delay to be respectful to API
        sleep(1)
    
    def test_endpoint(self, name, func, *args, **kwargs):
        """Generic endpoint test wrapper"""
        print(f"Testing: {name}...")
        try:
            response = func(*args, **kwargs)
            
            # Check for empty response
            if response is None:
                self.log_result(name, 'FAIL', 'NoneType', 0, 
                              error="None response")
                return
            
            # Check for API errors (Finnhub specific)
            if isinstance(response, dict):
                if 'error' in response:
                    self.log_result(name, 'FAIL', 'dict', 0, 
                                  error=response['error'])
                    return
            
            # Count data items
            data_count = 0
            sample_data = None
            
            if isinstance(response, dict):
                # For financial data dict
                if 'metric' in response or 'series' in response:
                    data_count = len(response)
                    sample_data = {k: v for k, v in list(response.items())[:2]}
                else:
                    data_count = len(response)
                    if response:
                        first_key = list(response.keys())[0]
                        sample_data = {first_key: response[first_key]}
            elif isinstance(response, list):
                data_count = len(response)
                sample_data = response[0] if response else None
            
            # Consider test passed if we got data or valid empty response
            if data_count > 0 or response == {} or response == []:
                self.log_result(name, 'PASS', type(response).__name__, data_count, 
                              sample_data=sample_data)
            else:
                self.log_result(name, 'FAIL', type(response).__name__, 0,
                              error="Unexpected response format")
            
        except Exception as e:
            self.log_result(name, 'FAIL', 'exception', 0, error=str(e))
    
    def run_all_tests(self):
        """Run all Finnhub endpoint tests"""
        print("="*70)
        print(f"FINNHUB API TEST SUITE - {TEST_SYMBOL}")
        print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        print()
        
        # News
        print("📰 NEWS DATA")
        print("-"*70)
        self.test_endpoint(
            "get_company_news",
            get_company_news,
            TEST_SYMBOL,
            self.from_date,
            self.to_date
        )
        
        # Financials
        print("💰 FINANCIAL DATA")
        print("-"*70)
        self.test_endpoint(
            "get_basic_financials",
            get_basic_financials,
            TEST_SYMBOL,
            'all'
        )
        
        self.test_endpoint(
            "get_earnings_surprises",
            get_earnings_surprises,
            TEST_SYMBOL
        )
        
        self.test_endpoint(
            "get_earnings_calendar",
            get_earnings_calendar,
            self.from_date,
            self.to_date,
            TEST_SYMBOL
        )
        
        # Insider Data
        print("👔 INSIDER DATA")
        print("-"*70)
        self.test_endpoint(
            "get_insider_transactions",
            get_insider_transactions,
            TEST_SYMBOL,
            self.from_date,
            self.to_date
        )
        
        self.test_endpoint(
            "get_insider_sentiment",
            get_insider_sentiment,
            TEST_SYMBOL,
            self.from_date,
            self.to_date
        )
        
        # Government Data
        print("🏛️ GOVERNMENT DATA")
        print("-"*70)
        self.test_endpoint(
            "get_usa_spending",
            get_usa_spending,
            TEST_SYMBOL,
            self.from_date,
            self.to_date
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
            f'finnhub_test_results_{TEST_SYMBOL}_{TIMESTAMP}.csv'
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
            f'finnhub_summary_{TEST_SYMBOL}_{TIMESTAMP}.csv'
        )
        
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Test Symbol', TEST_SYMBOL])
            writer.writerow(['Total Tests', self.test_count])
            writer.writerow(['Passed', self.passed])
            writer.writerow(['Failed', self.failed])
            writer.writerow(['Success Rate', f"{self.passed/self.test_count*100:.1f}%"])
            writer.writerow(['Date Range', f"{self.from_date} to {self.to_date}"])
            writer.writerow(['Timestamp', TIMESTAMP])
        
        print(f"✓ Summary saved to: {summary_file}")
        print()


def main():
    """Run Finnhub test suite"""
    suite = FinnhubTestSuite()
    
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
