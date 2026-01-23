"""
Test Suite for Finnhub API - Full Response Version
Tests all endpoints with NVDA stock and saves COMPLETE responses to JSON files
"""

import sys
import os
import datetime
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
TEST_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'full_responses')
TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# Create output directory
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)


class FinnhubFullTestSuite:
    def __init__(self):
        self.test_count = 0
        self.passed = 0
        self.failed = 0
        
        # Date ranges for testing
        self.today = datetime.date.today()
        self.from_date = (self.today - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
        self.to_date = self.today.strftime('%Y-%m-%d')
        
    def save_response(self, endpoint_name, response, status, error=None):
        """Save full response to JSON file"""
        self.test_count += 1
        
        filename = f"{self.test_count:02d}_{endpoint_name}_{TEST_SYMBOL}_{TIMESTAMP}.json"
        filepath = os.path.join(TEST_OUTPUT_DIR, filename)
        
        output = {
            'test_number': self.test_count,
            'timestamp': datetime.datetime.now().isoformat(),
            'endpoint': endpoint_name,
            'symbol': TEST_SYMBOL,
            'date_range': {
                'from': self.from_date,
                'to': self.to_date
            },
            'status': status,
            'error': error,
            'response': response
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        if status == 'PASS':
            self.passed += 1
            print(f"✓ Test {self.test_count}: {endpoint_name} - PASSED")
            print(f"  Response saved to: {filename}")
        else:
            self.failed += 1
            print(f"✗ Test {self.test_count}: {endpoint_name} - FAILED")
            if error:
                print(f"  Error: {error}")
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
                self.save_response(name, response, 'FAIL', error="None response")
                return
            
            # Check for API errors (Finnhub specific)
            if isinstance(response, dict):
                if 'error' in response:
                    self.save_response(name, response, 'FAIL', error=response['error'])
                    return
            
            # Consider test passed if we got a valid response (even if empty list/dict)
            self.save_response(name, response, 'PASS')
            
        except Exception as e:
            self.save_response(name, None, 'FAIL', error=str(e))
    
    def run_all_tests(self):
        """Run all Finnhub endpoint tests"""
        print("="*70)
        print(f"FINNHUB API FULL RESPONSE TEST SUITE - {TEST_SYMBOL}")
        print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output Directory: {TEST_OUTPUT_DIR}")
        print(f"Date Range: {self.from_date} to {self.to_date}")
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
        print(f"All responses saved to: {TEST_OUTPUT_DIR}")
        print("="*70)
        print()


def main():
    """Run Finnhub full response test suite"""
    suite = FinnhubFullTestSuite()
    
    try:
        suite.run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test suite error: {e}")


if __name__ == "__main__":
    main()
