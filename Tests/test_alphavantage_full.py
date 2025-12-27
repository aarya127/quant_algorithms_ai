"""
Test Suite for Alpha Vantage API - Full Response Version
Tests all endpoints with NVDA stock and saves COMPLETE responses to JSON files
"""

import sys
import os
import datetime
import json
from time import sleep

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Data.alphavantage import AlphaVantage

# Test configuration
TEST_SYMBOL = "NVDA"
TEST_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'full_responses')
TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# Create output directory
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)


class AlphaVantageFullTestSuite:
    def __init__(self):
        self.av = AlphaVantage()
        self.test_count = 0
        self.passed = 0
        self.failed = 0
        
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
        
        # Small delay to respect API rate limits
        sleep(12)  # Alpha Vantage: 5 calls per minute = 12 seconds between calls
    
    def test_endpoint(self, name, func, *args, **kwargs):
        """Generic endpoint test wrapper"""
        print(f"Testing: {name}...")
        try:
            response = func(*args, **kwargs)
            
            if not response:
                self.save_response(name, response, 'FAIL', error="Empty response")
                return
            
            # Check for API errors
            if isinstance(response, dict):
                if 'Error Message' in response:
                    self.save_response(name, response, 'FAIL', error=response['Error Message'])
                    return
                if 'Note' in response:
                    self.save_response(name, response, 'FAIL', error=f"Rate limit: {response['Note']}")
                    return
            
            self.save_response(name, response, 'PASS')
            
        except Exception as e:
            self.save_response(name, None, 'FAIL', error=str(e))
    
    def run_all_tests(self):
        """Run all Alpha Vantage endpoint tests"""
        print("="*70)
        print(f"ALPHA VANTAGE API FULL RESPONSE TEST SUITE - {TEST_SYMBOL}")
        print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output Directory: {TEST_OUTPUT_DIR}")
        print("="*70)
        print()
        
        # News & Sentiment
        print("📰 NEWS & SENTIMENT TESTS")
        print("-"*70)
        self.test_endpoint(
            "get_market_news_sentiment",
            self.av.get_market_news_sentiment,
            tickers=TEST_SYMBOL,
            limit=50  # Get more news items
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
        print(f"All responses saved to: {TEST_OUTPUT_DIR}")
        print("="*70)
        print()


def main():
    """Run Alpha Vantage full response test suite"""
    suite = AlphaVantageFullTestSuite()
    
    try:
        suite.run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test suite error: {e}")


if __name__ == "__main__":
    main()
