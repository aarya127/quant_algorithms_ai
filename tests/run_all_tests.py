"""
Master Test Runner
Runs both Alpha Vantage and Finnhub test suites
"""

import sys
import os
import subprocess
import datetime

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def run_test_file(test_file, api_name):
    """Run a test file and capture output"""
    print("="*70)
    print(f"RUNNING {api_name} TESTS")
    print("="*70)
    print()
    
    test_path = os.path.join(os.path.dirname(__file__), test_file)
    
    try:
        # Run the test using the virtual environment's Python
        venv_python = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            '.venv', 
            'bin', 
            'python'
        )
        
        result = subprocess.run(
            [venv_python, test_path],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__)
        )
        
        # Print output
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running {api_name} tests: {e}")
        return False


def main():
    """Run all test suites"""
    print("\n" + "="*70)
    print("API TEST SUITE MASTER RUNNER")
    print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print()
    
    results = {}
    
    # Run Finnhub tests first (faster, no rate limits)
    print("🔹 Starting Finnhub tests...")
    results['Finnhub'] = run_test_file('test_finnhub.py', 'FINNHUB')
    
    print("\n" + "="*70)
    print("⏳ Waiting 15 seconds before Alpha Vantage tests...")
    print("="*70)
    import time
    time.sleep(15)
    
    # Run Alpha Vantage tests (slower due to rate limits)
    print("\n🔹 Starting Alpha Vantage tests...")
    results['Alpha Vantage'] = run_test_file('test_alphavantage.py', 'ALPHA VANTAGE')
    
    # Final summary
    print("\n" + "="*70)
    print("OVERALL TEST SUMMARY")
    print("="*70)
    for api, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{api}: {status}")
    print("="*70)
    print()
    
    # Check results directory
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    if os.path.exists(results_dir):
        csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
        print(f"📊 Generated {len(csv_files)} CSV files in Tests/results/")
        for f in sorted(csv_files)[-4:]:  # Show last 4 files
            print(f"   - {f}")
    print()


if __name__ == "__main__":
    main()
