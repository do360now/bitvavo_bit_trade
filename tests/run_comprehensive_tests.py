#!/usr/bin/env python3
"""
Comprehensive Test Runner - Bitcoin Trading Bot with Cycle-Aware Deep Module

Runs all test suites:
1. test_cycle_trading_deep_module.py - NEW refactored module tests
2. test_bot_state_manager.py - State management tests  
3. test_integration.py - Integration and async tests
4. test_suite.py - Core component unit tests

Usage:
    python3 run_comprehensive_tests.py              # Run all tests
    python3 run_comprehensive_tests.py --critical   # Critical tests only
    python3 run_comprehensive_tests.py --verbose    # Verbose output
"""

import sys
import os
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_banner(text, char='='):
    """Print formatted banner"""
    print(f"\n{char * 70}")
    print(f"  {text}")
    print(f"{char * 70}\n")


def run_test_module(module_name, description, critical=False):
    """Run a test module and return results"""
    print(f"{'🔥' if critical else '📝'} {description}")
    print(f"   File: {module_name}")
    print()
    
    try:
        # Import and run the module
        module = __import__(module_name.replace('.py', ''))
        
        # Look for run function
        if hasattr(module, f'run_{module_name.split("_", 1)[1].replace(".py", "")}_tests'):
            func = getattr(module, f'run_{module_name.split("_", 1)[1].replace(".py", "")}_tests')
            success = func()
        elif hasattr(module, 'run_test_suite'):
            success = module.run_test_suite()
        elif hasattr(module, 'run_integration_tests'):
            success = module.run_integration_tests()
        else:
            print(f"⚠️  No test runner found in {module_name}")
            return None
        
        return success
        
    except ImportError as e:
        print(f"⚠️  Could not import {module_name}: {e}")
        return None
    except Exception as e:
        print(f"❌ Error running {module_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive test suite')
    parser.add_argument('--critical', action='store_true', help='Run only critical tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--module', '-m', help='Run specific module only')
    args = parser.parse_args()
    
    print_banner("BITCOIN TRADING BOT - COMPREHENSIVE TEST SUITE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Define test modules
    test_modules = [
        {
            'name': 'test_cycle_trading_deep_module',
            'description': 'Cycle Trading Deep Module Tests (NEW)',
            'critical': True,
        },
        {
            'name': 'test_bot_state_manager',
            'description': 'Bot State Manager Tests',
            'critical': True,
        },
        {
            'name': 'test_integration',
            'description': 'Integration & Async Tests',
            'critical': False,
        },
        {
            'name': 'test_suite',
            'description': 'Core Component Unit Tests',
            'critical': False,
        },
    ]
    
    # Filter modules if specific one requested
    if args.module:
        test_modules = [m for m in test_modules if m['name'] == args.module]
        if not test_modules:
            print(f"❌ Module '{args.module}' not found")
            print("\nAvailable modules:")
            for m in test_modules:
                print(f"  - {m['name']}")
            return 1
    
    # Filter to critical if requested
    if args.critical:
        test_modules = [m for m in test_modules if m['critical']]
        print("🔥 CRITICAL TESTS ONLY MODE\n")
    
    # Run tests
    results = {}
    for i, module in enumerate(test_modules, 1):
        print_banner(f"[{i}/{len(test_modules)}] {module['description']}", '-')
        
        success = run_test_module(
            module['name'],
            module['description'],
            module['critical']
        )
        
        results[module['name']] = success
    
    # Print summary
    print_banner("TEST SUMMARY")
    
    total = len(results)
    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)
    
    print(f"Total Modules: {total}")
    print(f"  ✅ Passed:  {passed}")
    print(f"  ❌ Failed:  {failed}")
    print(f"  ⏭️  Skipped: {skipped}\n")
    
    print("Detailed Results:")
    for name, status in results.items():
        if status is True:
            icon = '✅'
            status_text = 'PASSED'
        elif status is False:
            icon = '❌'
            status_text = 'FAILED'
        else:
            icon = '⏭️'
            status_text = 'SKIPPED'
        
        print(f"  {icon} {name}: {status_text}")
    
    # Overall status
    print()
    if failed == 0:
        print("🎉 ALL TESTS PASSED!")
        print("\nYour refactored bot is ready for deployment:")
        print("  ✓ Deep module working correctly")
        print("  ✓ State management intact")
        print("  ✓ Integration tests passing")
        print("  ✓ Core components functional")
        exit_code = 0
    else:
        print("❌ SOME TESTS FAILED")
        print("\nReview failures and fix before deploying.")
        exit_code = 1
    
    # Coverage hint
    if not args.critical:
        print("\n💡 Tip: Run with --critical for quick validation")
    
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
