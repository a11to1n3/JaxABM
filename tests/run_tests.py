#!/usr/bin/env python
"""
Test runner for the JaxABM test suite.

This script runs both unit tests and integration tests for the JaxABM framework.
"""
import os
import sys
import argparse
import unittest
import pytest


def run_unittest_suite(pattern='test_*.py', verbose=False):
    """Run the unittest-based test suite."""
    print("Running unittest-based tests...")
    
    # Discover and run unit tests
    loader = unittest.TestLoader()
    
    # Run unit tests
    print("\nRunning unit tests:")
    unit_tests = loader.discover('tests/unit', pattern=pattern)
    unit_result = unittest.TextTestRunner(verbosity=2 if verbose else 1).run(unit_tests)
    
    # Run integration tests
    print("\nRunning integration tests:")
    integration_tests = loader.discover('tests/integration', pattern=pattern)
    integration_result = unittest.TextTestRunner(verbosity=2 if verbose else 1).run(integration_tests)
    
    # Return success if all tests passed
    return unit_result.wasSuccessful() and integration_result.wasSuccessful()


def run_pytest_suite(verbose=False):
    """Run the pytest-based test suite."""
    print("Running pytest-based tests...")
    
    # Build pytest arguments
    pytest_args = ['tests']
    if verbose:
        pytest_args.append('-v')
    
    # Run pytest and return result
    return pytest.main(pytest_args)


def main():
    """Main function to run the tests."""
    parser = argparse.ArgumentParser(description='Run JaxABM test suite')
    parser.add_argument('--unittest-only', action='store_true',
                        help='Run only unittest-based tests')
    parser.add_argument('--pytest-only', action='store_true',
                        help='Run only pytest-based tests')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--pattern', '-p', default='test_*.py',
                        help='Pattern for test files (default: test_*.py)')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*70)
    print("JaxABM Test Suite".center(70))
    print("="*70 + "\n")
    
    # Run the tests
    unittest_success = True
    pytest_success = True
    
    if not args.pytest_only:
        unittest_success = run_unittest_suite(pattern=args.pattern, verbose=args.verbose)
    
    if not args.unittest_only:
        pytest_result = run_pytest_suite(verbose=args.verbose)
        pytest_success = pytest_result == 0
    
    # Print summary
    print("\n" + "="*70)
    print("Test Results".center(70))
    print("="*70)
    
    if not args.pytest_only:
        print(f"Unittest: {'PASSED' if unittest_success else 'FAILED'}")
    
    if not args.unittest_only:
        print(f"Pytest: {'PASSED' if pytest_success else 'FAILED'}")
    
    # Exit with appropriate status code
    success = (args.pytest_only or unittest_success) and (args.unittest_only or pytest_success)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 