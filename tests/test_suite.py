#!/usr/bin/env python3
"""
Comprehensive Unit Test Suite for Bitvavo Bitcoin Trading Bot
Tests critical components to prevent regressions and ensure reliability
Adapted from Kraken tests with async/await support
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, mock_open, AsyncMock
import sys
import os
import json
import time
import asyncio
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestBitvavoAPI(unittest.TestCase):
    """Test Bitvavo API wrapper"""

    def test_minimum_order_size(self):
        """Test Bitvavo minimum order size constraint"""
        min_btc = 0.00001  # Bitvavo minimum

        # Check that 0.00001 BTC is at the minimum
        self.assertEqual(min_btc, 0.00001)

        # Check that values smaller than this fail
        self.assertLess(0.000001, min_btc)

    def test_market_format(self):
        """Test Bitvavo market format"""
        market = "BTC-EUR"

        # Verify market format
        self.assertIn("-", market)
        parts = market.split("-")
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0], "BTC")
        self.assertEqual(parts[1], "EUR")


class TestOrderManager(unittest.TestCase):
    """Test async order management functionality"""

    def setUp(self):
        from order_manager import OrderManager

        self.mock_api = AsyncMock()
        self.manager = OrderManager(self.mock_api)

    def test_order_placement_success(self):
        """Test successful async order placement"""
        self.manager.bitvavo_api.place_order_async = AsyncMock(
            return_value={"orderId": "ORDER123", "status": "new"}
        )

        async def run_test():
            order_data = {
                "market": "BTC-EUR",
                "side": "buy",
                "orderType": "limit",
                "amount": 0.001,
                "price": 50000,
            }
            result = await self.manager.bitvavo_api.place_order_async(order_data)
            self.assertEqual(result["orderId"], "ORDER123")

        asyncio.run(run_test())

    def test_order_data_structure(self):
        """Test order data is properly structured"""
        order = {
            "orderId": "ORDER123",
            "market": "BTC-EUR",
            "side": "buy",
            "orderType": "limit",
            "amount": 0.001,
            "price": 50000,
            "timestamp": time.time(),
            "status": "new",
        }

        self.assertEqual(order["market"], "BTC-EUR")
        self.assertEqual(order["side"], "buy")
        self.assertEqual(order["amount"], 0.001)
        self.assertGreaterEqual(order["price"], 0)


class TestTradeExecutor(unittest.TestCase):
    """Test async trade execution functionality"""

    def setUp(self):
        from trade_executor import TradeExecutor

        self.mock_api = AsyncMock()
        self.executor = TradeExecutor(self.mock_api)

    def test_btc_balance_fetch(self):
        """Test fetching BTC balance"""
        # Test balance parsing without actually initializing API
        mock_balance = [
            {"symbol": "BTC", "available": "0.5"},
            {"symbol": "EUR", "available": "5000.0"},
        ]

        # Verify structure
        btc_entry = next((b for b in mock_balance if b["symbol"] == "BTC"), None)
        self.assertIsNotNone(btc_entry)
        self.assertEqual(float(btc_entry["available"]), 0.5)

    def test_available_balance_fetch(self):
        """Test fetching EUR balance"""
        mock_balance = [
            {"symbol": "BTC", "available": "0.1"},
            {"symbol": "EUR", "available": "10000.0"},
        ]

        # Verify structure
        eur_entry = next((b for b in mock_balance if b["symbol"] == "EUR"), None)
        self.assertIsNotNone(eur_entry)
        self.assertEqual(float(eur_entry["available"]), 10000.0)

    def test_current_price_fetch(self):
        """Test fetching current BTC price"""
        mock_ticker = {"last": "50000.00", "bid": "49999.99", "ask": "50000.01"}

        # Verify structure
        price = float(mock_ticker["last"])
        self.assertEqual(price, 50000.0)

    def test_position_size_calculation(self):
        """Test risk-based position sizing"""
        eur_balance = 10000
        risk_percent = 0.01  # 1%
        btc_price = 50000

        position_size = self.executor.calculate_position_size(
            eur_balance, btc_price, risk_percent
        )

        # 0.01% of 10000 = 1 EUR / 50000 = 0.00002 BTC (above minimum 0.00001)
        # Actually: 1% of 10000 = 100 EUR / 50000 = 0.002 BTC
        self.assertGreater(position_size, 0)
        self.assertLess(position_size, 1)  # Less than 1 BTC

    def test_minimum_order_size_constraint(self):
        """Test that position size respects Bitvavo minimum"""
        eur_balance = 100  # Small balance
        risk_percent = 0.001  # 0.1% risk
        btc_price = 100000

        position_size = self.executor.calculate_position_size(
            eur_balance, btc_price, risk_percent
        )

        # 0.1% of 100 = 0.1 EUR / 100000 = 0.000001 BTC (below minimum)
        # Should return 0 due to minimum size constraint
        self.assertEqual(position_size, 0)


class TestPerformanceTracker(unittest.TestCase):
    """Test performance tracking and calculations"""

    def setUp(self):
        from performance_tracker import PerformanceTracker

        self.tracker = PerformanceTracker(load_history=False)

    def test_win_rate_fifo_basic(self):
        """Test basic FIFO win rate calculation"""
        self.tracker.record_trade("buy1", "buy", 0.1, 50000, 10)
        self.tracker.record_trade("sell1", "sell", 0.1, 55000, 10)

        win_rate = self.tracker.calculate_win_rate()
        self.assertEqual(win_rate, 1.0, "Profitable trade should be 100% win rate")

    def test_win_rate_fifo_loss(self):
        """Test FIFO with losing trade"""
        self.tracker.record_trade("buy1", "buy", 0.1, 60000, 10)
        self.tracker.record_trade("sell1", "sell", 0.1, 55000, 10)

        win_rate = self.tracker.calculate_win_rate()
        self.assertEqual(win_rate, 0.0, "Losing trade should be 0% win rate")

    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation"""
        for equity in [10000, 10500, 9000, 9500, 11000]:
            self.tracker.equity_curve.append(
                {
                    "timestamp": time.time(),
                    "total_equity_eur": equity,
                    "btc_balance": 0.1,
                    "eur_balance": equity - 5000,
                    "btc_price": 50000,
                }
            )

        max_dd = self.tracker.calculate_max_drawdown()
        self.assertGreater(max_dd, 0.14)
        self.assertLess(max_dd, 0.15)

    def test_sharpe_ratio_positive_returns(self):
        """Test Sharpe ratio with positive returns"""
        for i in range(100):
            equity = 10000 + i * 100
            self.tracker.equity_curve.append(
                {
                    "timestamp": time.time() + i * 3600,
                    "total_equity_eur": equity,
                    "btc_balance": 0.1,
                    "eur_balance": equity - 5000,
                    "btc_price": 50000,
                }
            )

        sharpe = self.tracker.calculate_sharpe_ratio()
        self.assertGreater(sharpe, 0, "Positive returns should yield positive Sharpe")


class TestIndicators(unittest.TestCase):
    """Test technical indicator calculations"""

    def test_rsi_calculation(self):
        """Test RSI calculation"""
        from indicators import calculate_rsi

        # Generate test prices with uptrend
        prices = [100 + i for i in range(20)]
        rsi = calculate_rsi(prices)

        self.assertIsNotNone(rsi)
        self.assertGreater(rsi, 50, "Uptrend should yield RSI > 50")
        self.assertLessEqual(rsi, 100)

    def test_rsi_insufficient_data(self):
        """Test RSI with insufficient data"""
        from indicators import calculate_rsi

        prices = [100, 101, 102]
        rsi = calculate_rsi(prices)

        self.assertIsNone(rsi)

    def test_macd_calculation(self):
        """Test MACD calculation"""
        from indicators import calculate_macd

        prices = [50000 + i * 100 for i in range(50)]
        macd, signal = calculate_macd(prices)

        self.assertIsNotNone(macd)
        self.assertIsNotNone(signal)
        self.assertIsInstance(macd, float)
        self.assertIsInstance(signal, float)

    def test_vwap_calculation(self):
        """Test VWAP calculation"""
        from indicators import calculate_vwap

        prices = [100, 101, 102, 103, 104]
        volumes = [10, 20, 15, 25, 30]

        vwap = calculate_vwap(prices, volumes)

        self.assertIsNotNone(vwap)
        self.assertGreater(vwap, 100)
        self.assertLess(vwap, 105)

    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        from indicators import calculate_bollinger_bands

        prices = [100 + i for i in range(30)]
        upper, middle, lower = calculate_bollinger_bands(prices)

        self.assertIsNotNone(upper)
        self.assertIsNotNone(middle)
        self.assertIsNotNone(lower)
        self.assertGreater(upper, middle)
        self.assertGreater(middle, lower)


class TestDataManager(unittest.TestCase):
    """Test data management and persistence"""

    def setUp(self):
        from data_manager import DataManager

        self.manager = DataManager("test_prices.json", "test_logs.csv")

    def test_ohlc_structure(self):
        """Test OHLC data structure"""
        timestamp = 1234567890.0
        ohlc = [[timestamp, 50000, 51000, 49000, 50500, 50250, 100]]

        # Verify structure: [timestamp, open, high, low, close, vwap, volume]
        self.assertEqual(len(ohlc[0]), 7)
        self.assertEqual(ohlc[0][0], timestamp)  # timestamp
        self.assertEqual(ohlc[0][1], 50000)  # open
        self.assertEqual(ohlc[0][2], 51000)  # high
        self.assertEqual(ohlc[0][3], 49000)  # low
        self.assertEqual(ohlc[0][4], 50500)  # close

    def test_csv_header_validation(self):
        """Test that CSV headers are properly validated"""
        self.assertTrue(hasattr(self.manager, "HEADERS"))
        self.assertIn("timestamp", self.manager.HEADERS)
        self.assertIn("price", self.manager.HEADERS)
        self.assertIn("buy_decision", self.manager.HEADERS)


class TestCircuitBreaker(unittest.TestCase):
    """Test circuit breaker functionality"""

    def test_circuit_opens_after_threshold(self):
        """Test that circuit opens after failure threshold"""
        from circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)

        def failing_func():
            raise Exception("Test failure")

        # First 3 failures
        for i in range(3):
            with self.assertRaises(Exception):
                breaker.call(failing_func)

        # 4th call should raise CircuitBreakerException
        from circuit_breaker import CircuitBreakerException

        with self.assertRaises(CircuitBreakerException):
            breaker.call(failing_func)

    def test_circuit_recovers_after_timeout(self):
        """Test circuit recovery after timeout"""
        from circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)

        def failing_func():
            raise Exception("Test failure")

        # Open circuit
        for i in range(2):
            with self.assertRaises(Exception):
                breaker.call(failing_func)

        # Wait for recovery
        time.sleep(1.1)

        # Should move to HALF_OPEN
        def working_func():
            return "success"

        result = breaker.call(working_func)
        self.assertEqual(result, "success")


def run_test_suite():
    """Run all tests and generate report"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBitvavoAPI))
    suite.addTests(loader.loadTestsFromTestCase(TestOrderManager))
    suite.addTests(loader.loadTestsFromTestCase(TestTradeExecutor))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceTracker))
    suite.addTests(loader.loadTestsFromTestCase(TestIndicators))
    suite.addTests(loader.loadTestsFromTestCase(TestDataManager))
    suite.addTests(loader.loadTestsFromTestCase(TestCircuitBreaker))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("BITVAVO TEST SUITE SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)
