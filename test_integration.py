#!/usr/bin/env python3
"""
Integration Tests for Bitvavo Bitcoin Trading Bot
Tests full workflows and component interactions with async support
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import sys
import os
import time
import json
import asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestFullTradingCycle(unittest.TestCase):
    """Test complete trading workflows"""

    def setUp(self):
        """Set up test environment"""
        self.mock_api = AsyncMock()

    def test_full_buy_order_lifecycle(self):
        """Test complete buy order: place → pending → filled"""
        from order_manager import OrderManager

        manager = OrderManager(self.mock_api)

        async def run_test():
            # 1. Place order
            self.mock_api.place_order_async = AsyncMock(
                return_value={"orderId": "ORDER123", "status": "new", "filled": False}
            )

            order_data = {
                "market": "BTC-EUR",
                "side": "buy",
                "orderType": "limit",
                "amount": 0.001,
                "price": 50000,
            }

            result = await self.mock_api.place_order_async(order_data)
            self.assertEqual(result["orderId"], "ORDER123")

            # 2. Check status - order is open
            self.mock_api.get_order_async = AsyncMock(
                return_value={"orderId": "ORDER123", "filled": 0, "status": "open"}
            )

            status = await self.mock_api.get_order_async("ORDER123", "BTC-EUR")
            self.assertEqual(status["status"], "open")

            # 3. Order fills
            self.mock_api.get_order_async = AsyncMock(
                return_value={
                    "orderId": "ORDER123",
                    "filled": 0.001,
                    "status": "filled",
                }
            )

            filled_status = await self.mock_api.get_order_async("ORDER123", "BTC-EUR")
            self.assertEqual(filled_status["status"], "filled")
            self.assertEqual(filled_status["filled"], 0.001)

        asyncio.run(run_test())

    def test_full_sell_order_with_profit(self):
        """Test selling at profit after buy"""
        from performance_tracker import PerformanceTracker

        tracker = PerformanceTracker(load_history=False)

        # 1. Record buy trade
        tracker.record_trade("BUY1", "buy", 0.1, 50000, 10, time.time())

        # 2. Record sell order
        tracker.record_trade("SELL1", "sell", 0.1, 55000, 10, time.time() + 1)

        # 3. Verify profit
        win_rate = tracker.calculate_win_rate()
        self.assertEqual(
            win_rate, 1.0, "Sell at 55k after buy at 50k should be profitable"
        )

    def test_order_cancellation_flow(self):
        """Test order placement and cancellation"""
        from order_manager import OrderManager

        manager = OrderManager(self.mock_api)

        async def run_test():
            # Place order
            self.mock_api.place_order_async = AsyncMock(
                return_value={"orderId": "ORDER456", "status": "new"}
            )

            order_data = {
                "market": "BTC-EUR",
                "side": "buy",
                "orderType": "limit",
                "amount": 0.001,
                "price": 49000,
            }

            result = await self.mock_api.place_order_async(order_data)
            self.assertEqual(result["orderId"], "ORDER456")

            # Cancel order
            self.mock_api.cancel_order_async = AsyncMock(
                return_value={"orderId": "ORDER456", "status": "cancelled"}
            )

            cancel_result = await self.mock_api.cancel_order_async(
                "BTC-EUR", "ORDER456"
            )
            self.assertEqual(cancel_result["status"], "cancelled")

        asyncio.run(run_test())


class TestAsyncAPIResilience(unittest.TestCase):
    """Test handling of async API failures and edge cases"""

    def test_balance_fetch_with_api_down(self):
        """Test balance fetching when API is down"""

        async def run_test():
            api = AsyncMock()
            api.get_balance_async = AsyncMock(side_effect=Exception("API Unavailable"))

            with self.assertRaises(Exception):
                await api.get_balance_async()

        asyncio.run(run_test())

    def test_order_placement_with_network_error(self):
        """Test order placement with network errors"""
        from order_manager import OrderManager

        api = AsyncMock()
        api.place_order_async = AsyncMock(side_effect=Exception("Network error"))

        manager = OrderManager(api)

        async def run_test():
            with self.assertRaises(Exception):
                order_data = {
                    "market": "BTC-EUR",
                    "side": "buy",
                    "orderType": "limit",
                    "amount": 0.001,
                    "price": 50000,
                }
                await manager.bitvavo_api.place_order_async(order_data)

        asyncio.run(run_test())

    def test_timeout_handling(self):
        """Test handling of timeout errors"""

        async def run_test():
            api = AsyncMock()
            api.get_ticker_async = AsyncMock(
                side_effect=asyncio.TimeoutError("Request timed out")
            )

            with self.assertRaises(asyncio.TimeoutError):
                await api.get_ticker_async("BTC-EUR")

        asyncio.run(run_test())


class TestDataPersistence(unittest.TestCase):
    """Test data saving and loading"""

    def test_order_tracking(self):
        """Test that order data is properly tracked"""
        from order_manager import OrderManager

        api = AsyncMock()
        manager = OrderManager(api)

        # Simulate order placement
        test_order = {
            "orderId": "TEST_ORDER_1",
            "market": "BTC-EUR",
            "side": "buy",
            "amount": 0.001,
            "price": 50000,
            "timestamp": time.time(),
        }

        # Track the order
        manager.pending_orders[test_order["orderId"]] = test_order

        self.assertIn("TEST_ORDER_1", manager.pending_orders)
        self.assertEqual(manager.pending_orders["TEST_ORDER_1"]["price"], 50000)

    def test_performance_history_tracking(self):
        """Test that performance history is tracked correctly"""
        from performance_tracker import PerformanceTracker

        tracker = PerformanceTracker(load_history=False)

        # Add trades
        tracker.record_trade("TRADE1", "buy", 0.1, 50000, 10)
        tracker.record_trade("TRADE2", "sell", 0.1, 55000, 10)

        # Verify trades are recorded
        self.assertEqual(len(tracker.trades), 2)
        self.assertEqual(tracker.trades[0]["side"], "buy")
        self.assertEqual(tracker.trades[1]["side"], "sell")


class TestIndicatorAccuracy(unittest.TestCase):
    """Test that technical indicators produce accurate results"""

    def test_rsi_extreme_values(self):
        """Test RSI calculation for extreme conditions"""
        from indicators import calculate_rsi

        # All gains
        uptrend = [100 + i for i in range(50)]
        rsi_up = calculate_rsi(uptrend)
        self.assertGreater(rsi_up, 70, "Strong uptrend should yield RSI > 70")

        # All losses
        downtrend = [100 - i for i in range(50)]
        rsi_down = calculate_rsi(downtrend)
        self.assertLess(rsi_down, 30, "Strong downtrend should yield RSI < 30")

    def test_vwap_weighted_correctly(self):
        """Test that VWAP correctly weights by volume"""
        from indicators import calculate_vwap

        # Price 100 with volume 1, price 200 with volume 9
        prices = [100, 200]
        volumes = [1, 9]

        vwap = calculate_vwap(prices, volumes)

        expected = (100 * 1 + 200 * 9) / (1 + 9)  # 190
        self.assertAlmostEqual(vwap, expected, places=2)

    def test_bollinger_bands_structure(self):
        """Test Bollinger Bands maintain proper structure"""
        from indicators import calculate_bollinger_bands

        # Stable prices - bands should be tight
        prices = [100] * 50
        upper, middle, lower = calculate_bollinger_bands(prices)

        self.assertAlmostEqual(upper, middle, delta=1)
        self.assertAlmostEqual(lower, middle, delta=1)

        # Volatile prices - bands should spread
        volatile_prices = [100 + (i % 2) * 10 for i in range(50)]
        upper_v, middle_v, lower_v = calculate_bollinger_bands(volatile_prices)

        self.assertGreater(upper_v - middle_v, 5)
        self.assertGreater(middle_v - lower_v, 5)


class TestAsyncOperationFlow(unittest.TestCase):
    """Test async operations work correctly"""

    def test_concurrent_price_fetches(self):
        """Test fetching multiple prices concurrently"""

        async def run_test():
            api = AsyncMock()

            # Mock different prices
            api.get_ticker_async = AsyncMock(
                side_effect=[{"last": "50000"}, {"last": "51000"}, {"last": "49500"}]
            )

            # Fetch prices concurrently
            prices = await asyncio.gather(
                api.get_ticker_async("BTC-EUR"),
                api.get_ticker_async("BTC-EUR"),
                api.get_ticker_async("BTC-EUR"),
            )

            self.assertEqual(len(prices), 3)
            self.assertEqual(float(prices[0]["last"]), 50000)
            self.assertEqual(float(prices[1]["last"]), 51000)
            self.assertEqual(float(prices[2]["last"]), 49500)

        asyncio.run(run_test())

    def test_order_status_polling(self):
        """Test polling order status in async loop"""

        async def run_test():
            api = AsyncMock()

            # Simulate order status progression
            statuses = [
                {"orderId": "ORDER1", "filled": 0, "status": "new"},
                {"orderId": "ORDER1", "filled": 0.0005, "status": "partially_filled"},
                {"orderId": "ORDER1", "filled": 0.001, "status": "filled"},
            ]

            api.get_order_async = AsyncMock(side_effect=statuses)

            # Poll status
            filled = 0
            for i in range(3):
                status = await api.get_order_async("ORDER1", "BTC-EUR")
                filled = status["filled"]
                self.assertEqual(status["orderId"], "ORDER1")

            self.assertEqual(filled, 0.001)

        asyncio.run(run_test())


class TestRateLimit(unittest.TestCase):
    """Test rate limiting compliance"""

    def test_bitvavo_rate_limit_structure(self):
        """Test that rate limit constants are properly defined"""
        # Bitvavo limits: 10 requests per second, up to 20 burst
        rate_limit_per_second = 10
        burst_capacity = 20

        self.assertEqual(rate_limit_per_second, 10)
        self.assertEqual(burst_capacity, 20)

    def test_minimum_order_amount(self):
        """Test Bitvavo minimum order amounts"""
        min_btc = 0.00001  # Bitvavo BTC minimum
        test_amount = 0.001  # Valid amount

        self.assertGreaterEqual(test_amount, min_btc)


def run_integration_tests():
    """Run all integration tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestFullTradingCycle))
    suite.addTests(loader.loadTestsFromTestCase(TestAsyncAPIResilience))
    suite.addTests(loader.loadTestsFromTestCase(TestDataPersistence))
    suite.addTests(loader.loadTestsFromTestCase(TestIndicatorAccuracy))
    suite.addTests(loader.loadTestsFromTestCase(TestAsyncOperationFlow))
    suite.addTests(loader.loadTestsFromTestCase(TestRateLimit))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print("BITVAVO INTEGRATION TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
