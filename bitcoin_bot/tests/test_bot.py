# test_bot.py
import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
import os
import tempfile
import time

# Import modules to test
from trading_bot import TradingBot
from data_manager import DataManager
from trade_executor import TradeExecutor
from indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands
from enhanced_trading_bot import EnhancedTradingBot
from peak_avoidance_system import PeakAvoidanceSystem
from order_manager import OrderManager


class TestDataManager(unittest.TestCase):
    """Test DataManager functionality"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.price_file = os.path.join(self.temp_dir, "test_prices.json")
        self.log_file = os.path.join(self.temp_dir, "test_logs.csv")
        self.data_manager = DataManager(self.price_file, self.log_file)

    def test_initialization(self):
        """Test DataManager initialization"""
        self.assertTrue(os.path.exists(self.price_file))
        self.assertTrue(os.path.exists(self.log_file))

    def test_append_ohlc_data(self):
        """Test OHLC data appending"""
        test_ohlc = [
            [1609459200, 50000, 51000, 49000, 50500, 100],  # Valid candle
            [1609459260, 50500, 51500, 50000, 51000, 150],
        ]

        added = self.data_manager.append_ohlc_data(test_ohlc)
        self.assertEqual(added, 2)

        # Verify data was saved
        prices, volumes = self.data_manager.load_price_history()
        self.assertEqual(len(prices), 2)
        self.assertAlmostEqual(prices[0], 50500)

    def test_duplicate_prevention(self):
        """Test that duplicate timestamps are not added"""
        test_ohlc = [
            [1609459200, 50000, 51000, 49000, 50500, 100],
        ]

        # Add twice
        self.data_manager.append_ohlc_data(test_ohlc)
        self.data_manager.append_ohlc_data(test_ohlc)

        prices, _ = self.data_manager.load_price_history()
        self.assertEqual(len(prices), 1)  # Should only have one entry

    def test_log_strategy(self):
        """Test strategy logging"""
        self.data_manager.log_strategy(
            price=50000, trade_volume=0.01, side="buy", reason="Test buy", rsi=45
        )

        # Verify CSV was written
        df = pd.read_csv(self.log_file)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["side"], "buy")


class TestIndicators(unittest.TestCase):
    """Test technical indicators"""

    def test_rsi_calculation(self):
        """Test RSI calculation"""
        # Generate test prices
        prices = [100 + i for i in range(20)]  # Uptrend
        rsi = calculate_rsi(prices)

        self.assertIsNotNone(rsi)
        self.assertGreater(rsi, 50)  # Should be > 50 in uptrend
        self.assertLessEqual(rsi, 100)

    def test_rsi_insufficient_data(self):
        """Test RSI with insufficient data"""
        prices = [100, 101, 102]  # Too few
        rsi = calculate_rsi(prices)
        self.assertIsNone(rsi)

    def test_macd_calculation(self):
        """Test MACD calculation"""
        prices = [100 + np.sin(i / 5) * 10 for i in range(50)]
        macd, signal = calculate_macd(prices)

        self.assertIsNotNone(macd)
        self.assertIsNotNone(signal)
        self.assertIsInstance(macd, float)
        self.assertIsInstance(signal, float)

    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        prices = [100] * 20  # Flat prices
        upper, middle, lower = calculate_bollinger_bands(prices)

        self.assertIsNotNone(upper)
        self.assertAlmostEqual(middle, 100)
        self.assertGreater(upper, middle)
        self.assertLess(lower, middle)


class TestTradeExecutor(unittest.TestCase):
    """Test TradeExecutor functionality"""

    def setUp(self):
        self.mock_api = Mock()
        self.executor = TradeExecutor(self.mock_api)

    def test_fetch_current_price(self):
        """Test price fetching"""
        self.mock_api.fetch_ohlcv.return_value = [
            [1609459200, 50000, 51000, 49000, 50500, 100]
        ]

        price, volume = self.executor.fetch_current_price()

        self.assertEqual(price, 50500)
        self.assertEqual(volume, 100)

    def test_fetch_price_error_handling(self):
        """Test price fetch error handling"""
        self.mock_api.fetch_ohlcv.side_effect = Exception("API Error")

        price, volume = self.executor.fetch_current_price()

        self.assertIsNone(price)
        self.assertEqual(volume, 0.0)

    def test_get_optimal_buy_price(self):
        """Test optimal buy price calculation"""
        order_book = {
            "asks": [[50000, 1], [50100, 2]],
            "bids": [[49900, 1], [49800, 2]],
        }

        price = self.executor.get_optimal_price(order_book, "buy")

        self.assertIsNotNone(price)
        self.assertLess(price, 50000)  # Should be slightly below best ask

    def test_execute_trade_validation(self):
        """Test trade execution with validation"""
        self.mock_api.load_markets.return_value = {
            "BTC/EUR": {"limits": {"amount": {"min": 0.0001}, "cost": {"min": 10}}}
        }

        # Test volume too small
        self.mock_api.create_limit_order.return_value = {"id": "test123"}

        success = self.executor.execute_trade(0.00001, "buy", 50000)

        self.assertFalse(success)  # Should fail due to min amount


class TestOrderManager(unittest.TestCase):
    """Test OrderManager functionality"""

    def setUp(self):
        self.mock_api = Mock()
        self.order_manager = OrderManager(self.mock_api)

    def test_place_order(self):
        """Test order placement"""
        self.mock_api.create_limit_order.return_value = {
            "id": "order123",
            "status": "open",
            "filled": 0,
        }

        order_id = self.order_manager.place_limit_order_with_timeout(
            volume=0.01, side="buy", price=50000, timeout=300
        )

        self.assertEqual(order_id, "order123")
        self.assertIn("order123", self.order_manager.pending_orders)

    def test_order_timeout(self):
        """Test order timeout handling"""
        # Place an order with 1 second timeout
        self.mock_api.create_limit_order.return_value = {
            "id": "timeout_order",
            "status": "open",
            "filled": 0,
        }

        self.order_manager.place_limit_order_with_timeout(
            volume=0.01, side="buy", price=50000, timeout=1
        )

        # Mock cancel order
        self.mock_api.cancel_order.return_value = True
        self.mock_api.fetch_order.return_value = {"status": "cancelled", "filled": 0}

        # Wait for timeout
        time.sleep(1.5)

        # Check orders
        results = self.order_manager.check_and_update_orders()

        self.assertIn("timeout_order", results["cancelled"])


class TestEnhancedTradingBot(unittest.TestCase):
    """Test ML-enhanced trading bot"""

    def setUp(self):
        self.mock_data_manager = Mock()
        self.mock_trade_executor = Mock()
        self.mock_onchain = Mock()
        self.mock_order_manager = Mock()

        # Create basic bot
        self.basic_bot = Mock()
        self.basic_bot.data_manager = self.mock_data_manager
        self.basic_bot.trade_executor = self.mock_trade_executor

        # Mock bot logs for ML training
        self.mock_data_manager.bot_logs_file = "test_logs.csv"

        with patch("os.path.exists", return_value=False):
            self.enhanced_bot = EnhancedTradingBot(self.basic_bot)

    def test_feature_extraction(self):
        """Test ML feature extraction"""
        indicators = {
            "rsi": 45,
            "macd": 100,
            "signal": 50,
            "current_price": 50000,
            "upper_band": 51000,
            "lower_band": 49000,
            "sentiment": 0.1,
            "news_analysis": {"risk_off_probability": 0.3},
        }

        features = self.enhanced_bot.learning_engine.extract_features(indicators)

        self.assertEqual(len(features), 12)  # Should have 12 features
        self.assertTrue(all(0 <= f <= 1 or -1 <= f <= 1 for f in features))

    def test_enhanced_signal_generation(self):
        """Test enhanced signal generation"""
        indicators = {
            "rsi": 30,  # Oversold
            "macd": 100,
            "signal": 50,
            "current_price": 50000,
            "sentiment": 0.2,
            "news_analysis": {"risk_off_probability": 0.2},
        }

        # Mock price history
        self.mock_data_manager.load_price_history.return_value = (
            [50000] * 100,
            [100] * 100,
        )

        signal = self.enhanced_bot.generate_enhanced_signal(indicators)

        self.assertIn(signal.action, ["buy", "sell", "hold"])
        self.assertGreaterEqual(signal.confidence, 0)
        self.assertLessEqual(signal.confidence, 1)


class TestPeakAvoidanceSystem(unittest.TestCase):
    """Test peak avoidance system"""

    def setUp(self):
        self.peak_system = PeakAvoidanceSystem(lookback_days=30)

    def test_peak_detection(self):
        """Test peak detection in price data"""
        # Create price data with clear peak
        prices = [100] * 10 + [110, 120, 130, 120, 110] + [100] * 10

        peaks = self.peak_system._detect_peaks_multi_method(np.array(prices))

        self.assertIn(12, peaks)  # Should detect peak at index 12 (130)

    def test_peak_probability_calculation(self):
        """Test peak probability calculation"""
        current_indicators = {
            "rsi": 75,  # Overbought
            "macd": 100,
            "volume_ratio": 2.5,  # High volume
        }

        recent_prices = [100] * 20 + [105, 110]  # Rising prices
        recent_volumes = [100] * 22

        result = self.peak_system.predict_peak_probability(
            110, current_indicators, recent_prices, recent_volumes
        )

        self.assertIn("peak_probability", result)
        self.assertIn("recommended_action", result)
        self.assertGreaterEqual(result["peak_probability"], 0)
        self.assertLessEqual(result["peak_probability"], 1)


class TestRiskManagement(unittest.TestCase):
    """Test risk management functions"""

    def setUp(self):
        self.mock_data_manager = Mock()
        self.mock_trade_executor = Mock()
        self.mock_onchain = Mock()
        self.mock_order_manager = Mock()

        self.bot = TradingBot(
            self.mock_data_manager,
            self.mock_trade_executor,
            self.mock_onchain,
            self.mock_order_manager,
        )

    def test_risk_adjusted_position_sizing(self):
        """Test position sizing with risk adjustment"""
        indicators = {
            "news_analysis": {"risk_off_probability": 0.7},  # High risk
            "volatility": 0.06,  # High volatility
            "rsi": 45,
            "current_price": 50000,
            "performance_report": {"risk_metrics": {"win_rate": "30%"}},
        }

        position = self.bot.calculate_risk_adjusted_position_size(
            "buy", indicators, btc_balance=1.0, eur_balance=10000
        )

        # Should reduce position size due to high risk
        self.assertLess(position, 0.08)  # Less than base 8%

    def test_emergency_sell_conditions(self):
        """Test emergency sell trigger"""
        indicators = {
            "news_analysis": {"risk_off_probability": 0.85},  # Extreme risk
            "sentiment": -0.2,
            "current_price": 50000,
            "avg_buy_price": 48000,
            "rsi": 50,
            "volatility": 0.05,
            "vwap": 50000,
            "netflow": 0,
            "macd": 0,
            "signal": 0,
            "market_trend": "ranging",
            "performance_report": {
                "risk_metrics": {"win_rate": "50%"},
                "returns": {"total": "5%"},
            },
        }

        action = self.bot.enhanced_decide_action_with_risk_override(indicators)

        self.assertEqual(action, "sell")  # Should trigger emergency sell


class TestIntegration(unittest.TestCase):
    """Integration tests for complete flow"""

    @patch("trade_executor.TradeExecutor.fetch_current_price")
    @patch("trade_executor.TradeExecutor.get_total_btc_balance")
    @patch("trade_executor.TradeExecutor.get_available_balance")
    def test_complete_trading_cycle(self, mock_eur, mock_btc, mock_price):
        """Test complete trading cycle"""
        # Setup mocks
        mock_price.return_value = (50000, 100)
        mock_btc.return_value = 0.5
        mock_eur.return_value = 5000

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create bot with real components
            data_manager = DataManager(
                os.path.join(tmpdir, "prices.json"), os.path.join(tmpdir, "logs.csv")
            )

            mock_api = Mock()
            mock_api.fetch_ohlcv.return_value = [
                [time.time() - i * 900, 50000, 51000, 49000, 50500, 100]
                for i in range(100)
            ]

            trade_executor = TradeExecutor(mock_api)
            mock_onchain = Mock()
            mock_onchain.get_onchain_signals.return_value = {
                "fee_rate": 10,
                "netflow": -5000,
                "volume": 10000,
                "old_utxos": 5,
            }

            order_manager = OrderManager(mock_api)

            bot = TradingBot(data_manager, trade_executor, mock_onchain, order_manager)

            # Mock successful order placement
            mock_api.create_limit_order.return_value = {
                "id": "test_order",
                "status": "open",
            }

            # Execute strategy
            with patch("indicators.fetch_enhanced_news", return_value=[]):
                with patch(
                    "indicators.calculate_enhanced_sentiment",
                    return_value={"sentiment": 0.1, "risk_off_probability": 0.2},
                ):
                    bot.execute_strategy()

            # Verify execution
            self.assertTrue(
                mock_api.create_limit_order.called or mock_api.fetch_order_book.called
            )


class TestSecurityAndErrorHandling(unittest.TestCase):
    """Test security and error handling"""

    def test_no_hardcoded_credentials(self):
        """Ensure no hardcoded API keys"""
        # Check config.py uses environment variables
        with open("config.py", "r") as f:
            content = f.read()
            self.assertIn("os.getenv", content)
            self.assertNotIn('api_key = "', content.lower())
            self.assertNotIn('api_secret = "', content.lower())

    def test_api_error_handling(self):
        """Test API error handling"""
        mock_api = Mock()
        mock_api.fetch_balance.side_effect = Exception("API Error")

        executor = TradeExecutor(mock_api)
        balance = executor.get_total_btc_balance()

        self.assertIsNone(balance)  # Should return None, not crash

    def test_file_corruption_handling(self):
        """Test handling of corrupted data files"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("corrupted json {[}")
            temp_file = f.name

        try:
            data_manager = DataManager(temp_file, "test.csv")
            prices, volumes = data_manager.load_price_history()

            self.assertEqual(prices, [])  # Should return empty, not crash
        finally:
            os.unlink(temp_file)


class TestPerformance(unittest.TestCase):
    """Performance and efficiency tests"""

    def test_indicator_calculation_performance(self):
        """Test indicator calculation speed"""
        import timeit

        prices = [50000 + np.random.randn() * 1000 for _ in range(1000)]

        rsi_time = timeit.timeit(lambda: calculate_rsi(prices), number=100)

        self.assertLess(rsi_time / 100, 0.01)  # Should take < 10ms per calculation

    def test_memory_usage_price_history(self):
        """Test memory efficiency of price history"""

        data_manager = DataManager("test.json", "test.csv")

        # Add 10000 candles
        large_ohlc = [
            [1609459200 + i * 60, 50000, 51000, 49000, 50500, 100] for i in range(10000)
        ]

        data_manager.append_ohlc_data(large_ohlc)

        # Load and check memory
        prices, volumes = data_manager.load_price_history()

        # Should maintain only last 10000 candles
        self.assertLessEqual(len(prices), 10000)


def run_tests():
    """Run all tests with coverage report"""
    import coverage

    # Start coverage
    cov = coverage.Coverage()
    cov.start()

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataManager))
    suite.addTests(loader.loadTestsFromTestCase(TestIndicators))
    suite.addTests(loader.loadTestsFromTestCase(TestTradeExecutor))
    suite.addTests(loader.loadTestsFromTestCase(TestOrderManager))
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedTradingBot))
    suite.addTests(loader.loadTestsFromTestCase(TestPeakAvoidanceSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestRiskManagement))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestSecurityAndErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Stop coverage and print report
    cov.stop()
    cov.save()

    print("\n" + "=" * 60)
    print("COVERAGE REPORT")
    print("=" * 60)
    cov.report()

    return result


if __name__ == "__main__":
    run_tests()
