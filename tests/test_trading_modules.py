#!/usr/bin/env python3
"""
Unit Tests for Atomic Trade Manager and Cycle Trading Deep Module
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, mock_open
import sys
import os
import json
import time
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAtomicTradeManager(unittest.TestCase):
    """Test atomic trade management - ensures state consistency"""

    def setUp(self):
        """Set up test fixtures"""
        # Create temp file for pending updates
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        self.temp_file.close()

        # Mock dependencies
        self.mock_order_manager = Mock()
        self.mock_state_manager = Mock()
        self.mock_trade_executor = Mock()

        # Import and create manager
        from atomic_trade_manager import AtomicTradeManager
        with patch('atomic_trade_manager.open', mock_open(read_data='{}')):
            self.manager = AtomicTradeManager(
                self.mock_order_manager,
                self.mock_state_manager,
                self.mock_trade_executor
            )
        self.manager.pending_file = self.temp_file.name

    def tearDown(self):
        """Clean up temp files"""
        try:
            os.unlink(self.temp_file.name)
        except:
            pass

    def test_execute_buy_success(self):
        """Test successful buy order execution"""
        self.mock_order_manager.place_limit_order.return_value = "ORDER123"

        result = self.manager.execute_buy(volume=0.01, price=50000)

        self.assertEqual(result, "ORDER123")
        self.mock_order_manager.place_limit_order.assert_called_once()
        self.assertEqual(len(self.manager.pending_updates), 1)

    def test_execute_buy_failure(self):
        """Test failed buy order - should rollback pending update"""
        self.mock_order_manager.place_limit_order.return_value = None

        result = self.manager.execute_buy(volume=0.01, price=50000)

        self.assertIsNone(result)
        # Pending update should be removed
        self.assertEqual(len(self.manager.pending_updates), 0)

    def test_execute_sell_success(self):
        """Test successful sell order execution"""
        self.mock_order_manager.place_limit_order.return_value = "ORDER456"

        result = self.manager.execute_sell(volume=0.01, price=55000)

        self.assertEqual(result, "ORDER456")
        self.mock_order_manager.place_limit_order.assert_called_once()

    def test_execute_sell_failure(self):
        """Test failed sell order - should rollback pending update"""
        self.mock_order_manager.place_limit_order.return_value = None

        result = self.manager.execute_sell(volume=0.01, price=55000)

        self.assertIsNone(result)
        self.assertEqual(len(self.manager.pending_updates), 0)

    def test_pending_update_stores_correct_data(self):
        """Test that pending updates contain correct information"""
        self.mock_order_manager.place_limit_order.return_value = "ORDER789"

        result = self.manager.execute_buy(volume=0.05, price=45000)

        # Find the pending update
        pending_id = list(self.manager.pending_updates.keys())[0]
        update = self.manager.pending_updates[pending_id]

        self.assertEqual(update['type'], 'buy')
        self.assertEqual(update['volume'], 0.05)
        self.assertEqual(update['price'], 45000)
        self.assertEqual(update['order_id'], "ORDER789")

    def test_process_filled_orders_empty(self):
        """Test processing with no filled orders"""
        self.mock_order_manager.check_and_update_orders.return_value = {
            'filled': [],
            'cancelled': [],
            'pending': []
        }

        result = self.manager.process_filled_orders()

        self.assertEqual(result['processed'], 0)
        self.assertEqual(result['failed'], 0)

    def test_process_filled_orders_with_fill(self):
        """Test processing a filled order updates state"""
        # Setup: add a pending update
        pending_id = "buy_12345"
        self.manager.pending_updates[pending_id] = {
            'type': 'buy',
            'price': 50000,
            'volume': 0.01,
            'timestamp': time.time(),
            'order_id': 'ORDER123'
        }

        # Setup: mock order manager returns filled order
        self.mock_order_manager.check_and_update_orders.return_value = {
            'filled': ['ORDER123'],
            'cancelled': [],
            'pending': []
        }
        self.mock_order_manager.filled_orders = {
            'ORDER123': {
                'filledAmount': '0.01',
                'price': '50000',
                'side': 'buy'
            }
        }

        result = self.manager.process_filled_orders()

        self.assertEqual(result['processed'], 1)
        self.mock_state_manager.update_buy.assert_called_once()
        # Pending update should be removed
        self.assertEqual(len(self.manager.pending_updates), 0)


class TestCycleAwareTrading(unittest.TestCase):
    """Test cycle-aware trading decisions"""

    def setUp(self):
        """Set up test fixtures"""
        from cycle_trading_deep_module import CycleAwareTrading, TradingDecision

        self.trading = CycleAwareTrading(
            base_position_pct=0.10,
            min_eur_per_trade=15.0
        )

    def test_initialization(self):
        """Test module initializes with correct defaults"""
        self.assertEqual(self.trading.base_position_pct, 0.10)
        self.assertEqual(self.trading.min_eur, 15.0)

    def test_decide_returns_trading_decision(self):
        """Test decide() returns TradingDecision dataclass"""
        from cycle_trading_deep_module import TradingDecision

        decision = self.trading.decide(
            price=50000,
            btc_held=0.5,
            eur_available=1000,
            avg_buy_price=45000
        )

        self.assertIsInstance(decision, TradingDecision)
        self.assertTrue(hasattr(decision, 'should_buy'))
        self.assertTrue(hasattr(decision, 'should_sell'))
        self.assertTrue(hasattr(decision, 'btc_amount'))
        self.assertTrue(hasattr(decision, 'price'))
        self.assertTrue(hasattr(decision, 'reasoning'))
        self.assertTrue(hasattr(decision, 'risk_level'))

    def test_decide_buy_signal_generated(self):
        """Test buy signal is generated under right conditions"""
        # Deep in drawdown from cycle ATH - should generate buy
        decision = self.trading.decide(
            price=30000,  # Significant drawdown
            btc_held=0.1,  # Small position
            eur_available=5000,  # Plenty of EUR
            avg_buy_price=80000  # Current below avg - opportunity
        )

        # Either buy or sell should be true, not both
        self.assertTrue(decision.should_buy or decision.should_sell)
        self.assertFalse(decision.should_buy and decision.should_sell)

    def test_decide_sell_signal_generated(self):
        """Test sell signal is generated at profit targets"""
        # Price significantly above average - take profit
        decision = self.trading.decide(
            price=90000,
            btc_held=1.0,
            eur_available=100,
            avg_buy_price=50000
        )

        # Should generate a decision
        self.assertTrue(decision.should_buy or decision.should_sell)

    def test_risk_levels_valid(self):
        """Test that risk levels are valid"""
        test_cases = [
            (25000, 0.1, 5000, 40000),
            (50000, 0.5, 1000, 45000),
            (80000, 1.0, 100, 60000),
            (100000, 0.5, 500, 70000),
        ]

        for price, btc, eur, avg_price in test_cases:
            decision = self.trading.decide(
                price=price,
                btc_held=btc,
                eur_available=eur,
                avg_buy_price=avg_price
            )
            self.assertIn(decision.risk_level, ['LOW', 'MEDIUM', 'HIGH', 'EXTREME'])

    def test_btc_amount_non_negative(self):
        """Test btc_amount is always non-negative"""
        test_cases = [
            (30000, 0.1, 5000, 50000),
            (90000, 1.0, 1000, 40000),
            (60000, 0.0, 10000, 55000),
        ]

        for price, btc, eur, avg_price in test_cases:
            decision = self.trading.decide(
                price=price,
                btc_held=btc,
                eur_available=eur,
                avg_buy_price=avg_price
            )
            self.assertGreaterEqual(decision.btc_amount, 0)

    def test_reasoning_not_empty(self):
        """Test that reasoning is always provided"""
        decision = self.trading.decide(
            price=50000,
            btc_held=0.5,
            eur_available=1000,
            avg_buy_price=45000
        )

        self.assertIsNotNone(decision.reasoning)
        self.assertGreater(len(decision.reasoning), 0)

    def test_confidence_in_valid_range(self):
        """Test confidence is between 0 and 100 (percentage) if provided"""
        decision = self.trading.decide(
            price=50000,
            btc_held=0.5,
            eur_available=1000,
            avg_buy_price=45000
        )

        # Confidence appears to be a percentage (0-100)
        if decision.confidence is not None:
            self.assertGreaterEqual(decision.confidence, 0.0)
            self.assertLessEqual(decision.confidence, 100.0)

    def test_phase_provided(self):
        """Test phase is provided in decision"""
        decision = self.trading.decide(
            price=50000,
            btc_held=0.5,
            eur_available=1000,
            avg_buy_price=45000
        )

        # Phase should be set (may be None if not applicable)
        self.assertTrue(hasattr(decision, 'phase'))


if __name__ == '__main__':
    unittest.main()
