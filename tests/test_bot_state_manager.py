"""
Tests for BotStateManager - Critical for accumulation strategy
"""

import pytest
import json
import os
import sys
from datetime import datetime

# Add parent directory to path so we can import bot_state_manager
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit_breaker import CircuitBreaker
from bot_state_manager import BotStateManager


@pytest.fixture
def temp_state_file(tmp_path):
    """Create temporary state file for testing"""
    return str(tmp_path / "test_state.json")


@pytest.fixture
def state_manager(temp_state_file):
    """Create fresh state manager for each test"""
    return BotStateManager(temp_state_file)


class TestBotStateManager:
    """Test suite for BotStateManager"""
    
    def test_fresh_state_initialization(self, state_manager):
        """Test that fresh state has correct defaults"""
        assert state_manager.get_avg_buy_price() == 0.0
        assert state_manager.get_total_btc_bought() == 0.0
        assert state_manager.get_peak_price() == 0.0
    
    def test_single_buy_updates_avg_price(self, state_manager):
        """Test that first buy sets avg price correctly"""
        state_manager.update_buy(price=50000.0, volume=0.01)
        
        assert state_manager.get_avg_buy_price() == 50000.0
        assert state_manager.get_total_btc_bought() == 0.01
    
    def test_multiple_buys_calculate_weighted_average(self, state_manager):
        """Test weighted average calculation with multiple buys"""
        # Buy 0.01 BTC at €50,000
        state_manager.update_buy(price=50000.0, volume=0.01)
        
        # Buy 0.02 BTC at €60,000
        state_manager.update_buy(price=60000.0, volume=0.02)
        
        # Expected avg: (50000*0.01 + 60000*0.02) / 0.03 = €56,666.67
        expected_avg = (50000 * 0.01 + 60000 * 0.02) / 0.03
        
        assert abs(state_manager.get_avg_buy_price() - expected_avg) < 0.01
        assert state_manager.get_total_btc_bought() == 0.03
    
    def test_buy_with_fees_increases_cost_basis(self, state_manager):
        """Test that fees are included in cost basis"""
        # Buy 0.01 BTC at €50,000 with €12.50 fee (0.25%)
        fee = 50000 * 0.01 * 0.0025
        state_manager.update_buy(price=50000.0, volume=0.01, fee_eur=fee)
        
        # Cost basis should include fee: (50000*0.01 + 12.50) / 0.01 = €51,250
        expected_avg = (50000 * 0.01 + fee) / 0.01
        
        assert abs(state_manager.get_avg_buy_price() - expected_avg) < 0.01
        assert state_manager.state['total_fees_eur'] == fee
    
    def test_profitable_sell_increments_success_counter(self, state_manager):
        """Test that profitable sells are counted correctly"""
        # Buy at €50,000
        state_manager.update_buy(price=50000.0, volume=0.01)
        
        # Sell at €55,000 (profitable)
        state_manager.update_sell(price=55000.0, volume=0.01)
        
        stats = state_manager.get_statistics()
        assert stats['round_trips'] == 1
        assert stats['successful_round_trips'] == 1
        assert stats['win_rate'] == 1.0
    
    def test_unprofitable_sell_does_not_increment_success(self, state_manager):
        """Test that unprofitable sells are tracked but not counted as wins"""
        # Buy at €50,000
        state_manager.update_buy(price=50000.0, volume=0.01)
        
        # Sell at €45,000 (loss)
        state_manager.update_sell(price=45000.0, volume=0.01)
        
        stats = state_manager.get_statistics()
        assert stats['round_trips'] == 1
        assert stats['successful_round_trips'] == 0
        assert stats['win_rate'] == 0.0
    
    def test_state_persists_across_restarts(self, temp_state_file):
        """Test that state is saved and loaded correctly"""
        # Create first instance and make trades
        manager1 = BotStateManager(temp_state_file)
        manager1.update_buy(price=50000.0, volume=0.01)
        manager1.update_buy(price=60000.0, volume=0.02)
        
        avg_before = manager1.get_avg_buy_price()
        total_before = manager1.get_total_btc_bought()
        
        # Create new instance (simulates restart)
        manager2 = BotStateManager(temp_state_file)
        
        # Should load previous state
        assert manager2.get_avg_buy_price() == avg_before
        assert manager2.get_total_btc_bought() == total_before
    
    def test_peak_price_tracking(self, state_manager):
        """Test that peak price is tracked correctly"""
        prices = [50000, 55000, 60000, 58000, 57000]
        
        for price in prices:
            state_manager.update_peak(price)
        
        assert state_manager.get_peak_price() == 60000  # Highest price
    
    def test_true_profit_margin_includes_fees(self, state_manager):
        """Test that profit calculation includes fees"""
        # Buy 0.01 BTC at €50,000 with €1.25 fee (0.25%)
        state_manager.update_buy(price=50000.0, volume=0.01, fee_eur=1.25)
        
        # Current price €55,000
        current_price = 55000.0
        
        # True cost basis: €50,125 (€50,000 + €1.25 fee = €501.25 total / 0.01 BTC)
        # Profit: (€55,000 - €50,125) / €50,125 = 9.73%
        profit_margin = state_manager.get_true_profit_margin(current_price)
        
        expected_margin = ((55000 - 50125) / 50125) * 100
        assert abs(profit_margin - expected_margin) < 0.01
    
    def test_edge_case_zero_volume_buy(self, state_manager):
        """Test handling of zero volume (should not crash)"""
        state_manager.update_buy(price=50000.0, volume=0.0)
        
        # Should handle gracefully (might result in inf or special handling)
        # At minimum, should not crash
        assert True  # If we get here, no crash occurred
    
    def test_edge_case_very_small_volumes(self, state_manager):
        """Test with satoshi-level precision"""
        # Buy 1 satoshi worth of BTC
        state_manager.update_buy(price=50000.0, volume=0.00000001)
        
        assert state_manager.get_avg_buy_price() == 50000.0
        assert state_manager.get_total_btc_bought() == 0.00000001
    
    def test_complex_trading_scenario(self, state_manager):
        """Test realistic trading scenario with multiple buys and sells"""
        # Initial buys
        state_manager.update_buy(price=50000.0, volume=0.01, fee_eur=12.50)
        state_manager.update_buy(price=48000.0, volume=0.02, fee_eur=24.00)
        state_manager.update_buy(price=52000.0, volume=0.01, fee_eur=13.00)
        
        # Check intermediate state
        total_cost = (50000*0.01 + 12.50) + (48000*0.02 + 24.00) + (52000*0.01 + 13.00)
        expected_avg = total_cost / 0.04
        
        assert abs(state_manager.get_avg_buy_price() - expected_avg) < 0.01
        
        # Sell some
        state_manager.update_sell(price=55000.0, volume=0.02, fee_eur=27.50)
        
        # Stats should reflect trades
        stats = state_manager.get_statistics()
        assert stats['round_trips'] == 1
        assert stats['total_fees_eur'] == 12.50 + 24.00 + 13.00 + 27.50


# Run with: pytest tests/test_bot_state_manager.py -v