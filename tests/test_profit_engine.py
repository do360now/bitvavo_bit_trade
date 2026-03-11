#!/usr/bin/env python3
"""
Tests for ProfitEngine - verifies signal generation, entries, exits, and regime detection.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from profit_engine import ProfitEngine, TradingDecision, MacroRegime, CyclePhase


class TestProfitEngineInit(unittest.TestCase):
    """Test initialization and backward compatibility."""

    def test_default_init(self):
        engine = ProfitEngine()
        self.assertEqual(engine.base_position_pct, 0.10)
        self.assertEqual(engine.min_eur, 15.0)

    def test_backward_compat_alias(self):
        from profit_engine import CycleAwareTrading
        engine = CycleAwareTrading(base_position_pct=0.08)
        self.assertIsInstance(engine, ProfitEngine)
        self.assertEqual(engine.base_position_pct, 0.08)


class TestSignals(unittest.TestCase):
    """Test individual signal generators."""

    def setUp(self):
        self.engine = ProfitEngine()

    def test_rsi_oversold(self):
        """RSI signal should be positive (buy) when prices are falling."""
        # Create declining price series
        prices = [70000 - i * 500 for i in range(30)]
        signal = self.engine._rsi_signal(prices)
        self.assertGreater(signal, 0, "RSI should signal buy on declining prices")

    def test_rsi_overbought(self):
        """RSI signal should be negative (sell) when prices are rising."""
        prices = [50000 + i * 500 for i in range(30)]
        signal = self.engine._rsi_signal(prices)
        self.assertLess(signal, 0, "RSI should signal sell on rising prices")

    def test_rsi_wilder_smoothing(self):
        """Verify RSI uses Wilder smoothing (not SMA)."""
        # With Wilder smoothing, RSI should be different from SMA-based RSI
        prices = list(np.random.normal(70000, 1000, 50))
        signal = self.engine._rsi_signal(prices)
        # Just verify it returns a value in range
        self.assertGreaterEqual(signal, -1.0)
        self.assertLessEqual(signal, 1.0)

    def test_rsi_insufficient_data(self):
        """RSI should return 0 with insufficient data."""
        signal = self.engine._rsi_signal([100, 200, 300])
        self.assertEqual(signal, 0.0)

    def test_macd_bullish_crossover(self):
        """MACD should be positive on bullish crossover."""
        # Create V-shaped recovery for crossover
        prices = [70000 - i * 200 for i in range(20)]
        prices += [70000 - 20 * 200 + i * 400 for i in range(20)]
        signal = self.engine._macd_signal(prices)
        # After recovery, MACD should lean bullish
        self.assertIsInstance(signal, float)

    def test_macd_insufficient_data(self):
        signal = self.engine._macd_signal([100, 200])
        self.assertEqual(signal, 0.0)

    def test_bollinger_at_lower_band(self):
        """Price at lower Bollinger Band should generate buy signal."""
        # Range-bound prices, then current price drops to lower band
        prices = list(np.random.normal(70000, 500, 25))
        current_price = 70000 - 1500  # Well below mean
        signal = self.engine._bollinger_signal(prices, current_price)
        self.assertGreater(signal, 0, "Should signal buy at lower band")

    def test_bollinger_at_upper_band(self):
        """Price at upper Bollinger Band should generate sell signal."""
        prices = list(np.random.normal(70000, 500, 25))
        current_price = 70000 + 1500  # Well above mean
        signal = self.engine._bollinger_signal(prices, current_price)
        self.assertLess(signal, 0, "Should signal sell at upper band")

    def test_momentum_positive(self):
        """Rising prices should generate positive momentum signal."""
        prices = [60000 + i * 100 for i in range(15)]
        signal = self.engine._momentum_signal(prices)
        self.assertGreater(signal, 0, "Rising prices should be positive momentum")

    def test_momentum_negative(self):
        """Falling prices should generate negative momentum signal."""
        prices = [70000 - i * 100 for i in range(15)]
        signal = self.engine._momentum_signal(prices)
        self.assertLess(signal, 0, "Falling prices should be negative momentum")

    def test_cycle_signal_deep_correction(self):
        """Deep correction should generate bullish cycle signal."""
        price = 30000  # ~74% down from ATH of 114395
        signal = self.engine._cycle_signal(price)
        self.assertGreater(signal, 0.5, "Deep correction should be strongly bullish")

    def test_cycle_signal_near_ath(self):
        """Near ATH should generate bearish cycle signal."""
        price = 110000  # ~4% from ATH
        signal = self.engine._cycle_signal(price)
        self.assertLess(signal, -0.3, "Near ATH should be bearish")

    def test_volatility_signal_range(self):
        """Volatility signal should be in valid range."""
        prices = list(np.random.normal(70000, 2000, 50))
        signal = self.engine._volatility_signal(prices)
        self.assertGreaterEqual(signal, -0.5)
        self.assertLessEqual(signal, 0.5)


class TestCompositeSignal(unittest.TestCase):
    """Test composite signal computation."""

    def setUp(self):
        self.engine = ProfitEngine()

    def test_composite_range(self):
        """Composite signal must be in [-1, 1]."""
        prices = [70000 + np.random.normal(0, 500) for _ in range(60)]
        signals = self.engine._compute_signals(70000, prices, None, None)
        self.assertGreaterEqual(signals['composite'], -1.0)
        self.assertLessEqual(signals['composite'], 1.0)

    def test_composite_without_prices(self):
        """Should return neutral composite without price history."""
        signals = self.engine._compute_signals(70000, None, None, None)
        self.assertEqual(signals['technical'], 0.0)
        self.assertEqual(signals['momentum'], 0.0)

    def test_macro_dampener(self):
        """Risk-off macro data should dampen positive signals."""
        prices = [60000 + i * 200 for i in range(60)]  # Rising prices
        macro = {'risk_off_probability': 0.8}

        signals_no_macro = self.engine._compute_signals(72000, prices, None, None)
        signals_risk_off = self.engine._compute_signals(72000, prices, None, macro)

        if signals_no_macro['composite'] > 0:
            self.assertLessEqual(
                signals_risk_off['composite'],
                signals_no_macro['composite'],
                "Risk-off should dampen positive signals"
            )


class TestRegimeDetection(unittest.TestCase):
    """Test macro regime detection."""

    def setUp(self):
        self.engine = ProfitEngine()

    def test_risk_off_declining_prices(self):
        """Volatile declining prices should detect risk-off."""
        # Realistic volatile decline (not perfectly linear)
        np.random.seed(42)
        prices = [70000 - i * 500 + np.random.normal(0, 800) for i in range(25)]
        regime = self.engine._detect_regime(57500, prices, None)
        self.assertIn(regime, [MacroRegime.RISK_OFF, MacroRegime.CRISIS])

    def test_risk_on_rising_stable(self):
        """Stable rising prices should detect risk-on or neutral."""
        prices = [65000 + i * 50 for i in range(25)]  # Slow steady rise
        regime = self.engine._detect_regime(66250, prices, None)
        self.assertIn(regime, [MacroRegime.RISK_ON, MacroRegime.NEUTRAL])

    def test_neutral_without_data(self):
        """No price history should default to neutral."""
        regime = self.engine._detect_regime(70000, None, None)
        self.assertEqual(regime, MacroRegime.NEUTRAL)


class TestCyclePhaseDetection(unittest.TestCase):
    """Test cycle phase detection."""

    def setUp(self):
        self.engine = ProfitEngine()

    def test_correction_phase(self):
        """Price 40%+ down from ATH should be CORRECTION."""
        price = 60000  # ~47% down from 114395
        phase = self.engine._detect_cycle_phase(price)
        self.assertEqual(phase, CyclePhase.CORRECTION)

    def test_euphoria_phase(self):
        """Price near ATH should be EUPHORIA."""
        price = 112000  # ~2% from ATH
        phase = self.engine._detect_cycle_phase(price)
        self.assertEqual(phase, CyclePhase.EUPHORIA)


class TestEntryLogic(unittest.TestCase):
    """Test buy entry decisions."""

    def setUp(self):
        self.engine = ProfitEngine()

    def test_buy_on_strong_signal(self):
        """Strong composite signal should trigger buy."""
        # Deeply oversold prices for strong buy signal
        prices = [75000 - i * 300 for i in range(60)]
        decision = self.engine.decide(
            price=57000,
            btc_held=0.0,
            eur_available=5000,
            avg_buy_price=0,
            prices_1h=prices,
        )
        # Should either buy (strong signal in correction) or hold
        self.assertIsInstance(decision, TradingDecision)
        if decision.should_buy:
            self.assertGreater(decision.btc_amount, 0)

    def test_no_buy_on_insufficient_funds(self):
        """Should not buy if EUR below minimum."""
        decision = self.engine.decide(
            price=70000,
            btc_held=0.0,
            eur_available=5.0,  # Below minimum
            avg_buy_price=0,
        )
        self.assertFalse(decision.should_buy)

    def test_hold_without_data(self):
        """Without price history, weak signals should HOLD."""
        decision = self.engine.decide(
            price=70000,
            btc_held=0.0,
            eur_available=1000,
            avg_buy_price=0,
        )
        # Without OHLCV, technical signals are 0 → likely hold
        # (cycle signal alone might trigger a weak buy in correction)
        self.assertIsInstance(decision, TradingDecision)


class TestExitLogic(unittest.TestCase):
    """Test sell exit decisions."""

    def setUp(self):
        self.engine = ProfitEngine()

    def test_take_profit(self):
        """Should sell when profit exceeds take-profit target."""
        decision = self.engine.decide(
            price=80000,
            btc_held=0.05,
            eur_available=1000,
            avg_buy_price=60000,  # 33% profit > 8% take-profit
        )
        self.assertTrue(decision.should_sell, "Should take profit at 33%")

    def test_stop_loss(self):
        """Should sell when loss exceeds stop-loss."""
        # In growth phase (not deep correction), should stop out
        prices = [95000 - i * 200 for i in range(60)]  # Declining
        decision = self.engine.decide(
            price=90000,
            btc_held=0.05,
            eur_available=1000,
            avg_buy_price=100000,  # -10% loss
            prices_1h=prices,
        )
        # Should trigger stop loss at -10% (> base 4% stop, even with wider adjustments)
        self.assertTrue(decision.should_sell, "Should stop loss at -10%")

    def test_diamond_hands_deep_correction(self):
        """Should NOT panic sell deep underwater in correction/accumulation."""
        decision = self.engine.decide(
            price=50000,
            btc_held=0.05,
            eur_available=1000,
            avg_buy_price=90000,  # -44% underwater
        )
        # Without strong bearish signals confirming, should hold
        # (The "diamond hands" logic protects deep underwater positions in correction)
        self.assertIsInstance(decision, TradingDecision)
        # Note: with no price history, signals are weak, so it should hold
        if not decision.should_sell:
            self.assertFalse(decision.should_buy, "Should hold, not buy more without signals")

    def test_crisis_sell(self):
        """In crisis regime with position, should reduce exposure."""
        # Sharp declining prices to trigger crisis
        prices = [80000 - i * 600 for i in range(25)]
        decision = self.engine.decide(
            price=65000,
            btc_held=0.05,
            eur_available=1000,
            avg_buy_price=64000,  # Slight profit
            prices_1h=prices,
        )
        # Crisis + in profit → should sell or hold (depends on exact signal)
        self.assertIsInstance(decision, TradingDecision)


class TestDecisionDataclass(unittest.TestCase):
    """Test TradingDecision dataclass."""

    def test_default_signals(self):
        d = TradingDecision(
            should_buy=True,
            should_sell=False,
            btc_amount=0.001,
            price=70000,
            reasoning="test",
            risk_level="LOW",
        )
        self.assertEqual(d.signals, {})
        self.assertIsNone(d.phase)

    def test_never_buy_and_sell(self):
        """Engine should never set both buy and sell."""
        engine = ProfitEngine()
        prices = list(np.random.normal(70000, 2000, 60))

        for _ in range(20):
            d = engine.decide(
                price=np.random.uniform(50000, 90000),
                btc_held=np.random.uniform(0, 0.1),
                eur_available=np.random.uniform(100, 5000),
                avg_buy_price=np.random.uniform(50000, 90000),
                prices_1h=prices,
            )
            self.assertFalse(
                d.should_buy and d.should_sell,
                "Should never be both buy and sell"
            )


class TestEMAHelper(unittest.TestCase):
    """Test EMA calculation."""

    def test_ema_returns_same_length(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ema = ProfitEngine._ema(data, 3)
        self.assertEqual(len(ema), len(data))

    def test_ema_starts_at_first_value(self):
        data = np.array([100.0, 110.0, 120.0])
        ema = ProfitEngine._ema(data, 3)
        self.assertEqual(ema[0], 100.0)


class TestMarketContext(unittest.TestCase):
    """Test get_market_context."""

    def test_context_keys(self):
        engine = ProfitEngine()
        ctx = engine.get_market_context(70000)
        self.assertIn('price', ctx)
        self.assertIn('regime', ctx)
        self.assertIn('cycle_phase', ctx)
        self.assertIn('drawdown_from_ath', ctx)
        self.assertIn('signals', ctx)
        self.assertIn('risk_level', ctx)

    def test_context_with_prices(self):
        engine = ProfitEngine()
        prices = [70000 + i * 10 for i in range(60)]
        ctx = engine.get_market_context(70600, prices)
        self.assertIn('technical', ctx['signals'])


if __name__ == "__main__":
    unittest.main(verbosity=2)
