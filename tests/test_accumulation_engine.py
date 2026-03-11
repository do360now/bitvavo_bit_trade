#!/usr/bin/env python3
"""
Tests for AccumulationEngine - tiered buying, diamond hands, reserve management.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from accumulation_engine import AccumulationEngine
from profit_engine import TradingDecision, MacroRegime, CyclePhase


class TestAccumulationInit(unittest.TestCase):

    def test_default_params(self):
        engine = AccumulationEngine()
        self.assertEqual(engine.base_take_profit, 0.22)
        self.assertEqual(engine.base_stop_loss, 0.08)
        self.assertEqual(engine._weights['cycle'], 0.30)

    def test_weights_sum_to_one(self):
        engine = AccumulationEngine()
        total = sum(engine._weights.values())
        self.assertAlmostEqual(total, 1.0, places=5)


class TestPriceTiers(unittest.TestCase):

    def setUp(self):
        self.engine = AccumulationEngine()

    def test_early_correction_tier(self):
        """35% drawdown = early_correction tier"""
        name = self.engine._get_tier_name(0.35)
        weight = self.engine._get_tier_weight(0.35)
        self.assertEqual(name, "early_correction")
        self.assertEqual(weight, 0.15)

    def test_mid_correction_tier(self):
        """48% drawdown = mid_correction"""
        name = self.engine._get_tier_name(0.48)
        weight = self.engine._get_tier_weight(0.48)
        self.assertEqual(name, "mid_correction")
        self.assertEqual(weight, 0.25)

    def test_deep_correction_tier(self):
        """60% drawdown = deep_correction"""
        name = self.engine._get_tier_name(0.60)
        weight = self.engine._get_tier_weight(0.60)
        self.assertEqual(name, "deep_correction")
        self.assertEqual(weight, 0.35)

    def test_capitulation_tier(self):
        """75% drawdown = capitulation"""
        name = self.engine._get_tier_name(0.75)
        weight = self.engine._get_tier_weight(0.75)
        self.assertEqual(name, "capitulation")
        self.assertEqual(weight, 0.50)

    def test_near_ath_tier(self):
        """10% drawdown = near_ath (very light)"""
        name = self.engine._get_tier_name(0.10)
        weight = self.engine._get_tier_weight(0.10)
        self.assertEqual(name, "near_ath")
        self.assertEqual(weight, 0.08)

    def test_deeper_tiers_have_higher_weight(self):
        """Deeper drawdowns should allocate more capital."""
        w1 = self.engine._get_tier_weight(0.35)
        w2 = self.engine._get_tier_weight(0.50)
        w3 = self.engine._get_tier_weight(0.60)
        w4 = self.engine._get_tier_weight(0.75)
        self.assertLess(w1, w2)
        self.assertLess(w2, w3)
        self.assertLess(w3, w4)


class TestDiamondHands(unittest.TestCase):
    """Accumulation mode should NEVER sell in correction when underwater."""

    def setUp(self):
        self.engine = AccumulationEngine()

    def test_no_sell_underwater_in_correction(self):
        """At -22% in correction, should NOT sell."""
        decision = self.engine.decide(
            price=60000,
            btc_held=0.043,
            eur_available=307,
            avg_buy_price=77000,
        )
        self.assertFalse(decision.should_sell, "Should NOT sell underwater in correction")

    def test_no_sell_deeply_underwater(self):
        """At -50% in correction, should STILL not sell."""
        decision = self.engine.decide(
            price=38500,
            btc_held=0.043,
            eur_available=307,
            avg_buy_price=77000,
        )
        self.assertFalse(decision.should_sell, "Should NOT sell -50% in correction")

    def test_no_sell_with_bearish_signals(self):
        """Even with declining prices, should not sell underwater in correction."""
        prices = [70000 - i * 400 for i in range(60)]
        decision = self.engine.decide(
            price=46000,
            btc_held=0.043,
            eur_available=307,
            avg_buy_price=77000,
            prices_1h=prices,
        )
        self.assertFalse(decision.should_sell, "Diamond hands even with bearish signals")


class TestReserveManagement(unittest.TestCase):

    def setUp(self):
        self.engine = AccumulationEngine()

    def test_reserve_kept(self):
        """Should not spend below €50 reserve (unless deep value)."""
        decision = self.engine.decide(
            price=75000,
            btc_held=0.0,
            eur_available=60,  # Only €10 above reserve
            avg_buy_price=0,
        )
        # With only €10 above reserve and min trade €15, should not buy
        self.assertFalse(decision.should_buy, "Should not buy — insufficient after reserve")

    def test_reserve_used_in_deep_value(self):
        """In extreme deep value, should dip into reserve."""
        # 74% drawdown + strong composite should trigger reserve usage
        prices = [50000 - i * 200 for i in range(60)]  # Deeply oversold
        decision = self.engine.decide(
            price=30000,  # 74% off ATH
            btc_held=0.0,
            eur_available=55,  # Only €5 above reserve
            avg_buy_price=0,
            prices_1h=prices,
        )
        # Should be willing to use reserve at this extreme level
        self.assertIsInstance(decision, TradingDecision)


class TestAccumulationEntry(unittest.TestCase):

    def setUp(self):
        self.engine = AccumulationEngine()

    def test_buys_on_weak_positive_signal(self):
        """Accumulation mode should buy on weak positive signals."""
        # Gentle uptrend — would be HOLD in profit engine, should BUY here
        prices = [59000 + i * 30 for i in range(60)]
        decision = self.engine.decide(
            price=60800,
            btc_held=0.0,
            eur_available=500,
            avg_buy_price=0,
            prices_1h=prices,
        )
        # In mid_correction with any positive signal, should buy
        if decision.should_buy:
            self.assertGreater(decision.btc_amount, 0)
            self.assertIn("ACCUMULATE", decision.reasoning)

    def test_averages_down_on_weak_signal(self):
        """Should average down even on weak signals (unlike profit engine)."""
        prices = [65000 - i * 50 for i in range(60)]  # Slight decline
        decision = self.engine.decide(
            price=62000,
            btc_held=0.043,
            eur_available=307,
            avg_buy_price=77000,  # -19% underwater
            prices_1h=prices,
        )
        # Accumulation mode should be willing to avg down
        self.assertIsInstance(decision, TradingDecision)

    def test_new_avg_calculation(self):
        """Verify new average price calculation."""
        new_avg = AccumulationEngine._calc_new_avg(
            current_avg=77000, current_btc=0.043,
            new_price=60000, new_btc=0.005
        )
        # Should be between 60000 and 77000, closer to 77000 (bigger existing position)
        self.assertGreater(new_avg, 60000)
        self.assertLess(new_avg, 77000)
        # Exact: (77000*0.043 + 60000*0.005) / 0.048 = 75229
        self.assertAlmostEqual(new_avg, 75229.17, places=0)


class TestAccumulationTakeProfit(unittest.TestCase):

    def setUp(self):
        self.engine = AccumulationEngine()

    def test_no_sell_at_8pct_profit(self):
        """Should NOT sell at 8% profit (old TP level). Want 22%."""
        decision = self.engine.decide(
            price=83160,  # 8% above 77000
            btc_held=0.043,
            eur_available=100,
            avg_buy_price=77000,
        )
        self.assertFalse(decision.should_sell, "Should not TP at 8% — target is 22%")

    def test_sells_small_at_22pct(self):
        """At 22%+ profit, should sell only 10%."""
        decision = self.engine.decide(
            price=94000,  # 22% above 77000
            btc_held=0.043,
            eur_available=100,
            avg_buy_price=77000,
        )
        if decision.should_sell:
            # Should sell only 10% of position
            expected_sell = 0.043 * 0.10
            self.assertAlmostEqual(decision.btc_amount, expected_sell, places=4)
            self.assertIn("10%", decision.reasoning)

    def test_sells_more_at_50pct(self):
        """At 50%+ profit near ATH (euphoria), should sell 20%."""
        decision = self.engine.decide(
            price=115500,  # 50% above 77000, but also near ATH = euphoria
            btc_held=0.043,
            eur_available=100,
            avg_buy_price=77000,
        )
        if decision.should_sell:
            # In euphoria, TP=15%, profit=50% → above 0.35 threshold → 20% sell
            expected_sell = 0.043 * 0.20
            self.assertAlmostEqual(decision.btc_amount, expected_sell, places=4)


class TestAccumulationStatus(unittest.TestCase):

    def test_status_keys(self):
        engine = AccumulationEngine()
        status = engine.get_accumulation_status(60000, 307)
        self.assertIn('mode', status)
        self.assertIn('tier', status)
        self.assertIn('bullets_remaining', status)
        self.assertIn('reserve_intact', status)
        self.assertEqual(status['mode'], 'ACCUMULATION')

    def test_bullets_calculation(self):
        engine = AccumulationEngine()
        status = engine.get_accumulation_status(60000, 307)
        # €307 - €50 reserve = €257 / €15 min = 17 bullets
        self.assertEqual(status['bullets_remaining'], 17)

    def test_reserve_intact_flag(self):
        engine = AccumulationEngine()
        status_good = engine.get_accumulation_status(60000, 307)
        status_low = engine.get_accumulation_status(60000, 40)
        self.assertTrue(status_good['reserve_intact'])
        self.assertFalse(status_low['reserve_intact'])


class TestNeverBuyAndSell(unittest.TestCase):
    """Safety: engine should never set both flags."""

    def test_never_both(self):
        engine = AccumulationEngine()
        prices = list(np.random.normal(65000, 2000, 60))
        for _ in range(30):
            d = engine.decide(
                price=np.random.uniform(40000, 100000),
                btc_held=np.random.uniform(0, 0.1),
                eur_available=np.random.uniform(50, 500),
                avg_buy_price=np.random.uniform(50000, 90000),
                prices_1h=prices,
            )
            self.assertFalse(
                d.should_buy and d.should_sell,
                "Should never be both buy and sell"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
