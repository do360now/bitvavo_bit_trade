"""
Accumulation Engine - Signal-Timed Bitcoin Accumulation

PHILOSOPHY:
You believe BTC goes higher long-term. You want to buy more. But you want
to be SMART about it — time entries with technical signals, buy heavier at
lower prices, and preserve dry powder for deeper dips.

This is NOT a DCA bot. It uses the full signal architecture from ProfitEngine
but biases everything toward accumulation:
- Lower entry thresholds (you want to buy, not wait forever)
- Tiered price ladder (heavier buys at lower prices)
- Wide stops (don't get shaken out of conviction positions)
- Minimal selling (only on extreme signals or euphoria take-profit)
- Reserve management (always keep a final bullet)

INHERITS: ProfitEngine (all signal computation, regime detection, etc.)
OVERRIDES: Entry logic, exit logic, position sizing, thresholds

USAGE:
    engine = AccumulationEngine(total_capital=307.0)
    decision = engine.decide(
        price=60000, btc_held=0.043, eur_available=307,
        avg_buy_price=77000, prices_1h=prices
    )
"""

from typing import Optional, List, Dict
from logger_config import logger
from profit_engine import (
    ProfitEngine, TradingDecision, MacroRegime, CyclePhase
)


class AccumulationEngine(ProfitEngine):
    """
    Signal-timed accumulation engine.

    Same interface as ProfitEngine. Same decide() method.
    Different behavior: biased toward buying, reluctant to sell.

    KEY DIFFERENCES FROM ProfitEngine:
    ┌─────────────────────┬──────────────┬──────────────────┐
    │ Parameter           │ ProfitEngine │ AccumulationEngine│
    ├─────────────────────┼──────────────┼──────────────────┤
    │ Take Profit         │ 8%           │ 22%              │
    │ Stop Loss           │ 4%           │ 8% (15% in corr) │
    │ Entry threshold     │ +0.20        │ +0.08            │
    │ Avg-down threshold  │ +0.40        │ +0.05            │
    │ Signal-exit thresh  │ -0.60        │ -0.80            │
    │ Cycle weight        │ 15%          │ 30%              │
    │ Min sell at exit    │ 25-60%       │ 10-30%           │
    │ Reserve             │ None         │ €50 always kept  │
    │ Price tiers         │ None         │ 4 drawdown tiers │
    └─────────────────────┴──────────────┴──────────────────┘
    """

    # Price tiers: drawdown_from_ath → capital allocation weight
    # Deeper dip = more aggressive buying
    _PRICE_TIERS = [
        # (min_drawdown, max_drawdown, tier_name, capital_weight)
        (0.30, 0.45, "early_correction", 0.15),    # 30-45% off: light buys
        (0.45, 0.55, "mid_correction", 0.25),       # 45-55% off: moderate buys
        (0.55, 0.70, "deep_correction", 0.35),      # 55-70% off: heavy buys
        (0.70, 1.00, "capitulation", 0.50),          # 70%+ off:  maximum buys
    ]

    # Reserve: always keep this much EUR as a final bullet
    _RESERVE_EUR = 50.0

    def __init__(self,
                 base_position_pct: float = 0.12,
                 min_eur_per_trade: float = 15.0,
                 total_capital: float = 0.0):
        """
        Initialize accumulation engine.

        Args:
            base_position_pct: Base position size (12% default, higher than profit engine)
            min_eur_per_trade: Minimum trade value
            total_capital: Starting EUR capital (for tier allocation tracking)
        """
        super().__init__(
            base_position_pct=base_position_pct,
            min_eur_per_trade=min_eur_per_trade,
            max_position_pct=0.40,       # Allow up to 40% in one buy
            take_profit_pct=0.22,        # 22% TP — hold through recovery
            stop_loss_pct=0.08,          # 8% SL — wide, don't get shaken out
            enable_shorting=False,
        )

        # Override signal weights: cycle matters MORE for accumulation
        self._weights = {
            'technical': 0.35,    # Still important for timing
            'momentum': 0.20,     # Helps avoid catching knives
            'cycle': 0.30,        # Cycle context is key for accumulation
            'volatility': 0.15,   # Vol regime adjustment
        }

        self.total_capital = total_capital
        self._tier_spent = {tier[2]: 0.0 for tier in self._PRICE_TIERS}

        # Cooldown tracking: don't spray bullets at the same price
        self._last_buy_price = 0.0
        self._last_buy_time = 0.0
        self._MIN_PRICE_DROP_FOR_REBUY = 0.025   # Need 2.5% drop from last buy
        self._MIN_HOURS_BETWEEN_BUYS = 4.0        # OR 4 hours must pass
        self._MIN_COMPOSITE_JUMP = 0.15           # OR composite jumps significantly

        logger.info(
            f"AccumulationEngine initialized: base={base_position_pct:.0%}, "
            f"TP=22%, SL=8%, reserve=€{self._RESERVE_EUR}, "
            f"cooldown={self._MIN_HOURS_BETWEEN_BUYS}h/{self._MIN_PRICE_DROP_FOR_REBUY:.1%}, "
            f"mode=ACCUMULATE"
        )

    # ========================================================================
    # OVERRIDDEN: EXIT LOGIC (much less aggressive selling)
    # ========================================================================

    def _check_exit(self,
                    price: float,
                    btc_held: float,
                    avg_buy_price: float,
                    signals: Dict,
                    regime: MacroRegime,
                    cycle_phase: CyclePhase) -> Optional[TradingDecision]:
        """
        Accumulation exit logic — VERY reluctant to sell.

        PHILOSOPHY:
        You want to hold BTC long-term. Selling should be rare:
        - No stop-loss selling in correction (you're accumulating!)
        - Take profit only at 22%+ (you want recovery, not scalps)
        - Signal exit only on extreme bearish signals (-0.80)
        - Crisis exit only at -0.80+ composite AND euphoria phase
        - NEVER sell more than 30% at once (keep the stack)
        """
        profit_pct = (price - avg_buy_price) / avg_buy_price
        composite = signals.get('composite', 0)
        risk_level = self._regime_to_risk(regime)

        # --- STOP LOSS: Very wide, and only in non-correction phases ---
        # In correction/accumulation: NEVER stop out. You're here to buy.
        if cycle_phase in [CyclePhase.CORRECTION, CyclePhase.ACCUMULATION]:
            if profit_pct < 0:
                logger.info(
                    f"💎 ACCUMULATION MODE: Holding {profit_pct:+.1%} in {cycle_phase.value} "
                    f"(accumulating, not selling)"
                )
                return None  # Diamond hands in correction — always

        # In growth/euphoria: wider stop than profit engine
        stop_loss = self.base_stop_loss  # 8%
        if cycle_phase == CyclePhase.GROWTH:
            stop_loss = 0.12  # 12% stop in growth — very wide
        elif cycle_phase == CyclePhase.EUPHORIA:
            stop_loss = 0.06  # 6% in euphoria — tighter to protect gains

        if profit_pct <= -stop_loss and composite < -0.5:
            # Only stop out if signals CONFIRM the decline
            sell_pct = 0.25  # Sell only 25% — keep the stack
            return TradingDecision(
                should_buy=False,
                should_sell=True,
                btc_amount=btc_held * sell_pct,
                price=price,
                reasoning=(
                    f"ACCUMULATION STOP: {profit_pct:+.1%} "
                    f"(signals confirm: {composite:.2f}), selling only {sell_pct:.0%}"
                ),
                risk_level=risk_level,
                phase=cycle_phase.value,
                confidence=80,
                signals=signals,
            )

        # --- TAKE PROFIT: Only at 22%+, and sell small ---
        take_profit = self.base_take_profit  # 22%
        if cycle_phase == CyclePhase.EUPHORIA:
            take_profit = 0.15  # Lower target in euphoria — take some off

        if profit_pct >= take_profit:
            # Scale sell amount conservatively — keep most BTC
            if profit_pct > 0.50:
                sell_pct = 0.30  # At 50%+ profit, take 30%
            elif profit_pct > 0.35:
                sell_pct = 0.20  # At 35%+ profit, take 20%
            else:
                sell_pct = 0.10  # At TP target, take only 10%

            return TradingDecision(
                should_buy=False,
                should_sell=True,
                btc_amount=btc_held * sell_pct,
                price=price,
                reasoning=(
                    f"ACCUMULATION TP: {profit_pct:+.1%} (target: +{take_profit:.0%}), "
                    f"selling {sell_pct:.0%} — keeping {1-sell_pct:.0%} for long-term"
                ),
                risk_level=risk_level,
                phase=cycle_phase.value,
                confidence=85,
                signals=signals,
            )

        # --- SIGNAL EXIT: Only on extreme signals + in profit ---
        if composite <= -0.80 and profit_pct > 0.05:
            sell_pct = 0.15  # Very small — just trim
            return TradingDecision(
                should_buy=False,
                should_sell=True,
                btc_amount=btc_held * sell_pct,
                price=price,
                reasoning=(
                    f"ACCUMULATION TRIM: extreme signal {composite:.2f}, "
                    f"profit={profit_pct:+.1%}, trimming {sell_pct:.0%}"
                ),
                risk_level=risk_level,
                phase=cycle_phase.value,
                confidence=abs(composite) * 100,
                signals=signals,
            )

        # --- NO crisis exit in accumulation mode ---
        # You believe BTC goes higher. Crisis is a buying opportunity, not exit.

        return None  # Default: HOLD (accumulate, don't sell)

    # ========================================================================
    # OVERRIDDEN: ENTRY LOGIC (tiered price ladder, low threshold)
    # ========================================================================

    def _check_entry(self,
                     price: float,
                     eur_available: float,
                     btc_held: float,
                     avg_buy_price: float,
                     signals: Dict,
                     regime: MacroRegime,
                     cycle_phase: CyclePhase) -> Optional[TradingDecision]:
        """
        Accumulation entry logic — tiered price ladder with low threshold.

        KEY DIFFERENCES:
        - Much lower composite threshold (0.08 vs 0.20)
        - Averaging down allowed on weak signals (0.05 vs 0.40)
        - Price tier system: buy heavier at lower prices
        - Reserve management: always keep €50 for emergencies
        - Regime barely affects entry (you're accumulating regardless)
        """
        composite = signals.get('composite', 0)
        risk_level = self._regime_to_risk(regime)
        drawdown = (self._CYCLE_ATH - price) / self._CYCLE_ATH

        # --- BUY COOLDOWN: Don't spray bullets at the same price ---
        # After a buy, wait for EITHER:
        #   a) Price drops 2.5%+ from last buy (new opportunity)
        #   b) 4+ hours pass (conditions genuinely changed)
        #   c) Composite jumps significantly (strong new signal)
        #   d) Price crosses into a new tier (deeper drawdown)
        import time as _time
        if self._last_buy_price > 0:
            price_drop = (self._last_buy_price - price) / self._last_buy_price
            hours_since = (_time.time() - self._last_buy_time) / 3600
            composite_jump = composite > (0.08 + self._MIN_COMPOSITE_JUMP)
            new_tier = self._get_tier_name(drawdown) != self._get_tier_name(
                (self._CYCLE_ATH - self._last_buy_price) / self._CYCLE_ATH
            )

            cooldown_met = (
                price_drop >= self._MIN_PRICE_DROP_FOR_REBUY or  # Price dropped enough
                hours_since >= self._MIN_HOURS_BETWEEN_BUYS or   # Enough time passed
                composite_jump or                                  # Strong new signal
                new_tier                                           # Entered deeper tier
            )

            if not cooldown_met:
                logger.info(
                    f"⏳ COOLDOWN: Last buy at €{self._last_buy_price:,.0f} "
                    f"({hours_since:.1f}h ago, {price_drop:+.1%} from last). "
                    f"Need: {self._MIN_PRICE_DROP_FOR_REBUY:.1%} drop OR "
                    f"{self._MIN_HOURS_BETWEEN_BUYS}h wait"
                )
                return None

        # --- Minimum signal threshold (much lower than profit engine) ---
        min_signal = 0.08   # Almost any positive signal = buy
        if regime == MacroRegime.CRISIS:
            min_signal = 0.15  # Slightly higher bar in crisis
            # But still much lower than profit engine's 0.60
        elif regime == MacroRegime.RISK_OFF:
            min_signal = 0.12  # Slightly higher in risk-off

        # Special: in deep correction, even neutral signals are fine
        if drawdown > 0.50:
            min_signal = min(min_signal, 0.05)

        if composite < min_signal:
            return None

        # --- Reserve management ---
        available_after_reserve = max(0, eur_available - self._RESERVE_EUR)
        if available_after_reserve < self.min_eur:
            # Check if we should dip into reserve (extreme opportunity)
            if drawdown > 0.65 and composite > 0.30:
                available_after_reserve = eur_available  # Use everything
                logger.info(
                    f"🔥 DEEP VALUE: Using reserve! Drawdown {drawdown:.1%}, "
                    f"composite {composite:+.2f}"
                )
            else:
                return None  # Preserve reserve

        # --- Price tier position sizing ---
        tier_weight = self._get_tier_weight(drawdown)

        # Cycle multiplier (accumulation mode: heavier)
        cycle_multiplier = {
            CyclePhase.ACCUMULATION: 3.0,   # Go big at the bottom
            CyclePhase.CORRECTION: 1.5,     # Solid buys in correction
            CyclePhase.GROWTH: 0.8,         # Lighter in growth
            CyclePhase.EUPHORIA: 0.15,      # Tiny buys near top (if at all)
        }[cycle_phase]

        # Regime multiplier (less impact than profit engine)
        regime_multiplier = {
            MacroRegime.RISK_ON: 1.2,
            MacroRegime.NEUTRAL: 1.0,
            MacroRegime.RISK_OFF: 0.7,      # Still buy, just smaller
            MacroRegime.CRISIS: 0.5,         # Still buy! But careful
        }[regime]

        # Signal strength adds 0-50% more
        signal_bonus = max(0, (composite - min_signal) * 2)  # 0 to ~1.8

        # Position sizing: tier weight IS the base allocation
        # (Don't multiply tier × base_pct — that double-shrinks small accounts)
        base_eur = available_after_reserve * tier_weight  # e.g. €257 × 25% = €64
        position_eur = base_eur * cycle_multiplier * regime_multiplier * (1 + signal_bonus)

        # Cap: don't exceed tier allocation (ladder discipline)
        position_eur = min(position_eur, base_eur)

        # Absolute cap
        position_eur = min(position_eur, available_after_reserve * self.max_position_pct)

        # Floor: if we have enough capital, at least do the minimum trade
        if position_eur < self.min_eur and available_after_reserve >= self.min_eur:
            # Force minimum trade if signal is there and we can afford it
            position_eur = self.min_eur

        if position_eur < self.min_eur:
            return None

        btc_amount = position_eur / price

        # --- Averaging down ---
        # In accumulation mode: we WANT to average down. No penalty.
        # Only gate: composite must be > 0.05 (not actively bearish)
        if btc_held > 0 and avg_buy_price > 0:
            current_pnl = (price - avg_buy_price) / avg_buy_price
            if current_pnl < -0.10:
                if composite < 0.05:
                    return None
                # No position reduction — accumulation means we avg down willingly

        # Build reasoning
        tier_name = self._get_tier_name(drawdown)
        new_avg = self._calc_new_avg(avg_buy_price, btc_held, price, btc_amount)

        reasoning_parts = [
            f"tier={tier_name}",
            f"drawdown={drawdown:.0%}",
            f"composite={composite:+.2f}",
            f"€{position_eur:.0f}",
            f"regime={regime.value}",
        ]

        if avg_buy_price > 0 and btc_held > 0:
            reasoning_parts.append(f"avg €{avg_buy_price:,.0f}→€{new_avg:,.0f}")

        remaining = eur_available - position_eur
        reasoning_parts.append(f"remaining=€{remaining:.0f}")

        # Update cooldown tracking
        import time as _time
        self._last_buy_price = price
        self._last_buy_time = _time.time()

        return TradingDecision(
            should_buy=True,
            should_sell=False,
            btc_amount=btc_amount,
            price=price,
            reasoning=f"ACCUMULATE: {', '.join(reasoning_parts)}",
            risk_level=risk_level,
            phase=cycle_phase.value,
            confidence=composite * 100,
            signals=signals,
        )

    # ========================================================================
    # PRICE TIER HELPERS
    # ========================================================================

    def _get_tier_weight(self, drawdown: float) -> float:
        """Get capital allocation weight for current drawdown tier."""
        for min_dd, max_dd, _, weight in self._PRICE_TIERS:
            if min_dd <= drawdown < max_dd:
                return weight
        # Below 30% drawdown = very light buying
        if drawdown < 0.30:
            return 0.08
        return 0.15  # Fallback

    def _get_tier_name(self, drawdown: float) -> str:
        """Get human-readable tier name."""
        for min_dd, max_dd, name, _ in self._PRICE_TIERS:
            if min_dd <= drawdown < max_dd:
                return name
        if drawdown < 0.30:
            return "near_ath"
        return "unknown"

    @staticmethod
    def _calc_new_avg(current_avg: float, current_btc: float,
                      new_price: float, new_btc: float) -> float:
        """Calculate what the new average buy price would be after this trade."""
        if current_avg <= 0 or current_btc <= 0:
            return new_price
        total_cost = (current_avg * current_btc) + (new_price * new_btc)
        total_btc = current_btc + new_btc
        return total_cost / total_btc if total_btc > 0 else new_price

    def get_accumulation_status(self, price: float, eur_available: float) -> dict:
        """Get detailed accumulation status including cooldown."""
        import time as _time
        drawdown = (self._CYCLE_ATH - price) / self._CYCLE_ATH
        tier_name = self._get_tier_name(drawdown)
        tier_weight = self._get_tier_weight(drawdown)
        available_after_reserve = max(0, eur_available - self._RESERVE_EUR)

        # Cooldown status
        if self._last_buy_price > 0:
            hours_since = (_time.time() - self._last_buy_time) / 3600
            price_drop = (self._last_buy_price - price) / self._last_buy_price
            cooldown_remaining = max(0, self._MIN_HOURS_BETWEEN_BUYS - hours_since)
            price_needed = self._last_buy_price * (1 - self._MIN_PRICE_DROP_FOR_REBUY)
        else:
            hours_since = 999
            price_drop = 0
            cooldown_remaining = 0
            price_needed = 0

        return {
            'mode': 'ACCUMULATION',
            'drawdown': drawdown,
            'tier': tier_name,
            'tier_weight': tier_weight,
            'eur_available': eur_available,
            'eur_after_reserve': available_after_reserve,
            'reserve': self._RESERVE_EUR,
            'reserve_intact': eur_available > self._RESERVE_EUR,
            'max_buy_eur': available_after_reserve * tier_weight,
            'bullets_remaining': max(0, int(available_after_reserve / self.min_eur)),
            'cooldown_hours_left': cooldown_remaining,
            'cooldown_price_trigger': price_needed,
            'last_buy_price': self._last_buy_price,
            'hours_since_last_buy': hours_since if self._last_buy_price > 0 else None,
        }


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

# Also export as CycleAwareTrading for seamless main.py swap
CycleAwareTrading = AccumulationEngine


def example_usage():
    """Show accumulation engine in action."""
    engine = AccumulationEngine(total_capital=307.0)

    # Current situation: underwater, correction phase
    prices = [72000 - i * 150 for i in range(60)]

    decision = engine.decide(
        price=60000,
        btc_held=0.0433,
        eur_available=307,
        avg_buy_price=77000,
        prices_1h=prices,
    )

    print(f"\nDecision: {'BUY' if decision.should_buy else 'SELL' if decision.should_sell else 'HOLD'}")
    print(f"Reasoning: {decision.reasoning}")

    # Show accumulation status
    status = engine.get_accumulation_status(60000, 307)
    print(f"\nAccumulation Status:")
    print(f"  Tier: {status['tier']} (weight: {status['tier_weight']})")
    print(f"  Drawdown: {status['drawdown']:.0%}")
    print(f"  Available: €{status['eur_after_reserve']:.0f} (reserve: €{status['reserve']:.0f})")
    print(f"  Max buy: €{status['max_buy_eur']:.0f}")
    print(f"  Bullets remaining: {status['bullets_remaining']}")


if __name__ == "__main__":
    example_usage()
