"""
Macro-Aware Profit Engine - Active Trading for the New Bitcoin Regime

PHILOSOPHY:
Bitcoin no longer trades on the 4-year halving cycle alone. As of 2025-2026,
the dominant price drivers are (in order):
  1. ETF flows & institutional positioning
  2. Fed policy, global liquidity, real interest rates
  3. Macro correlations (SPY, DXY, Gold)
  4. Technical momentum & mean reversion
  5. Halving cycle (context, not driver)

This module replaces the accumulation-only CycleAwareTrading with an active
profit engine that generates BOTH buy and sell signals using a multi-signal
architecture.

DESIGN (Ousterhout):
- Same simple interface as before: decide() → TradingDecision
- All complexity hidden: macro regime, technical signals, position management
- Deep module: ~600 lines of implementation behind a 4-parameter interface
- Backward compatible: TradingDecision dataclass unchanged

SIGNAL ARCHITECTURE:
  MacroRegime (context) → adjusts aggression and bias
  TechnicalSignals (timing) → generates entry/exit triggers
  CycleContext (background) → provides floor/ceiling awareness
  PositionManager (risk) → sizes positions, manages stops

USAGE:
    engine = ProfitEngine()
    decision = engine.decide(
        price=65000,
        btc_held=0.05,
        eur_available=2000,
        avg_buy_price=72000,
        prices_1h=recent_hourly_closes,   # Optional: enables technical signals
        volumes_1h=recent_hourly_volumes,  # Optional: enables volume analysis
    )
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from enum import Enum
import numpy as np

from logger_config import logger


# ============================================================================
# PUBLIC INTERFACE (unchanged from CycleAwareTrading for backward compat)
# ============================================================================

@dataclass
class TradingDecision:
    """
    A trading decision with all necessary information.
    Same interface as before — drop-in replacement.
    """
    should_buy: bool
    should_sell: bool
    btc_amount: float
    price: float
    reasoning: str
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'EXTREME'

    # Metadata for logging
    phase: Optional[str] = None
    confidence: Optional[float] = None
    signals: Optional[Dict] = field(default_factory=dict)


# ============================================================================
# ENUMS
# ============================================================================

class MacroRegime(Enum):
    """Macro environment classification."""
    RISK_ON = "risk_on"          # Liquidity expanding, risk appetite high
    NEUTRAL = "neutral"          # Mixed signals, range-bound
    RISK_OFF = "risk_off"        # Tightening, risk aversion
    CRISIS = "crisis"            # Acute stress, correlations spike to 1


class TrendState(Enum):
    """Price trend classification from technicals."""
    STRONG_UP = "strong_uptrend"
    UP = "uptrend"
    NEUTRAL = "neutral"
    DOWN = "downtrend"
    STRONG_DOWN = "strong_downtrend"


class CyclePhase(Enum):
    """Bitcoin cycle phase (background context only)."""
    ACCUMULATION = "accumulation"
    GROWTH = "growth"
    EUPHORIA = "euphoria"
    CORRECTION = "correction"


# ============================================================================
# PROFIT ENGINE
# ============================================================================

class ProfitEngine:
    """
    Deep module: Active profit-seeking trading engine.

    WHAT CHANGED FROM CycleAwareTrading:
    - Sells actively (not just at 200%+ profit)
    - Uses technical indicators for entry/exit timing
    - Adapts to macro regime (risk-on vs risk-off)
    - Dynamic take-profit and stop-loss levels
    - Trend following in strong moves, mean reversion in ranges
    - Cycle phase is context, not the sole decision driver

    INTERFACE: Same as before — decide() returns TradingDecision
    """

    # Historical cycle data (context only, not primary driver)
    _HALVING_DATE = datetime(2024, 4, 19)
    _CYCLE_ATH = 114395.0       # EUR, October 2025
    _CYCLE_ATH_DATE = datetime(2025, 10, 6)
    _PREVIOUS_LOW = 14069.0     # EUR, November 2022

    def __init__(self,
                 base_position_pct: float = 0.10,
                 min_eur_per_trade: float = 15.0,
                 max_position_pct: float = 0.50,
                 take_profit_pct: float = 0.08,
                 stop_loss_pct: float = 0.04,
                 enable_shorting: bool = False):
        """
        Initialize profit engine.

        Args:
            base_position_pct: Base position size as fraction of capital
            min_eur_per_trade: Minimum trade value in EUR
            max_position_pct: Maximum capital in a single trade
            take_profit_pct: Default take-profit target (adjusted dynamically)
            stop_loss_pct: Default stop-loss (adjusted dynamically)
            enable_shorting: Whether to generate sell signals without position
                            (False for spot-only Bitvavo)
        """
        self.base_position_pct = base_position_pct
        self.min_eur = min_eur_per_trade
        self.max_position_pct = max_position_pct
        self.base_take_profit = take_profit_pct
        self.base_stop_loss = stop_loss_pct
        self.enable_shorting = enable_shorting

        # Signal weights (sum to 1.0)
        self._weights = {
            'technical': 0.45,    # RSI, MACD, Bollinger, VWAP
            'momentum': 0.25,     # Price momentum & trend
            'cycle': 0.15,        # Cycle phase context
            'volatility': 0.15,   # Vol regime adjustment
        }

        # Track state for adaptive behavior
        self._recent_decisions: List[str] = []
        self._win_streak = 0
        self._loss_streak = 0

        logger.info(
            f"ProfitEngine initialized: base={base_position_pct:.0%}, "
            f"TP={take_profit_pct:.0%}, SL={stop_loss_pct:.0%}"
        )

    # ========================================================================
    # PUBLIC INTERFACE
    # ========================================================================

    def decide(self,
               price: float,
               btc_held: float,
               eur_available: float,
               avg_buy_price: float = 0.0,
               prices_1h: Optional[List[float]] = None,
               volumes_1h: Optional[List[float]] = None,
               macro_data: Optional[Dict] = None) -> TradingDecision:
        """
        Make an active trading decision.

        Args:
            price: Current BTC price in EUR
            btc_held: BTC balance
            eur_available: Available EUR balance
            avg_buy_price: Average purchase price (0 if no position)
            prices_1h: Recent hourly close prices (most recent last)
            volumes_1h: Recent hourly volumes (most recent last)
            macro_data: Optional dict with macro indicators
                        (correlations, sentiment, etf_flows, etc.)

        Returns:
            TradingDecision with active buy/sell signals
        """
        # Sanitize inputs
        price = max(0.01, price)
        btc_held = max(0.0, btc_held)
        eur_available = max(0.0, eur_available)

        # Build signal composite
        signals = self._compute_signals(price, prices_1h, volumes_1h, macro_data)

        # Determine macro regime
        regime = self._detect_regime(price, prices_1h, macro_data)

        # Determine cycle context
        cycle_phase = self._detect_cycle_phase(price)

        # Check exit conditions first (protect capital)
        if btc_held > 0 and avg_buy_price > 0:
            exit_decision = self._check_exit(
                price, btc_held, avg_buy_price, signals, regime, cycle_phase
            )
            if exit_decision:
                return exit_decision

        # Check entry conditions
        entry_decision = self._check_entry(
            price, eur_available, btc_held, avg_buy_price,
            signals, regime, cycle_phase
        )
        if entry_decision:
            return entry_decision

        # Default: HOLD
        return TradingDecision(
            should_buy=False,
            should_sell=False,
            btc_amount=0.0,
            price=price,
            reasoning=f"HOLD — {regime.value}, {cycle_phase.value}, composite={signals.get('composite', 0):.2f}",
            risk_level=self._regime_to_risk(regime),
            phase=cycle_phase.value,
            confidence=abs(signals.get('composite', 0)) * 100,
            signals=signals,
        )

    def get_market_context(self, price: float,
                           prices_1h: Optional[List[float]] = None,
                           volumes_1h: Optional[List[float]] = None,
                           macro_data: Optional[Dict] = None) -> dict:
        """Get rich market context for logging/monitoring."""
        signals = self._compute_signals(price, prices_1h, volumes_1h, macro_data)
        regime = self._detect_regime(price, prices_1h, macro_data)
        cycle_phase = self._detect_cycle_phase(price)
        drawdown = (self._CYCLE_ATH - price) / self._CYCLE_ATH

        return {
            'price': price,
            'regime': regime.value,
            'cycle_phase': cycle_phase.value,
            'drawdown_from_ath': drawdown,
            'signals': signals,
            'risk_level': self._regime_to_risk(regime),
            'days_since_halving': (datetime.now() - self._HALVING_DATE).days,
            'days_since_ath': (datetime.now() - self._CYCLE_ATH_DATE).days,
        }

    # ========================================================================
    # SIGNAL COMPUTATION (private)
    # ========================================================================

    def _compute_signals(self,
                         price: float,
                         prices: Optional[List[float]],
                         volumes: Optional[List[float]],
                         macro_data: Optional[Dict]) -> Dict:
        """
        Compute composite signal from multiple sources.

        Returns dict with individual signals and composite score.
        Composite range: -1.0 (strong sell) to +1.0 (strong buy)
        """
        signals = {}

        # Technical signals (if we have price history)
        if prices and len(prices) >= 26:
            signals['rsi'] = self._rsi_signal(prices)
            signals['macd'] = self._macd_signal(prices)
            signals['bollinger'] = self._bollinger_signal(prices, price)
            signals['ma_cross'] = self._ma_crossover_signal(prices)

            tech_score = (
                signals['rsi'] * 0.30 +
                signals['macd'] * 0.30 +
                signals['bollinger'] * 0.20 +
                signals['ma_cross'] * 0.20
            )
            signals['technical'] = tech_score
        else:
            signals['technical'] = 0.0

        # Momentum signal
        if prices and len(prices) >= 10:
            signals['momentum'] = self._momentum_signal(prices)
        else:
            signals['momentum'] = 0.0

        # Cycle context signal
        signals['cycle'] = self._cycle_signal(price)

        # Volatility signal
        if prices and len(prices) >= 20:
            signals['volatility'] = self._volatility_signal(prices)
        else:
            signals['volatility'] = 0.0

        # Macro override (if available)
        if macro_data:
            signals['macro_sentiment'] = macro_data.get('sentiment', 0.0)
            signals['risk_off_prob'] = macro_data.get('risk_off_probability', 0.0)

        # Composite weighted score
        composite = (
            signals['technical'] * self._weights['technical'] +
            signals['momentum'] * self._weights['momentum'] +
            signals['cycle'] * self._weights['cycle'] +
            signals['volatility'] * self._weights['volatility']
        )

        # Apply macro dampener if risk-off
        risk_off = signals.get('risk_off_prob', 0.0)
        if risk_off > 0.5:
            # Dampen buy signals in risk-off, amplify sell signals
            if composite > 0:
                composite *= (1 - risk_off * 0.5)
            else:
                composite *= (1 + risk_off * 0.3)

        signals['composite'] = max(-1.0, min(1.0, composite))
        return signals

    def _rsi_signal(self, prices: List[float], period: int = 14) -> float:
        """
        RSI signal using proper Wilder smoothing.
        Returns: -1.0 (overbought/sell) to +1.0 (oversold/buy)
        """
        if len(prices) < period + 1:
            return 0.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        # Wilder smoothing (not SMA!)
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        # Convert RSI to signal
        if rsi <= 25:
            return 1.0      # Deeply oversold — strong buy
        elif rsi <= 30:
            return 0.7       # Oversold — buy
        elif rsi <= 40:
            return 0.3       # Approaching oversold
        elif rsi >= 75:
            return -1.0      # Deeply overbought — strong sell
        elif rsi >= 70:
            return -0.7      # Overbought — sell
        elif rsi >= 60:
            return -0.3      # Approaching overbought
        else:
            return 0.0        # Neutral zone

    def _macd_signal(self, prices: List[float]) -> float:
        """
        MACD crossover signal.
        Returns: -1.0 to +1.0
        """
        if len(prices) < 26:
            return 0.0

        prices_arr = np.array(prices, dtype=float)

        # EMA calculations
        ema12 = self._ema(prices_arr, 12)
        ema26 = self._ema(prices_arr, 26)
        macd_line = ema12 - ema26
        signal_line = self._ema(macd_line, 9)

        macd_current = macd_line[-1]
        signal_current = signal_line[-1]
        histogram = macd_current - signal_current

        # Previous values for crossover detection
        macd_prev = macd_line[-2] if len(macd_line) > 1 else macd_current
        signal_prev = signal_line[-2] if len(signal_line) > 1 else signal_current

        # Bullish crossover
        if macd_prev <= signal_prev and macd_current > signal_current:
            return 0.8
        # Bearish crossover
        elif macd_prev >= signal_prev and macd_current < signal_current:
            return -0.8

        # Use histogram magnitude for trend strength
        # Normalize by price to make it comparable across price levels
        avg_price = np.mean(prices_arr[-10:])
        if avg_price > 0:
            normalized_hist = histogram / avg_price * 100
            return max(-0.5, min(0.5, normalized_hist * 10))

        return 0.0

    def _bollinger_signal(self, prices: List[float], current_price: float,
                          period: int = 20, std_dev: float = 2.0) -> float:
        """
        Bollinger Band mean-reversion signal.
        Returns: -1.0 (at upper band, sell) to +1.0 (at lower band, buy)
        """
        if len(prices) < period:
            return 0.0

        window = prices[-period:]
        middle = np.mean(window)
        std = np.std(window)

        if std == 0:
            return 0.0

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        band_width = upper - lower

        if band_width == 0:
            return 0.0

        # Position within bands: -1 (lower) to +1 (upper)
        position = (current_price - lower) / band_width * 2 - 1

        # Mean reversion: buy at lower band, sell at upper
        # But also respect breakouts (don't fade strong moves)
        if position <= -0.8:
            return 0.8    # Near/below lower band — buy
        elif position >= 0.8:
            return -0.8   # Near/above upper band — sell
        elif position <= -0.5:
            return 0.4
        elif position >= 0.5:
            return -0.4
        else:
            return 0.0

    def _ma_crossover_signal(self, prices: List[float]) -> float:
        """
        Moving average crossover (20/50 EMA).
        Returns: -1.0 to +1.0
        """
        if len(prices) < 50:
            return 0.0

        prices_arr = np.array(prices, dtype=float)
        ema20 = self._ema(prices_arr, 20)
        ema50 = self._ema(prices_arr, 50)

        current_diff = ema20[-1] - ema50[-1]
        prev_diff = ema20[-2] - ema50[-2] if len(ema20) > 1 else current_diff

        # Fresh crossover
        if prev_diff <= 0 and current_diff > 0:
            return 0.9   # Golden cross
        elif prev_diff >= 0 and current_diff < 0:
            return -0.9  # Death cross

        # Existing trend strength
        avg_price = np.mean(prices_arr[-10:])
        if avg_price > 0:
            normalized = current_diff / avg_price * 100
            return max(-0.5, min(0.5, normalized * 5))

        return 0.0

    def _momentum_signal(self, prices: List[float]) -> float:
        """
        Multi-timeframe momentum.
        Returns: -1.0 to +1.0
        """
        if len(prices) < 10:
            return 0.0

        current = prices[-1]

        # Short-term momentum (3 periods)
        short_mom = (current - prices[-4]) / prices[-4] if len(prices) >= 4 else 0

        # Medium-term momentum (7 periods)
        med_mom = (current - prices[-8]) / prices[-8] if len(prices) >= 8 else 0

        # Rate of change acceleration
        if len(prices) >= 6:
            recent_roc = (prices[-1] - prices[-3]) / prices[-3]
            prior_roc = (prices[-3] - prices[-6]) / prices[-6]
            acceleration = recent_roc - prior_roc
        else:
            acceleration = 0

        # Weighted combination
        signal = short_mom * 5 * 0.4 + med_mom * 3 * 0.3 + acceleration * 10 * 0.3

        return max(-1.0, min(1.0, signal))

    def _cycle_signal(self, price: float) -> float:
        """
        Cycle context signal — provides bias, not timing.

        Deep in correction → bullish bias (accumulate)
        Near ATH → bearish bias (reduce)
        This is BACKGROUND context, not primary driver.
        """
        drawdown = (self._CYCLE_ATH - price) / self._CYCLE_ATH

        if drawdown >= 0.70:
            return 0.8     # Deep value zone
        elif drawdown >= 0.55:
            return 0.5     # Mid-correction opportunity
        elif drawdown >= 0.40:
            return 0.2     # Correction, cautious accumulation
        elif drawdown >= 0.20:
            return -0.1    # Recent from peak, slight caution
        elif drawdown >= 0.05:
            return -0.4    # Near peak, reduce exposure
        else:
            return -0.7    # At/near ATH, take profits

    def _volatility_signal(self, prices: List[float]) -> float:
        """
        Volatility regime signal.

        High vol → reduce position sizes (negative signal dampener)
        Low vol → can be more aggressive
        Mean-reverting vol → opportunity

        Returns: -0.5 to +0.5 (dampener, not primary signal)
        """
        if len(prices) < 20:
            return 0.0

        returns = np.diff(np.log(np.array(prices[-20:], dtype=float)))
        current_vol = np.std(returns)

        # Compare to longer-term vol if available
        if len(prices) >= 50:
            longer_returns = np.diff(np.log(np.array(prices[-50:], dtype=float)))
            avg_vol = np.std(longer_returns)
        else:
            avg_vol = current_vol

        if avg_vol == 0:
            return 0.0

        vol_ratio = current_vol / avg_vol

        # High vol regime: reduce aggression
        if vol_ratio > 2.0:
            return -0.5   # Very high vol — be defensive
        elif vol_ratio > 1.5:
            return -0.3   # Elevated vol
        elif vol_ratio < 0.5:
            return 0.3    # Low vol — potential breakout setup
        elif vol_ratio < 0.8:
            return 0.1    # Below average vol
        else:
            return 0.0

    # ========================================================================
    # REGIME & CYCLE DETECTION (private)
    # ========================================================================

    def _detect_regime(self,
                       price: float,
                       prices: Optional[List[float]],
                       macro_data: Optional[Dict]) -> MacroRegime:
        """
        Detect current macro regime.

        Uses available data: price action, macro indicators, correlations.
        Falls back to neutral if data is insufficient.
        """
        risk_off_score = 0.0

        # Price-based regime detection (always available)
        if prices and len(prices) >= 20:
            returns = np.diff(np.array(prices[-20:], dtype=float)) / np.array(prices[-20:-1], dtype=float)
            vol = np.std(returns)
            avg_return = np.mean(returns)

            # High vol + negative returns = risk-off
            if vol > 0.03 and avg_return < -0.005:
                risk_off_score += 0.4
            elif vol > 0.05:
                risk_off_score += 0.3
            # Sustained negative returns (even without extreme vol) = risk-off
            elif avg_return < -0.005:
                risk_off_score += 0.35
            # Mostly down candles = risk-off signal
            pct_negative = np.mean(returns < 0)
            if pct_negative > 0.70:
                risk_off_score += 0.15
            # Risk-on: stable positive returns, low vol
            if avg_return > 0.005 and vol < 0.02:
                risk_off_score -= 0.3

        # Macro data (if available)
        if macro_data:
            risk_off_score += macro_data.get('risk_off_probability', 0) * 0.4

            # DXY strength = risk-off for BTC
            dxy_corr = macro_data.get('correlations', {}).get('DXY', 0)
            if dxy_corr > 0.5:
                risk_off_score += 0.2

        # Classify
        if risk_off_score >= 0.7:
            return MacroRegime.CRISIS
        elif risk_off_score >= 0.35:
            return MacroRegime.RISK_OFF
        elif risk_off_score <= -0.2:
            return MacroRegime.RISK_ON
        else:
            return MacroRegime.NEUTRAL

    def _detect_cycle_phase(self, price: float) -> CyclePhase:
        """Detect cycle phase (context only)."""
        drawdown = (self._CYCLE_ATH - price) / self._CYCLE_ATH
        days_since_ath = (datetime.now() - self._CYCLE_ATH_DATE).days

        if drawdown > 0.65 and days_since_ath > 365:
            return CyclePhase.ACCUMULATION
        elif drawdown > 0.30:
            return CyclePhase.CORRECTION
        elif drawdown < 0.10:
            return CyclePhase.EUPHORIA
        else:
            return CyclePhase.GROWTH

    # ========================================================================
    # EXIT LOGIC (private)
    # ========================================================================

    def _check_exit(self,
                    price: float,
                    btc_held: float,
                    avg_buy_price: float,
                    signals: Dict,
                    regime: MacroRegime,
                    cycle_phase: CyclePhase) -> Optional[TradingDecision]:
        """
        Check if we should exit (sell) an existing position.

        EXIT RULES (checked in priority order):
        1. Stop-loss: Protect capital
        2. Take-profit: Lock in gains
        3. Signal-based exit: Technicals say sell
        4. Regime shift: Macro environment deteriorating
        """
        profit_pct = (price - avg_buy_price) / avg_buy_price
        composite = signals.get('composite', 0)
        risk_level = self._regime_to_risk(regime)

        # --- Dynamic stop-loss ---
        # Tighter stop in risk-off, wider in accumulation zone
        stop_loss = self.base_stop_loss
        if regime in [MacroRegime.RISK_OFF, MacroRegime.CRISIS]:
            stop_loss *= 0.7   # Tighter stop: 2.8% instead of 4%
        elif cycle_phase == CyclePhase.ACCUMULATION:
            stop_loss *= 2.5   # Much wider stop in deep value: 10%
        elif cycle_phase == CyclePhase.CORRECTION:
            # In correction: don't stop out at small losses
            # BUT do stop out if signals confirm continued decline
            if composite > -0.3:  # Signals not strongly bearish
                stop_loss *= 2.0  # Wider stop: 8%

        if profit_pct <= -stop_loss:
            # Exception: don't panic sell deep underwater positions
            # in accumulation zone when signals aren't screaming sell
            if profit_pct < -0.20 and cycle_phase in [CyclePhase.CORRECTION, CyclePhase.ACCUMULATION]:
                if composite > -0.5:
                    logger.info(
                        f"💎 Holding underwater position: {profit_pct:.1%} in {cycle_phase.value} "
                        f"(signals not confirming: {composite:.2f})"
                    )
                    return None

            sell_pct = 0.85 if regime == MacroRegime.CRISIS else 0.70
            return TradingDecision(
                should_buy=False,
                should_sell=True,
                btc_amount=btc_held * sell_pct,
                price=price,
                reasoning=f"STOP LOSS: {profit_pct:+.1%} (limit: -{stop_loss:.1%}, regime: {regime.value})",
                risk_level=risk_level,
                phase=cycle_phase.value,
                confidence=85,
                signals=signals,
            )

        # --- Dynamic take-profit ---
        take_profit = self.base_take_profit
        if regime == MacroRegime.RISK_ON:
            take_profit *= 1.5   # Let winners run: 12%
        elif regime == MacroRegime.RISK_OFF:
            take_profit *= 0.6   # Take profits early: 4.8%
        elif cycle_phase == CyclePhase.EUPHORIA:
            take_profit *= 0.5   # Very aggressive profit-taking near top

        if profit_pct >= take_profit:
            # Scale sell amount: sell more at higher profit
            if profit_pct > take_profit * 2:
                sell_pct = 0.60  # Take 60% off at 2x target
            elif profit_pct > take_profit * 1.5:
                sell_pct = 0.40  # Take 40% off at 1.5x target
            else:
                sell_pct = 0.25  # Take 25% off at target

            return TradingDecision(
                should_buy=False,
                should_sell=True,
                btc_amount=btc_held * sell_pct,
                price=price,
                reasoning=f"TAKE PROFIT: {profit_pct:+.1%} (target: +{take_profit:.1%}), selling {sell_pct:.0%}",
                risk_level=risk_level,
                phase=cycle_phase.value,
                confidence=80,
                signals=signals,
            )

        # --- Signal-based exit ---
        # Strong sell signal + in profit = sell
        if composite <= -0.6 and profit_pct > 0.01:
            sell_pct = min(0.50, abs(composite) * 0.5)
            return TradingDecision(
                should_buy=False,
                should_sell=True,
                btc_amount=btc_held * sell_pct,
                price=price,
                reasoning=f"SIGNAL EXIT: composite={composite:.2f}, profit={profit_pct:+.1%}",
                risk_level=risk_level,
                phase=cycle_phase.value,
                confidence=abs(composite) * 100,
                signals=signals,
            )

        # --- Regime deterioration sell ---
        # In crisis with any profit, reduce exposure
        if regime == MacroRegime.CRISIS and profit_pct > -0.02:
            return TradingDecision(
                should_buy=False,
                should_sell=True,
                btc_amount=btc_held * 0.30,
                price=price,
                reasoning=f"CRISIS EXIT: regime={regime.value}, profit={profit_pct:+.1%}",
                risk_level='EXTREME',
                phase=cycle_phase.value,
                confidence=70,
                signals=signals,
            )

        return None  # No exit signal

    # ========================================================================
    # ENTRY LOGIC (private)
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
        Check if we should enter (buy) a position.

        ENTRY RULES:
        1. Composite signal must be positive (buy bias)
        2. Position size scales with signal strength and regime
        3. In crisis: no new entries unless deeply oversold
        4. Cap total position exposure
        """
        composite = signals.get('composite', 0)
        risk_level = self._regime_to_risk(regime)

        # --- Minimum signal threshold ---
        min_signal = 0.20  # Need at least +0.20 composite to buy
        if regime == MacroRegime.RISK_OFF:
            min_signal = 0.40  # Higher bar in risk-off
        elif regime == MacroRegime.CRISIS:
            min_signal = 0.60  # Very high bar in crisis
        elif regime == MacroRegime.RISK_ON:
            min_signal = 0.15  # Lower bar in risk-on

        if composite < min_signal:
            return None  # Signal too weak

        # --- Position sizing ---
        # Base size adjusted by signal strength, regime, and cycle
        regime_multiplier = {
            MacroRegime.RISK_ON: 1.3,
            MacroRegime.NEUTRAL: 1.0,
            MacroRegime.RISK_OFF: 0.5,
            MacroRegime.CRISIS: 0.25,
        }[regime]

        cycle_multiplier = {
            CyclePhase.ACCUMULATION: 2.5,
            CyclePhase.CORRECTION: 1.2,
            CyclePhase.GROWTH: 1.0,
            CyclePhase.EUPHORIA: 0.3,
        }[cycle_phase]

        # Signal strength multiplier (0.2 → 1.0x, 1.0 → 2.5x)
        signal_multiplier = 1.0 + (composite - 0.2) * 1.875

        final_multiplier = regime_multiplier * cycle_multiplier * signal_multiplier
        position_eur = eur_available * self.base_position_pct * final_multiplier

        # Cap position size
        position_eur = min(position_eur, eur_available * self.max_position_pct)

        # Check minimum trade size
        if position_eur < self.min_eur:
            return None

        btc_amount = position_eur / price

        # --- Averaging down logic ---
        # If we already hold and are underwater, only buy if signal is strong
        if btc_held > 0 and avg_buy_price > 0:
            current_pnl = (price - avg_buy_price) / avg_buy_price
            if current_pnl < -0.10 and composite < 0.40:
                return None  # Don't average down on weak signals

        # Build reasoning
        reasoning_parts = [
            f"composite={composite:+.2f}",
            f"regime={regime.value}",
            f"cycle={cycle_phase.value}",
            f"size={final_multiplier:.1f}x",
        ]

        # Add strongest individual signal
        tech = signals.get('technical', 0)
        mom = signals.get('momentum', 0)
        if abs(tech) > abs(mom):
            reasoning_parts.append(f"tech={tech:+.2f}")
        else:
            reasoning_parts.append(f"momentum={mom:+.2f}")

        return TradingDecision(
            should_buy=True,
            should_sell=False,
            btc_amount=btc_amount,
            price=price,
            reasoning=f"BUY: {', '.join(reasoning_parts)}",
            risk_level=risk_level,
            phase=cycle_phase.value,
            confidence=composite * 100,
            signals=signals,
        )

    # ========================================================================
    # UTILITIES (private)
    # ========================================================================

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """Compute EMA using exponential weighting."""
        if len(data) < period:
            return data.copy()
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data, dtype=float)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema

    @staticmethod
    def _regime_to_risk(regime: MacroRegime) -> str:
        return {
            MacroRegime.RISK_ON: 'LOW',
            MacroRegime.NEUTRAL: 'MEDIUM',
            MacroRegime.RISK_OFF: 'HIGH',
            MacroRegime.CRISIS: 'EXTREME',
        }[regime]

    def record_trade_result(self, profit_pct: float):
        """Record trade outcome for adaptive behavior."""
        if profit_pct > 0:
            self._win_streak += 1
            self._loss_streak = 0
        else:
            self._loss_streak += 1
            self._win_streak = 0
        self._recent_decisions.append('win' if profit_pct > 0 else 'loss')
        # Keep only last 20
        self._recent_decisions = self._recent_decisions[-20:]


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

class CycleAwareTrading(ProfitEngine):
    """
    Backward-compatible alias.
    Drop-in replacement: same __init__ signature, same decide() interface.
    """
    def __init__(self, base_position_pct: float = 0.10,
                 min_eur_per_trade: float = 15.0):
        super().__init__(
            base_position_pct=base_position_pct,
            min_eur_per_trade=min_eur_per_trade,
        )


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_usage():
    """Demonstrate the profit engine."""
    engine = ProfitEngine(base_position_pct=0.10)

    # Simulated hourly prices (descending = bearish)
    prices = [72000 - i * 100 for i in range(50)]
    volumes = [100 + i * 2 for i in range(50)]

    decision = engine.decide(
        price=67000,
        btc_held=0.05,
        eur_available=2000,
        avg_buy_price=72000,
        prices_1h=prices,
        volumes_1h=volumes,
    )

    if decision.should_buy:
        print(f"BUY {decision.btc_amount:.8f} BTC @ €{decision.price:,.0f}")
    elif decision.should_sell:
        print(f"SELL {decision.btc_amount:.8f} BTC @ €{decision.price:,.0f}")
    else:
        print(f"HOLD: {decision.reasoning}")

    context = engine.get_market_context(67000, prices, volumes)
    print(f"Regime: {context['regime']}, Phase: {context['cycle_phase']}")
    print(f"Signals: {context['signals']}")


if __name__ == "__main__":
    example_usage()
