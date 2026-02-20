"""
Bitcoin Cycle-Aware Trading - Deep Module Design

PHILOSOPHY:
This module provides a simple interface for cycle-aware Bitcoin trading decisions
while hiding all the complexity of cycle analysis, phase detection, and position
sizing behind a clean abstraction.

Following Ousterhout's principles:
- Deep module: Simple interface, powerful implementation
- Information hiding: Cycle calculations are internal details
- General purpose: Works for any Bitcoin trading scenario
- Pull complexity downward: Users don't need to understand cycles
- Define errors out of existence: Always returns valid trading signals

USAGE:
    trading = CycleAwareTrading()
    decision = trading.decide(
        price=58000,
        btc_held=0.037,
        eur_available=1000,
        avg_buy_price=81000
    )
    
    if decision.should_buy:
        place_buy_order(decision.btc_amount, decision.price)
    elif decision.should_sell:
        place_sell_order(decision.btc_amount, decision.price)

IMPORTANT: All prices in EUR for BTC-EUR trading
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from enum import Enum
import logging

# Simple logger setup (replace with your logger_config import in production)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradingDecision:
    """
    A trading decision with all necessary information.
    
    This is the ONLY interface users need - they don't need to understand
    cycles, phases, drawdowns, or any internal complexity.
    """
    should_buy: bool
    should_sell: bool
    btc_amount: float
    price: float
    reasoning: str
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'EXTREME'
    
    # Optional metadata for logging/debugging
    phase: Optional[str] = None
    confidence: Optional[float] = None


class CycleAwareTrading:
    """
    Deep module: Simple interface for cycle-aware trading decisions.
    
    DESIGN RATIONALE:
    - Users call decide() and get a simple TradingDecision
    - All cycle complexity is hidden inside this module
    - No need to understand halvings, drawdowns, or phases
    - Always returns a valid decision (never fails)
    
    WHY THIS IS DEEP:
    - Interface: One method (decide) with 4 parameters
    - Implementation: ~400 lines of cycle analysis logic
    - Ratio: 1:100 interface-to-implementation
    
    WHY NOT SPLIT INTO MULTIPLE CLASSES:
    - Cycle detection and trading logic are fundamentally coupled
    - Splitting creates shallow modules with pass-through methods
    - Users would need to understand both modules to use either
    - Creates information leakage and temporal decomposition
    """
    
    # === HISTORICAL DATA (Hidden implementation detail) ===
    
    # Why these are class constants:
    # - Historical data doesn't change
    # - No reason to recalculate or pass around
    # - Pull complexity downward: users never see this
    
    _HALVING_DATE = datetime(2024, 4, 19)  # Current cycle (Era 4)
    _CYCLE_ATH = 114395.0  # EUR, October 2025
    _CYCLE_ATH_DATE = datetime(2025, 10, 6)
    _PREVIOUS_LOW = 14069.0  # EUR, November 2022
    
    # Historical bear market drawdowns (for context)
    # Why here: Used internally to estimate bottom, not exposed
    _HISTORICAL_DRAWDOWNS = [0.93, 0.88, 0.84, 0.78]  # 93%, 88%, 84%, 78%
    _AVG_DRAWDOWN = 0.80  # Conservative 80% estimate
    
    def __init__(self, 
                 base_position_pct: float = 0.10,
                 min_eur_per_trade: float = 15.0):
        """
        Initialize trading module.
        
        WHY THESE PARAMETERS:
        - base_position_pct: User's risk tolerance (what % to invest)
        - min_eur_per_trade: Exchange minimum (Bitvavo = €5, we use €15)
        
        WHY NOT MORE PARAMETERS:
        - More parameters = more complexity for users
        - Cycle thresholds are derived from historical data, not configurable
        - Position multipliers calculated automatically from phase
        - Stop losses derived from market conditions, not user settings
        
        DESIGN DECISION: Smart defaults over configuration
        """
        self.base_position_pct = base_position_pct
        self.min_eur = min_eur_per_trade
        
        # Pre-calculate derived values (pull complexity downward)
        self._estimated_bottom = self._CYCLE_ATH * (1 - self._AVG_DRAWDOWN)
        
        logger.info(f"Initialized cycle-aware trading (base position: {base_position_pct:.0%})")
    
    def decide(self,
               price: float,
               btc_held: float,
               eur_available: float,
               avg_buy_price: float = 0.0) -> TradingDecision:
        """
        Make a trading decision based on current market conditions.
        
        This is the ONLY method users need to call. Everything else is internal.
        
        WHY THIS INTERFACE:
        - price: What we need to know about the market
        - btc_held: What we own
        - eur_available: What we can invest
        - avg_buy_price: Our cost basis (0 if no position)
        
        WHY NOT EXPOSE:
        - Cycle phase (implementation detail)
        - Drawdown percentage (derived from price)
        - Position multiplier (calculated internally)
        - Stop loss percentage (determined by phase)
        
        RETURNS:
        - TradingDecision: Everything needed to execute the trade
        - Never returns None (define errors out of existence)
        - Never throws exceptions for market data (defensive)
        
        Args:
            price: Current BTC price in EUR
            btc_held: BTC balance
            eur_available: Available EUR balance
            avg_buy_price: Average purchase price (0 if no position)
            
        Returns:
            TradingDecision with all information needed to trade
        """
        # Validate inputs (but don't expose validation to user)
        price = max(0.01, price)  # Prevent division by zero
        btc_held = max(0.0, btc_held)
        eur_available = max(0.0, eur_available)
        
        # Analyze market (all complexity hidden)
        phase, confidence = self._detect_phase(price)
        risk_level = self._assess_risk(price, phase)
        
        # Check if we should sell first (stop loss or profit taking)
        if btc_held > 0 and avg_buy_price > 0:
            sell_decision = self._check_sell_conditions(
                price, btc_held, avg_buy_price, phase
            )
            if sell_decision:
                return sell_decision
        
        # Check if we should buy
        buy_decision = self._check_buy_conditions(
            price, eur_available, phase, confidence, risk_level
        )
        if buy_decision:
            return buy_decision
        
        # Default: HOLD
        return TradingDecision(
            should_buy=False,
            should_sell=False,
            btc_amount=0.0,
            price=price,
            reasoning=f"HOLD - {phase.value} phase, {risk_level} risk",
            risk_level=risk_level,
            phase=phase.value,
            confidence=confidence
        )
    
    # === PRIVATE IMPLEMENTATION (Hidden complexity) ===
    
    # Why these are private:
    # - Implementation details
    # - May change without affecting users
    # - Pull complexity downward
    # - Information hiding
    
    class _Phase(Enum):
        """
        Internal enum for cycle phases.
        
        Why private: Users don't need to know about phases.
        They just call decide() and get a decision.
        """
        ACCUMULATION = "accumulation"
        GROWTH = "growth"
        BUBBLE = "bubble"
        CORRECTION = "correction"
    
    def _detect_phase(self, price: float) -> tuple['CycleAwareTrading._Phase', float]:
        """
        Detect current cycle phase.
        
        WHY PRIVATE:
        - Implementation detail of how we analyze the market
        - Users don't need to understand phases
        - May change algorithm without affecting interface
        
        WHY COMBINED WITH CONFIDENCE:
        - Confidence affects position sizing
        - Keeping them together reduces method coupling
        - Returns tuple (phase, confidence) as atomic unit
        
        ALGORITHM RATIONALE:
        - Based on historical 4-year halving cycle patterns
        - Drawdown from ATH is most reliable phase indicator
        - Time since halving provides secondary confirmation
        - Confidence based on how clearly data matches pattern
        """
        days_since_halving = (datetime.now() - self._HALVING_DATE).days
        drawdown = (self._CYCLE_ATH - price) / self._CYCLE_ATH
        gain_from_low = (price - self._PREVIOUS_LOW) / self._PREVIOUS_LOW
        
        # CORRECTION: Significant drawdown from peak
        if drawdown > 0.30:  # >30% down from ATH
            confidence = min(90, 60 + (drawdown * 50))  # Higher confidence with deeper drop
            return self._Phase.CORRECTION, confidence
        
        # BUBBLE: Near peak timing and price
        if 350 <= days_since_halving <= 600 and drawdown < 0.30 and gain_from_low > 3.0:
            confidence = 80  # High confidence near historical peak window
            return self._Phase.BUBBLE, confidence
        
        # ACCUMULATION: Deep correction, time has passed
        days_since_ath = (datetime.now() - self._CYCLE_ATH_DATE).days
        if drawdown > 0.70 and days_since_ath > 365:
            confidence = 85
            return self._Phase.ACCUMULATION, confidence
        
        # GROWTH: Everything else (default)
        confidence = 60
        return self._Phase.GROWTH, confidence
    
    def _assess_risk(self, price: float, phase: '_Phase') -> str:
        """
        Assess current risk level.
        
        WHY PRIVATE:
        - Risk assessment is internal to decision making
        - User gets risk level in TradingDecision, doesn't need this method
        
        WHY SEPARATE FROM PHASE DETECTION:
        - Risk depends on price AND phase
        - Single responsibility: phase detection vs risk assessment
        - But both are private - user never calls either directly
        """
        drawdown = (self._CYCLE_ATH - price) / self._CYCLE_ATH
        
        if phase == self._Phase.BUBBLE:
            return 'EXTREME'  # Top is most dangerous time
        elif phase == self._Phase.CORRECTION:
            # Risk decreases as we fall deeper (more opportunity, less danger)
            if drawdown > 0.70:
                return 'LOW'  # Deep correction = accumulation opportunity
            elif drawdown > 0.50:
                return 'MEDIUM'
            else:
                return 'HIGH'  # Early correction = likely more downside
        elif phase == self._Phase.ACCUMULATION:
            return 'LOW'  # Accumulation zone is low risk
        else:  # GROWTH
            return 'MEDIUM'
    
    def _check_sell_conditions(self,
                               price: float,
                               btc_held: float,
                               avg_buy_price: float,
                               phase: '_Phase') -> Optional[TradingDecision]:
        """
        Check if we should sell based on stop loss or profit taking.
        
        WHY THIS LOGIC:
        - Stop losses prevent catastrophic losses
        - BUT: Don't stop out of underwater positions in bear markets
        - RATIONALE: Selling at -28% in a bear is panic selling
        - BETTER: Hold through bear, accumulate more, profit on recovery
        
        DESIGN DECISION:
        - No stop loss for positions >15% underwater in CORRECTION/ACCUMULATION
        - WHY: These are old positions from before cycle awareness
        - GOAL: Average down, not panic sell at the bottom
        """
        profit_pct = (price - avg_buy_price) / avg_buy_price
        
        # Deep underwater positions in bear market: HOLD, don't sell
        # RATIONALE: Selling at -28% locks in losses, prevents recovery
        if profit_pct < -0.15 and phase in [self._Phase.CORRECTION, self._Phase.ACCUMULATION]:
            logger.info(f"💎 Holding underwater position: {profit_pct:.1%} loss in {phase.value}")
            return None
        
        # Stop loss for smaller losses (protect capital)
        if profit_pct < -0.05:  # -5% stop loss
            return TradingDecision(
                should_buy=False,
                should_sell=True,
                btc_amount=btc_held * 0.90,  # Keep 10% for recovery
                price=price,
                reasoning=f"Stop loss: {profit_pct:.1%} (limit: -5%)",
                risk_level=self._assess_risk(price, phase),
                phase=phase.value
            )
        
        # Profit taking in bubble phase
        if phase == self._Phase.BUBBLE and profit_pct > 2.0:  # 200% profit
            return TradingDecision(
                should_buy=False,
                should_sell=True,
                btc_amount=btc_held * 0.60,  # Take 60% profit, ride 40%
                price=price,
                reasoning=f"Profit taking: {profit_pct:.1%} gain in bubble phase",
                risk_level='EXTREME',
                phase=phase.value
            )
        
        return None  # No sell signal
    
    def _check_buy_conditions(self,
                             price: float,
                             eur_available: float,
                             phase: '_Phase',
                             confidence: float,
                             risk_level: str) -> Optional[TradingDecision]:
        """
        Check if we should buy based on cycle phase and drawdown.
        
        WHY THIS LOGIC:
        - Buy more aggressively as price falls
        - Scale position size with conviction
        - Preserve capital in dangerous phases
        
        POSITION SIZING RATIONALE:
        - Bubble (top): 0.2x = minimize exposure
        - Correction (falling): 0.5x = cautious, wait for bottom
        - Growth (rising): 1.5x = moderate participation
        - Accumulation (bottom): 3.0x = maximum aggression
        """
        drawdown = (self._CYCLE_ATH - price) / self._CYCLE_ATH
        
        # Determine position multiplier based on phase
        if phase == self._Phase.BUBBLE:
            multiplier = 0.2  # Minimal exposure near peak
        elif phase == self._Phase.CORRECTION:
            # Scale up as we fall deeper
            if drawdown > 0.70:
                multiplier = 2.0  # Aggressive near historical bottom
            elif drawdown > 0.50:
                multiplier = 1.0  # Moderate in mid-correction
            else:
                multiplier = 0.35  # Cautious early in correction
        elif phase == self._Phase.ACCUMULATION:
            multiplier = 3.0  # Maximum aggression at the bottom
        else:  # GROWTH
            multiplier = 1.5  # Moderate participation
        
        # Calculate position size
        position_eur = eur_available * self.base_position_pct * multiplier
        
        # Cap at 50% of available (preserve capital)
        position_eur = min(position_eur, eur_available * 0.5)
        
        # Check minimum trade size
        if position_eur < self.min_eur:
            return None  # Too small to trade
        
        # Additional filter: Don't buy early in correction
        # RATIONALE: Wait for deeper drawdown before aggressive buying
        if phase == self._Phase.CORRECTION and drawdown < 0.40:
            # Only light buying if any
            if position_eur < eur_available * 0.05:  # Less than 5% of capital
                pass  # Allow very light buying
            else:
                return None  # Wait for better prices
        
        # Calculate BTC amount
        btc_amount = position_eur / price
        
        # Build reasoning
        reasoning_parts = []
        if phase == self._Phase.CORRECTION:
            reasoning_parts.append(f"{drawdown:.1%} drawdown")
            if drawdown > 0.70:
                reasoning_parts.append("approaching historical bottom")
            elif drawdown > 0.50:
                reasoning_parts.append("mid-correction")
            else:
                reasoning_parts.append("early correction, cautious")
        else:
            reasoning_parts.append(f"{phase.value} phase")
        
        reasoning_parts.append(f"{multiplier:.1f}x position")
        reasoning = "BUY: " + ", ".join(reasoning_parts)
        
        return TradingDecision(
            should_buy=True,
            should_sell=False,
            btc_amount=btc_amount,
            price=price,
            reasoning=reasoning,
            risk_level=risk_level,
            phase=phase.value,
            confidence=confidence
        )
    
    def get_market_context(self, price: float) -> dict:
        """
        Get current market context for logging/monitoring.
        
        WHY PUBLIC:
        - Useful for displaying status to user
        - Not required for trading (decide() is sufficient)
        - Convenience method, not core functionality
        
        WHY NOT IN decide():
        - decide() returns TradingDecision (what to do)
        - This returns analysis (what's happening)
        - Different purposes, different methods
        - Users who just want to trade don't need this
        """
        phase, confidence = self._detect_phase(price)
        drawdown = (self._CYCLE_ATH - price) / self._CYCLE_ATH
        days_since_halving = (datetime.now() - self._HALVING_DATE).days
        
        return {
            'phase': phase.value,
            'confidence': confidence,
            'price': price,
            'drawdown_pct': drawdown,
            'days_since_halving': days_since_halving,
            'estimated_bottom': self._estimated_bottom,
            'risk_level': self._assess_risk(price, phase),
        }


# === USAGE EXAMPLES ===

def example_usage():
    """
    Show how simple the interface is.
    
    WHY THIS IS GOOD DESIGN:
    - User needs 4 lines of code to make a trading decision
    - No need to understand cycles, phases, drawdowns
    - All complexity hidden behind simple decide() method
    """
    # Initialize once
    trading = CycleAwareTrading(base_position_pct=0.10)
    
    # Make decisions
    decision = trading.decide(
        price=58000,
        btc_held=0.037,
        eur_available=1000,
        avg_buy_price=81000
    )
    
    # Act on decision
    if decision.should_buy:
        print(f"Buy {decision.btc_amount:.8f} BTC @ €{decision.price:,.0f}")
        print(f"Reason: {decision.reasoning}")
    elif decision.should_sell:
        print(f"Sell {decision.btc_amount:.8f} BTC @ €{decision.price:,.0f}")
        print(f"Reason: {decision.reasoning}")
    else:
        print(f"Hold - {decision.reasoning}")
    
    # Optional: Get context for logging
    context = trading.get_market_context(58000)
    print(f"Phase: {context['phase']}, Risk: {context['risk_level']}")


if __name__ == "__main__":
    example_usage()
