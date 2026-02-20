"""
Bitcoin Cycle Detector - Leverage 4-year halving cycles for better trading decisions

⚠️ IMPORTANT: All prices in EUR (designed for BTC-EUR trading on Bitvavo)
Historical USD data converted at ~1.10 USD/EUR exchange rate

Based on historical analysis:
- Halvings occur every ~4 years (210,000 blocks)
- Major peaks typically 12-18 months (368-549 days) post-halving
- Bear markets drop 70-85% from peak
- Each cycle low never revisits previous cycle's low

Historical Cycle Data (EUR):
- 2011 Peak: €28 → Low: €1.95 (-93%)
- 2013 Peak: €1,129 → Low: €136 (-88%)
- 2017 Peak: €17,877 → Low: €2,838 (-84%)
- 2021 Peak: €62,767 → Low: €14,069 (-78%)
- 2025 Peak: €114,395 → Current: €59,000 (-48%)
"""

from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from enum import Enum
from logger_config import logger


class CyclePhase(Enum):
    """Bitcoin market cycle phases"""
    ACCUMULATION = "accumulation"  # Bear bottom, sideways, low volatility
    GROWTH = "growth"              # Recovery, steady uptrend
    BUBBLE = "bubble"              # Euphoria, exponential growth, near peak
    CORRECTION = "correction"      # Bear market, major drawdown


class BitcoinCycleDetector:
    """
    Detects Bitcoin cycle phase and provides cycle-aware trading signals
    """
    
    # Historical halving dates
    HALVINGS = {
        1: datetime(2012, 11, 28),  # 50 → 25 BTC
        2: datetime(2016, 7, 9),    # 25 → 12.5 BTC
        3: datetime(2020, 5, 11),   # 12.5 → 6.25 BTC
        4: datetime(2024, 4, 19),   # 6.25 → 3.125 BTC (CURRENT ERA)
        5: datetime(2028, 4, 15),   # Estimated next halving
    }
    
    # Historical cycle peaks (days post-halving) - IN EUR
    # Note: Converted from USD at approximate historical exchange rates
    HISTORICAL_PEAKS = {
        1: {'days': 368, 'price': 1129},      # 2013 (~$1,242 @ 1.10)
        2: {'days': 525, 'price': 17877},     # 2017 (~$19,665 @ 1.10)
        3: {'days': 549, 'price': 62767},     # 2021 (~$69,044 @ 1.10)
        4: {'days': 481, 'price': 114395},    # 2025 (~$125,835 @ 1.10) (current cycle)
    }
    
    # Historical cycle lows (absolute bottoms) - IN EUR
    HISTORICAL_LOWS = {
        1: {'date': datetime(2011, 11, 18), 'price': 1.95},      # ~$2.14 @ 1.10
        2: {'date': datetime(2015, 1, 14), 'price': 136},        # ~$150 @ 1.10
        3: {'date': datetime(2018, 12, 15), 'price': 2838},      # ~$3,122 @ 1.10
        4: {'date': datetime(2022, 11, 21), 'price': 14069},     # ~$15,476 @ 1.10
    }
    
    # Never-look-back prices (cycle lows that are never revisited) - IN EUR
    NEVER_LOOK_BACK_PRICES = [1.95, 136, 2838, 14069]
    
    def __init__(self, current_cycle: int = 4):
        """
        Initialize cycle detector
        
        Args:
            current_cycle: Current halving cycle number (4 = 2024 halving)
        """
        self.current_cycle = current_cycle
        self.current_halving_date = self.HALVINGS[current_cycle]
        self.next_halving_date = self.HALVINGS.get(current_cycle + 1)
        
        # Current cycle stats (IN EUR - for BTC-EUR pairs on Bitvavo)
        self.cycle_ath = 114395  # October 2025 ATH (~$125,835 @ 1.10 USD/EUR)
        self.cycle_ath_date = datetime(2025, 10, 6)
        self.cycle_low = 14069   # Previous cycle bottom Nov 2022 (~$15,476 @ 1.10)
        
        logger.info(f"Initialized Cycle Detector for Era {current_cycle}")
        logger.info(f"  Halving Date: {self.current_halving_date.date()}")
        logger.info(f"  Cycle ATH: €{self.cycle_ath:,}")
        logger.info(f"  Cycle Low: €{self.cycle_low:,}")
    
    def get_days_since_halving(self) -> int:
        """Calculate days since current halving"""
        delta = datetime.now() - self.current_halving_date
        return delta.days
    
    def get_days_since_ath(self) -> int:
        """Calculate days since cycle ATH"""
        delta = datetime.now() - self.cycle_ath_date
        return delta.days
    
    def estimate_cycle_phase(self, current_price: float) -> Tuple[CyclePhase, float]:
        """
        Estimate current cycle phase based on multiple signals
        
        Returns:
            (CyclePhase, confidence_score)
        """
        days_since_halving = self.get_days_since_halving()
        days_since_ath = self.get_days_since_ath()
        
        # Calculate key metrics
        drawdown_from_ath = (self.cycle_ath - current_price) / self.cycle_ath
        gain_from_cycle_low = (current_price - self.cycle_low) / self.cycle_low
        
        # Phase detection logic
        confidence = 0.0
        
        # BUBBLE PHASE: Near historical peak timing (350-600 days post-halving)
        if 350 <= days_since_halving <= 600 and drawdown_from_ath < 0.30:
            if gain_from_cycle_low > 3.0:  # More than 3x from cycle low
                return CyclePhase.BUBBLE, 0.9
        
        # CORRECTION PHASE: Major drawdown from ATH
        if drawdown_from_ath > 0.30:
            confidence = min(0.9, drawdown_from_ath / 0.70)  # Higher confidence as we approach 70% drop
            
            # Early correction (30-50% down)
            if drawdown_from_ath < 0.50:
                return CyclePhase.CORRECTION, confidence
            # Deep correction (50-70% down) - transitioning to accumulation
            elif drawdown_from_ath < 0.70:
                return CyclePhase.CORRECTION, confidence
            # Severe correction (70%+ down) - likely accumulation phase
            else:
                return CyclePhase.ACCUMULATION, confidence
        
        # ACCUMULATION PHASE: Long time in bear market, price near bottom
        if days_since_ath > 365:  # More than 1 year since ATH
            if drawdown_from_ath > 0.70:  # 70%+ down from peak
                return CyclePhase.ACCUMULATION, 0.85
        
        # GROWTH PHASE: Recovery from low, not yet near peak timing
        if days_since_halving < 350:  # Before typical peak window
            if 0.5 < gain_from_cycle_low < 3.0:  # Moderate gains from low
                return CyclePhase.GROWTH, 0.7
        
        # Default to GROWTH with low confidence if unclear
        return CyclePhase.GROWTH, 0.5
    
    def get_position_size_multiplier(self, phase: CyclePhase, confidence: float) -> float:
        """
        Get position size multiplier based on cycle phase
        
        Higher multiplier = more aggressive buying
        Lower multiplier = more conservative
        """
        multipliers = {
            CyclePhase.ACCUMULATION: 3.0,   # 3x - Aggressive accumulation
            CyclePhase.GROWTH: 1.5,         # 1.5x - Moderate buying
            CyclePhase.BUBBLE: 0.2,         # 0.2x - Minimal buying, prepare to sell
            CyclePhase.CORRECTION: 0.5,     # 0.5x - Cautious, wait for bottom
        }
        
        base_multiplier = multipliers[phase]
        
        # Adjust by confidence
        return base_multiplier * confidence
    
    def get_stop_loss_percentage(self, phase: CyclePhase, current_price: float) -> float:
        """
        Get dynamic stop-loss percentage based on cycle phase
        
        Returns:
            Stop loss as decimal (0.10 = 10%)
        """
        gain_from_low = (current_price - self.cycle_low) / self.cycle_low
        
        if phase == CyclePhase.ACCUMULATION:
            # Near bottom - tight stops to protect capital
            return 0.05  # 5%
        
        elif phase == CyclePhase.GROWTH:
            # Early-mid bull - moderate stops
            if gain_from_low < 1.0:  # Less than 2x from low
                return 0.08  # 8%
            else:
                return 0.12  # 12%
        
        elif phase == CyclePhase.BUBBLE:
            # Late bull - wide trailing stops to ride the wave
            if gain_from_low > 5.0:  # More than 6x from low
                return 0.25  # 25% - let winners run
            else:
                return 0.18  # 18%
        
        elif phase == CyclePhase.CORRECTION:
            # Bear market - tight stops or no new positions
            return 0.05  # 5%
        
        return 0.10  # Default 10%
    
    def get_take_profit_strategy(
        self, 
        phase: CyclePhase, 
        current_price: float,
        avg_buy_price: float
    ) -> Dict[str, any]:
        """
        Get take-profit strategy based on cycle phase
        
        Returns:
            {
                'should_take_profit': bool,
                'sell_percentage': float (0.0-1.0),
                'target_prices': [float],
                'reasoning': str
            }
        """
        if avg_buy_price <= 0:
            return {
                'should_take_profit': False,
                'sell_percentage': 0.0,
                'target_prices': [],
                'reasoning': 'No position to sell'
            }
        
        profit_pct = (current_price - avg_buy_price) / avg_buy_price
        
        if phase == CyclePhase.ACCUMULATION:
            # HODL - don't sell near bottom
            return {
                'should_take_profit': False,
                'sell_percentage': 0.0,
                'target_prices': [],
                'reasoning': 'Accumulation phase - HODL all BTC'
            }
        
        elif phase == CyclePhase.GROWTH:
            # Partial profits at milestones
            if profit_pct > 1.0:  # 100%+ profit
                return {
                    'should_take_profit': True,
                    'sell_percentage': 0.15,  # Sell 15%
                    'target_prices': [avg_buy_price * 2.5, avg_buy_price * 4.0],
                    'reasoning': f'Growth phase - take 15% profits at {profit_pct:.1%} gain'
                }
            return {'should_take_profit': False, 'sell_percentage': 0.0, 'target_prices': [], 'reasoning': 'Hold during growth'}
        
        elif phase == CyclePhase.BUBBLE:
            # Aggressive profit-taking
            days_since_halving = self.get_days_since_halving()
            
            # Near historical peak timing (450-550 days post-halving)
            if 450 <= days_since_halving <= 550:
                return {
                    'should_take_profit': True,
                    'sell_percentage': 0.60,  # Sell 60% - MAJOR exit
                    'target_prices': [],
                    'reasoning': f'BUBBLE PEAK RISK - Day {days_since_halving} post-halving (historical peak zone)'
                }
            
            # Early bubble
            elif profit_pct > 2.0:  # 200%+ profit
                return {
                    'should_take_profit': True,
                    'sell_percentage': 0.30,  # Sell 30%
                    'target_prices': [avg_buy_price * 5.0],
                    'reasoning': f'Bubble phase - take 30% profits at {profit_pct:.1%} gain'
                }
            
            return {'should_take_profit': False, 'sell_percentage': 0.0, 'target_prices': [], 'reasoning': 'Early bubble - hold'}
        
        elif phase == CyclePhase.CORRECTION:
            # Preserve capital - sell on any bounce
            if profit_pct > 0.20:  # Any profit >20%
                return {
                    'should_take_profit': True,
                    'sell_percentage': 0.40,  # Sell 40%
                    'target_prices': [],
                    'reasoning': f'Bear market bounce - take profits at {profit_pct:.1%}'
                }
            
            return {'should_take_profit': False, 'sell_percentage': 0.0, 'target_prices': [], 'reasoning': 'Wait for better exit'}
        
        return {'should_take_profit': False, 'sell_percentage': 0.0, 'target_prices': [], 'reasoning': 'Default hold'}
    
    def get_drawdown_buy_signal(self, current_price: float) -> Dict[str, any]:
        """
        Generate buy signals based on drawdown from ATH
        
        Historical pattern: Bear markets drop 70-85% before bottoming
        """
        drawdown = (self.cycle_ath - current_price) / self.cycle_ath
        
        # Estimate potential bottom (75% drawdown as midpoint)
        estimated_bottom = self.cycle_ath * 0.25  # 75% drop
        distance_to_bottom = (current_price - estimated_bottom) / estimated_bottom
        
        signal_strength = "NONE"
        buy_multiplier = 0.0
        reasoning = ""
        
        if drawdown >= 0.75:  # 75%+ down - likely near absolute bottom
            signal_strength = "EXTREME_BUY"
            buy_multiplier = 3.0
            reasoning = f"Historic buying opportunity - {drawdown:.1%} drawdown (historical bottoms: 70-85%)"
        
        elif drawdown >= 0.65:  # 65-75% down - deep value zone
            signal_strength = "STRONG_BUY"
            buy_multiplier = 2.0
            reasoning = f"Strong buy zone - {drawdown:.1%} drawdown approaching historical bottom range"
        
        elif drawdown >= 0.50:  # 50-65% down - mid-correction
            signal_strength = "MODERATE_BUY"
            buy_multiplier = 1.0
            reasoning = f"Moderate correction - {drawdown:.1%} drawdown, could go deeper"
        
        elif drawdown >= 0.30:  # 30-50% down - early correction
            signal_strength = "LIGHT_BUY"
            buy_multiplier = 0.5
            reasoning = f"Early correction - {drawdown:.1%} drawdown, expect more downside"
        
        else:  # <30% down - not enough correction
            signal_strength = "HOLD"
            buy_multiplier = 0.2
            reasoning = f"Insufficient correction - {drawdown:.1%} drawdown (historical bears: 70-85%)"
        
        return {
            'signal': signal_strength,
            'buy_multiplier': buy_multiplier,
            'drawdown': drawdown,
            'estimated_bottom': estimated_bottom,
            'distance_to_bottom_pct': distance_to_bottom,
            'reasoning': reasoning,
            'current_price': current_price,
            'cycle_ath': self.cycle_ath
        }
    
    def check_never_look_back_violation(self, current_price: float) -> bool:
        """
        Check if price is approaching previous cycle lows (rare opportunity)
        
        Returns:
            True if price is near a never-look-back level (extreme buy signal)
        """
        previous_cycle_low = self.HISTORICAL_LOWS[self.current_cycle - 1]['price']
        
        # If within 10% of previous cycle low, this is EXTREME buying opportunity
        if current_price < previous_cycle_low * 1.10:
            logger.warning(
                f"⚠️ EXTREME OPPORTUNITY: Price €{current_price:,} approaching "
                f"previous cycle low €{previous_cycle_low:,}"
            )
            return True
        
        return False
    
    def get_cycle_summary(self, current_price: float) -> Dict:
        """Generate comprehensive cycle analysis summary"""
        phase, confidence = self.estimate_cycle_phase(current_price)
        drawdown_signal = self.get_drawdown_buy_signal(current_price)
        days_since_halving = self.get_days_since_halving()
        days_since_ath = self.get_days_since_ath()
        
        return {
            'cycle_number': self.current_cycle,
            'days_since_halving': days_since_halving,
            'days_since_ath': days_since_ath,
            'current_phase': phase.value,
            'phase_confidence': confidence,
            'current_price': current_price,
            'cycle_ath': self.cycle_ath,
            'cycle_low': self.cycle_low,
            'drawdown_from_ath': drawdown_signal['drawdown'],
            'gain_from_low': (current_price - self.cycle_low) / self.cycle_low,
            'position_multiplier': self.get_position_size_multiplier(phase, confidence),
            'stop_loss_pct': self.get_stop_loss_percentage(phase, current_price),
            'buy_signal': drawdown_signal,
            'near_never_look_back': self.check_never_look_back_violation(current_price)
        }
    
    def print_cycle_analysis(self, current_price: float):
        """Print formatted cycle analysis"""
        summary = self.get_cycle_summary(current_price)
        
        logger.info("=" * 70)
        logger.info("📊 BITCOIN CYCLE ANALYSIS")
        logger.info("=" * 70)
        logger.info(f"Cycle: {summary['cycle_number']} | Phase: {summary['current_phase'].upper()} ({summary['phase_confidence']:.0%} confidence)")
        logger.info(f"Days Since Halving: {summary['days_since_halving']} | Days Since ATH: {summary['days_since_ath']}")
        logger.info(f"")
        logger.info(f"Price Levels:")
        logger.info(f"  Current: €{summary['current_price']:,}")
        logger.info(f"  Cycle ATH: €{summary['cycle_ath']:,} (-{summary['drawdown_from_ath']:.1%})")
        logger.info(f"  Cycle Low: €{summary['cycle_low']:,} (+{summary['gain_from_low']:.1%})")
        logger.info(f"")
        logger.info(f"Trading Adjustments:")
        logger.info(f"  Position Multiplier: {summary['position_multiplier']:.2f}x")
        logger.info(f"  Stop Loss: {summary['stop_loss_pct']:.1%}")
        logger.info(f"")
        logger.info(f"Buy Signal: {summary['buy_signal']['signal']}")
        logger.info(f"  {summary['buy_signal']['reasoning']}")
        logger.info(f"  Estimated Bottom: €{summary['buy_signal']['estimated_bottom']:,.0f}")
        logger.info("=" * 70)