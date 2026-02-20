"""
Cycle-Aware Trading Strategy Integration

Enhances existing bot with Bitcoin cycle intelligence for better entry/exit timing

IMPORTANT: All prices in EUR (designed for BTC-EUR trading on Bitvavo)
Historical data converted from USD sources at ~1.10 USD/EUR exchange rate
"""

from typing import Dict, Optional, Tuple
from logger_config import logger
from bitcoin_cycle_detector import BitcoinCycleDetector, CyclePhase


class CycleAwareStrategy:
    """
    Wraps existing trading logic with cycle-aware enhancements
    
    Integrates with:
    - BotStateManager
    - TradeExecutor  
    - OrderManager
    - AtomicTradeManager
    """
    
    def __init__(
        self,
        cycle_detector: BitcoinCycleDetector,
        state_manager,
        trade_executor,
        atomic_trade_manager,
        base_position_pct: float = 0.10  # Base 10% position size
    ):
        self.cycle_detector = cycle_detector
        self.state_manager = state_manager
        self.trade_executor = trade_executor
        self.atomic_trade_manager = atomic_trade_manager
        self.base_position_pct = base_position_pct
        
        logger.info("🔄 Initialized Cycle-Aware Trading Strategy")
    
    def should_buy(self, current_price: float, available_eur: float) -> Tuple[bool, float, str]:
        """
        Cycle-aware buy decision
        
        Returns:
            (should_buy, position_size_btc, reasoning)
        """
        # Get cycle analysis
        cycle_summary = self.cycle_detector.get_cycle_summary(current_price)
        phase = CyclePhase(cycle_summary['current_phase'])
        drawdown_signal = cycle_summary['buy_signal']
        
        # Base position size
        base_size_eur = available_eur * self.base_position_pct
        
        # Apply cycle multiplier
        position_multiplier = cycle_summary['position_multiplier']
        adjusted_size_eur = base_size_eur * position_multiplier
        
        # Safety caps
        max_position_eur = available_eur * 0.50  # Never use more than 50% at once
        adjusted_size_eur = min(adjusted_size_eur, max_position_eur)
        
        # Calculate BTC amount
        position_size_btc = self.trade_executor.calculate_position_size(
            adjusted_size_eur,
            current_price,
            risk_percent=100.0  # Using full adjusted amount
        )
        
        # Decision logic based on phase
        reasoning = ""
        should_buy = False
        
        if phase == CyclePhase.ACCUMULATION:
            # Aggressive buying near bottom
            if drawdown_signal['signal'] in ['EXTREME_BUY', 'STRONG_BUY']:
                should_buy = True
                reasoning = f"ACCUMULATION PHASE: {drawdown_signal['reasoning']}"
            else:
                should_buy = True  # Still buy, but with normal size
                reasoning = "Accumulation phase - steady buying"
        
        elif phase == CyclePhase.GROWTH:
            # Moderate buying during recovery
            if drawdown_signal['signal'] in ['MODERATE_BUY', 'STRONG_BUY', 'EXTREME_BUY']:
                should_buy = True
                reasoning = f"GROWTH PHASE: {drawdown_signal['reasoning']}"
            else:
                should_buy = False
                reasoning = "Growth phase - waiting for dip"
        
        elif phase == CyclePhase.BUBBLE:
            # Minimal buying, prepare to sell
            if drawdown_signal['drawdown'] > 0.20:  # Only if we get a 20%+ correction
                should_buy = True
                reasoning = "BUBBLE CORRECTION: Buying the dip during euphoria"
            else:
                should_buy = False
                reasoning = "BUBBLE PHASE: Too risky to buy, prepare to take profits"
        
        elif phase == CyclePhase.CORRECTION:
            # Cautious - wait for deep drawdown
            if drawdown_signal['signal'] in ['STRONG_BUY', 'EXTREME_BUY']:
                should_buy = True
                reasoning = f"BEAR MARKET: {drawdown_signal['reasoning']}"
            else:
                should_buy = False
                reasoning = f"CORRECTION: Waiting for deeper drop ({drawdown_signal['drawdown']:.1%} down, target 70%+)"
        
        # Safety check: Never buy if position size too small
        if position_size_btc < 0.0001:
            should_buy = False
            reasoning = "Position size too small"
        
        # Log decision
        if should_buy:
            logger.info(f"✅ BUY SIGNAL: {position_size_btc:.8f} BTC (€{adjusted_size_eur:.2f})")
            logger.info(f"   {reasoning}")
            logger.info(f"   Multiplier: {position_multiplier:.2f}x | Phase: {phase.value}")
        else:
            logger.debug(f"❌ NO BUY: {reasoning}")
        
        return should_buy, position_size_btc, reasoning
    
    def should_sell(
        self, 
        current_price: float, 
        btc_balance: float
    ) -> Tuple[bool, float, str]:
        """
        Cycle-aware sell decision
        
        Returns:
            (should_sell, btc_amount, reasoning)
        """
        # Get current position info
        avg_buy_price = self.state_manager.get_avg_buy_price()
        
        if avg_buy_price <= 0 or btc_balance <= 0:
            return False, 0.0, "No position to sell"
        
        # Get cycle analysis
        cycle_summary = self.cycle_detector.get_cycle_summary(current_price)
        phase = CyclePhase(cycle_summary['current_phase'])
        
        # Get take-profit strategy
        tp_strategy = self.cycle_detector.get_take_profit_strategy(
            phase,
            current_price,
            avg_buy_price
        )
        
        should_sell = tp_strategy['should_take_profit']
        sell_pct = tp_strategy['sell_percentage']
        reasoning = tp_strategy['reasoning']
        
        # Calculate sell amount
        btc_to_sell = btc_balance * sell_pct if should_sell else 0.0
        
       # Apply stop-loss check (but NOT for deep underwater positions in bear market)
        stop_loss_pct = cycle_summary['stop_loss_pct']
        profit_pct = (current_price - avg_buy_price) / avg_buy_price

        # Don't trigger stop loss if:
        # 1. Already deep underwater (> -15% loss)
        # 2. In CORRECTION or ACCUMULATION phase (bear market)
        # Rationale: These are old positions, selling now = panic selling at the bottom
        if profit_pct < -stop_loss_pct:
            if profit_pct < -0.15 and phase in [CyclePhase.CORRECTION, CyclePhase.ACCUMULATION]:
                # Override: Don't sell deep underwater positions in bear market
                logger.warning(f"⚠️ Stop loss ignored for underwater position: {profit_pct:.1%} loss")
                logger.warning(f"   Reason: Deep drawdown in {phase.value} phase - HODL through bear market")
            else:
                # Normal stop loss for smaller losses or other phases
                should_sell = True
                btc_to_sell = btc_balance * 0.90
                reasoning = f"STOP LOSS HIT: {profit_pct:.1%} loss (limit: {-stop_loss_pct:.1%})"
        
        # Log decision
        if should_sell:
            logger.info(f"💰 SELL SIGNAL: {btc_to_sell:.8f} BTC ({sell_pct:.0%} of holdings)")
            logger.info(f"   {reasoning}")
            logger.info(f"   Profit: {profit_pct:.1%} | Phase: {phase.value}")
        else:
            logger.debug(f"💎 HODL: {reasoning}")
        
        return should_sell, btc_to_sell, reasoning
    
    def get_dynamic_stop_loss(self, current_price: float) -> float:
        """
        Get cycle-aware stop loss percentage
        
        Returns:
            Stop loss as decimal (0.10 = 10%)
        """
        cycle_summary = self.cycle_detector.get_cycle_summary(current_price)
        return cycle_summary['stop_loss_pct']
    
    def get_risk_assessment(self, current_price: float) -> Dict:
        """
        Comprehensive risk assessment based on cycle position
        
        Returns:
            {
                'risk_level': 'LOW' | 'MEDIUM' | 'HIGH' | 'EXTREME',
                'should_reduce_exposure': bool,
                'max_position_pct': float,
                'warnings': [str]
            }
        """
        cycle_summary = self.cycle_detector.get_cycle_summary(current_price)
        phase = CyclePhase(cycle_summary['current_phase'])
        days_since_halving = cycle_summary['days_since_halving']
        drawdown = cycle_summary['drawdown_from_ath']
        
        warnings = []
        risk_level = "MEDIUM"
        should_reduce = False
        max_position_pct = 0.15  # Default 15%
        
        # BUBBLE PHASE RISKS
        if phase == CyclePhase.BUBBLE:
            risk_level = "EXTREME"
            should_reduce = True
            max_position_pct = 0.05  # Max 5% new positions
            
            # Check if we're in historical peak danger zone
            if 450 <= days_since_halving <= 550:
                warnings.append(f"⚠️ PEAK DANGER ZONE: Day {days_since_halving} post-halving (historical peaks: 368-549 days)")
                warnings.append("📉 Consider taking 50-75% profits")
            
            gain_from_low = cycle_summary['gain_from_low']
            if gain_from_low > 5.0:
                warnings.append(f"⚠️ Price is {gain_from_low:.1f}x from cycle low - extreme euphoria")
        
        # CORRECTION PHASE RISKS  
        elif phase == CyclePhase.CORRECTION:
            if drawdown < 0.50:
                risk_level = "HIGH"
                should_reduce = False  # Don't sell in early correction
                max_position_pct = 0.08
                warnings.append(f"📊 Early bear market: {drawdown:.1%} down (historical: 70-85%)")
                warnings.append("⏳ Expect deeper correction before accumulation zone")
            elif drawdown < 0.70:
                risk_level = "MEDIUM"
                max_position_pct = 0.12
                warnings.append(f"📊 Mid correction: {drawdown:.1%} down, approaching bottom zone")
            else:
                risk_level = "LOW"
                max_position_pct = 0.20
                warnings.append(f"✅ Deep correction: {drawdown:.1%} down - historic buying zone")
        
        # ACCUMULATION PHASE
        elif phase == CyclePhase.ACCUMULATION:
            risk_level = "LOW"
            max_position_pct = 0.25  # Can take larger positions
            warnings.append("✅ Accumulation phase - excellent risk/reward")
        
        # GROWTH PHASE
        elif phase == CyclePhase.GROWTH:
            risk_level = "MEDIUM"
            max_position_pct = 0.15
            
            # Check if we're approaching bubble zone
            if days_since_halving > 300:
                warnings.append(f"⚠️ Day {days_since_halving} post-halving - approaching typical peak window")
        
        # Check for never-look-back violation (extreme opportunity)
        if cycle_summary['near_never_look_back']:
            risk_level = "LOW"
            max_position_pct = 0.30
            warnings.append("🚨 EXTREME OPPORTUNITY: Price near previous cycle low - historic buying zone")
        
        return {
            'risk_level': risk_level,
            'should_reduce_exposure': should_reduce,
            'max_position_pct': max_position_pct,
            'warnings': warnings,
            'phase': phase.value,
            'drawdown': drawdown,
            'days_since_halving': days_since_halving
        }
    
    def execute_cycle_aware_trade(
        self,
        current_price: float,
        available_eur: float,
        btc_balance: float
    ) -> Dict:
        """
        Main trading logic with cycle awareness
        
        Returns:
            {
                'action': 'BUY' | 'SELL' | 'HOLD',
                'amount': float,
                'price': float,
                'reasoning': str,
                'order_id': str | None
            }
        """
        # Print cycle analysis
        self.cycle_detector.print_cycle_analysis(current_price)
        
        # Get risk assessment
        risk_assessment = self.get_risk_assessment(current_price)
        
        # Print warnings
        if risk_assessment['warnings']:
            logger.warning("⚠️ CYCLE RISK WARNINGS:")
            for warning in risk_assessment['warnings']:
                logger.warning(f"  {warning}")
        
        # Check sell signals first
        should_sell, btc_to_sell, sell_reason = self.should_sell(current_price, btc_balance)
        
        if should_sell and btc_to_sell > 0:
            # Execute sell
            logger.info(f"🔴 Executing SELL: {btc_to_sell:.8f} BTC @ €{current_price:,.2f}")
            
            order_id = self.atomic_trade_manager.execute_sell(
                volume=btc_to_sell,
                price=current_price,
                market="BTC-EUR"
            )
            
            return {
                'action': 'SELL',
                'amount': btc_to_sell,
                'price': current_price,
                'reasoning': sell_reason,
                'order_id': order_id
            }
        
        # Check buy signals
        should_buy, btc_to_buy, buy_reason = self.should_buy(current_price, available_eur)
        
        if should_buy and btc_to_buy > 0:
            # Double-check risk limits
            if risk_assessment['should_reduce_exposure']:
                logger.warning(f"⚠️ Risk assessment suggests reducing exposure - skipping buy")
                return {
                    'action': 'HOLD',
                    'amount': 0,
                    'price': current_price,
                    'reasoning': 'Risk management override',
                    'order_id': None
                }
            
            # Execute buy
            logger.info(f"🟢 Executing BUY: {btc_to_buy:.8f} BTC @ €{current_price:,.2f}")
            
            order_id = self.atomic_trade_manager.execute_buy(
                volume=btc_to_buy,
                price=current_price,
                market="BTC-EUR"
            )
            
            return {
                'action': 'BUY',
                'amount': btc_to_buy,
                'price': current_price,
                'reasoning': buy_reason,
                'order_id': order_id
            }
        
        # No action
        return {
            'action': 'HOLD',
            'amount': 0,
            'price': current_price,
            'reasoning': 'No cycle-based signals',
            'order_id': None
        }
    
    def get_cycle_statistics(self) -> Dict:
        """Get current cycle statistics for monitoring"""
        current_price = self.trade_executor.get_current_price()
        
        if not current_price:
            return {}
        
        cycle_summary = self.cycle_detector.get_cycle_summary(current_price)
        risk_assessment = self.get_risk_assessment(current_price)
        
        return {
            'cycle_phase': cycle_summary['current_phase'],
            'phase_confidence': cycle_summary['phase_confidence'],
            'days_since_halving': cycle_summary['days_since_halving'],
            'drawdown_from_ath': cycle_summary['drawdown_from_ath'],
            'position_multiplier': cycle_summary['position_multiplier'],
            'stop_loss_pct': cycle_summary['stop_loss_pct'],
            'risk_level': risk_assessment['risk_level'],
            'buy_signal': cycle_summary['buy_signal']['signal'],
            'estimated_bottom': cycle_summary['buy_signal']['estimated_bottom']
        }


# Example integration with main.py

def integrate_with_main_loop(
    bitvavo_api,
    state_manager,
    trade_executor,
    atomic_trade_manager,
    order_manager
):
    """
    Example of how to integrate cycle-aware strategy with existing main.py
    
    Replace your main trading loop with this enhanced version
    """
    
    # Initialize cycle detector
    cycle_detector = BitcoinCycleDetector(current_cycle=4)
    
    # Initialize cycle-aware strategy
    strategy = CycleAwareStrategy(
        cycle_detector=cycle_detector,
        state_manager=state_manager,
        trade_executor=trade_executor,
        atomic_trade_manager=atomic_trade_manager,
        base_position_pct=0.10  # 10% base position
    )
    
    logger.info("🚀 Starting cycle-aware trading bot")
    
    while True:
        try:
            # Get current market state
            current_price = trade_executor.get_current_price()
            available_eur = trade_executor.get_available_balance("EUR")
            btc_balance = trade_executor.get_total_btc_balance()
            
            if not current_price:
                logger.error("Failed to get current price, skipping iteration")
                continue
            
            # Process any filled orders first
            atomic_trade_manager.process_filled_orders()
            
            # Execute cycle-aware trading logic
            trade_result = strategy.execute_cycle_aware_trade(
                current_price=current_price,
                available_eur=available_eur,
                btc_balance=btc_balance
            )
            
            # Log result
            logger.info(f"📊 Trade result: {trade_result['action']} - {trade_result['reasoning']}")
            
            # Get cycle statistics for monitoring
            cycle_stats = strategy.get_cycle_statistics()
            logger.info(f"📈 Cycle Stats: Phase={cycle_stats.get('cycle_phase')}, "
                       f"Risk={cycle_stats.get('risk_level')}, "
                       f"Signal={cycle_stats.get('buy_signal')}")
            
            # Sleep between iterations (e.g., 15 minutes)
            import time
            time.sleep(900)  # 15 minutes
            
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            import time
            time.sleep(60)  # Wait 1 minute on error
