#!/usr/bin/env python3
"""
Bitcoin Trading Bot - Refactored for Simplicity

DESIGN PHILOSOPHY (John Ousterhout):
- Deep modules: CycleAwareTrading hides all complexity
- Information hiding: Main loop doesn't know about cycles
- Pull complexity downward: Trading module handles details
- Simple interface: Just call decide() and execute

MAIN LOOP RESPONSIBILITY:
- Get market data
- Call trading.decide()
- Execute the decision
- Log results

That's it. Everything else is hidden in modules.
"""

import sys
import time
import signal
from datetime import datetime

from config import (
    BITVAVO_API_KEY,
    BITVAVO_API_SECRET,
    CYCLE_CONFIG,
    SLEEP_DURATION,
    validate_config,
    print_config_summary
)

from logger_config import logger
from bitvavo_api import EnhancedBitvavoAPI
from bot_state_manager import BotStateManager
from trade_executor import TradeExecutor
from order_manager import OrderManager
from atomic_trade_manager import AtomicTradeManager
from performance_tracker import PerformanceTracker

# Import our deep module
from cycle_trading_deep_module import CycleAwareTrading, TradingDecision


class TradingBot:
    """
    Main trading bot.
    
    WHY THIS IS SIMPLE:
    - Trading logic is in CycleAwareTrading (deep module)
    - State management is in BotStateManager
    - Order execution is in TradeExecutor
    - Main loop just coordinates
    
    DESIGN RATIONALE:
    - Each module has one job
    - No temporal decomposition (organized by what, not when)
    - No pass-through methods
    - Clear separation of concerns
    """
    
    def __init__(self):
        """Initialize bot components."""
        logger.info("🚀 Initializing Bitcoin Trading Bot...")
        
        validate_config()
        print_config_summary()
        
        # Core components
        self.api = EnhancedBitvavoAPI(BITVAVO_API_KEY, BITVAVO_API_SECRET)
        self.state = BotStateManager()
        self.executor = TradeExecutor(self.api)
        self.orders = OrderManager(self.api, self.executor)
        self.atomic = AtomicTradeManager(self.orders, self.state, self.executor)
        self.performance = PerformanceTracker(
            initial_btc_balance=0.0,
            initial_eur_balance=0.0,
            load_history=True
        )
        
        # Trading strategy (deep module - hides all cycle complexity)
        if CYCLE_CONFIG['ENABLED']:
            self.trading = CycleAwareTrading(
                base_position_pct=CYCLE_CONFIG['BASE_POSITION_PCT'],
                min_eur_per_trade=15.0
            )
            logger.info("✅ Cycle-aware trading enabled")
        else:
            self.trading = None
            logger.info("ℹ️  Cycle-aware trading disabled")
        
        # Bot control
        self.running = True
        self.iteration = 0
        
        # Graceful shutdown
        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)
        
        logger.info("✅ Bot initialization complete!")
    
    def _stop(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"\n🛑 Shutting down (signal {signum})...")
        self.running = False
    
    def run(self):
        """
        Main trading loop.
        
        WHY THIS IS SIMPLE:
        - Get market state (4 values)
        - Call trading.decide()
        - Execute decision
        - Log and sleep
        
        COMPLEXITY IS HIDDEN:
        - Cycle analysis: In CycleAwareTrading
        - Order execution: In TradeExecutor
        - State updates: In BotStateManager
        
        THIS METHOD JUST COORDINATES.
        """
        logger.info("\n" + "🚀" * 35)
        logger.info("STARTING BITCOIN TRADING BOT")
        logger.info("🚀" * 35 + "\n")
        
        try:
            while self.running:
                self.iteration += 1
                self._log_iteration_start()
                
                # Step 1: Get market state
                state = self._get_market_state()
                if not state:
                    logger.error("Failed to get market state, sleeping 60s...")
                    time.sleep(60)
                    continue
                
                # Step 2: Process any filled orders
                self._process_orders()
                
                # Step 3: Update performance tracking
                self._update_performance(state)
                
                # Step 4: Make trading decision (THE KEY METHOD)
                if self.trading:
                    decision = self.trading.decide(
                        price=state['price'],
                        btc_held=state['btc'],
                        eur_available=state['eur'],
                        avg_buy_price=self.state.get_avg_buy_price()
                    )
                else:
                    # Fallback: No trading if cycle-aware disabled
                    decision = TradingDecision(
                        should_buy=False,
                        should_sell=False,
                        btc_amount=0.0,
                        price=state['price'],
                        reasoning="Cycle-aware trading disabled",
                        risk_level="UNKNOWN"
                    )
                
                # Step 5: Execute decision
                self._execute_decision(decision, state)
                
                # Step 6: Log summary
                self._log_iteration_summary(state, decision)
                
                # Step 7: Sleep
                logger.info(f"😴 Sleeping {SLEEP_DURATION // 60} minutes...\n")
                time.sleep(SLEEP_DURATION)
        
        except KeyboardInterrupt:
            logger.info("\n🛑 Stopped by user")
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}", exc_info=True)
        finally:
            self._shutdown()
    
    def _get_market_state(self) -> dict:
        """
        Get current market state.
        
        WHY A DICT:
        - Simple return value
        - No need for a MarketState class (that would be shallow)
        - Easy to extend with new fields
        
        Returns dict with: price, btc, eur
        """
        try:
            price = self.executor.get_current_price()
            btc = self.executor.get_total_btc_balance()
            eur = self.executor.get_available_balance("EUR")
            
            if price is None:
                return None
            
            logger.info(f"💰 BTC: {btc:.8f} | EUR: €{eur:,.2f}")
            logger.info(f"📊 Price: €{price:,.2f}")
            
            return {'price': price, 'btc': btc, 'eur': eur}
        
        except Exception as e:
            logger.error(f"Failed to get market state: {e}")
            return None
    
    def _process_orders(self):
        """Process any filled orders."""
        try:
            results = self.atomic.process_filled_orders()
            if results['processed'] > 0:
                logger.info(f"✅ Processed {results['processed']} filled orders")
            if results['still_pending'] > 0:
                logger.info(f"⏳ {results['still_pending']} orders still pending")
        except Exception as e:
            logger.error(f"Failed to process orders: {e}")
    
    def _update_performance(self, state: dict):
        """Update performance tracking."""
        try:
            self.performance.update_equity(
                btc_balance=state['btc'],
                eur_balance=state['eur'],
                current_btc_price=state['price']
            )
        except Exception as e:
            logger.error(f"Failed to update performance: {e}")
    
    def _execute_decision(self, decision: TradingDecision, state: dict):
        """
        Execute a trading decision.
        
        WHY THIS IS SIMPLE:
        - Decision object has everything we need
        - Just check buy/sell flags and execute
        - No complex branching logic
        
        DESIGN:
        - TradingDecision is the interface between strategy and execution
        - Clean separation: decide() vs execute()
        """
        if decision.should_buy:
            logger.info(f"🟢 BUY: {decision.btc_amount:.8f} BTC @ €{decision.price:,.2f}")
            logger.info(f"   Reason: {decision.reasoning}")
            
            try:
                # Execute buy
                order_id = self.atomic.atomic_buy(
                    btc_amount=decision.btc_amount,
                    price=decision.price
                )
                if order_id:
                    logger.info(f"✅ Buy order placed: {order_id}")
            except Exception as e:
                logger.error(f"Failed to execute buy: {e}")
        
        elif decision.should_sell:
            logger.info(f"🔴 SELL: {decision.btc_amount:.8f} BTC @ €{decision.price:,.2f}")
            logger.info(f"   Reason: {decision.reasoning}")
            
            try:
                # Execute sell
                order_id = self.atomic.atomic_sell(
                    btc_amount=decision.btc_amount,
                    price=decision.price
                )
                if order_id:
                    logger.info(f"✅ Sell order placed: {order_id}")
            except Exception as e:
                logger.error(f"Failed to execute sell: {e}")
        
        else:
            logger.info(f"⏸️  HOLD - {decision.reasoning}")
    
    def _log_iteration_start(self):
        """Log iteration header."""
        logger.info("=" * 70)
        logger.info(f"ITERATION #{self.iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)
    
    def _log_iteration_summary(self, state: dict, decision: TradingDecision):
        """Log iteration summary."""
        logger.info("─" * 70)
        
        # Show position if we have one
        avg_buy = self.state.get_avg_buy_price()
        if avg_buy > 0:
            pnl = (state['price'] - avg_buy) / avg_buy
            logger.info(f"Position: Avg €{avg_buy:,.0f} | P&L: {pnl:+.1%}")
        
        # Show decision
        action = "BUY" if decision.should_buy else "SELL" if decision.should_sell else "HOLD"
        logger.info(f"Action: {action}")
        logger.info(f"Risk: {decision.risk_level}")
        
        if decision.phase:
            logger.info(f"Phase: {decision.phase}")
        
        logger.info("─" * 70)
        
        # Print full stats every 20 iterations
        if self.iteration % 20 == 0:
            self._print_statistics()
    
    def _print_statistics(self):
        """Print comprehensive statistics."""
        logger.info("\n" + "=" * 70)
        logger.info("📈 PERFORMANCE STATISTICS")
        logger.info("=" * 70)
        
        self.state.print_statistics()
        self.performance.print_performance_summary()
        
        logger.info("=" * 70 + "\n")
    
    def _shutdown(self):
        """Graceful shutdown."""
        logger.info("\n🛑 Shutting down...")
        
        try:
            self._print_statistics()
            self.state.save_state()
            self.atomic._save_pending()
            logger.info("✅ Shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        logger.info("👋 Goodbye!\n")


def main():
    """Entry point."""
    try:
        bot = TradingBot()
        bot.run()
    except Exception as e:
        logger.error(f"Failed to start bot: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
