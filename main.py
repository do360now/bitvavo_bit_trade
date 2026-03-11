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
        self.executor.initialize()  # Fetch market specs (tick size, min order)
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

        # Circuit breaker tracking
        self.iteration = 0
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3
        self.last_circuit_breaker_trigger = None
        self.circuit_breaker_wait_count = 0  # Track consecutive CB waits

        # Bot control
        self.running = True

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)

        # Run startup health check
        if not self._health_check():
            logger.error("❌ Startup health check failed - fix issues before running bot")
            logger.error("💡 Check API credentials and network connectivity")
            return

        logger.info("✅ Bot initialization complete!")

    def _health_check(self) -> bool:
        """
        Verify API connectivity and credentials before starting.
        Returns True if health check passes, False otherwise.
        """
        logger.info("🔍 Running startup health check...")

        checks = [
            ("Fetch BTC price", self.executor.get_current_price()),
            ("Fetch EUR balance", self.executor.get_available_balance("EUR")),
            ("Fetch BTC balance", self.executor.get_total_btc_balance()),
        ]

        all_passed = True
        for check_name, result in checks:
            if result is None:
                logger.error(f"❌ Health check failed: {check_name}")
                all_passed = False
            else:
                logger.info(f"✅ {check_name}: OK")

        return all_passed

    def _health_check(self) -> bool:
        """
        Run startup health check to verify API connectivity.
        Returns True if all checks pass, False otherwise.
        """
        logger.info("🔍 Running startup health check...")

        checks = {
            'price': False,
            'balance_btc': False,
            'balance_eur': False,
        }

        # Check 1: Can fetch current price
        try:
            price = self.executor.get_current_price()
            if price and price > 0:
                checks['price'] = True
                logger.info(f"  ✅ Price API: €{price:,.2f}")
            else:
                logger.error(f"  ❌ Price API: Invalid price returned")
        except Exception as e:
            logger.error(f"  ❌ Price API: {e}")

        # Check 2: Can fetch BTC balance
        try:
            btc = self.executor.get_total_btc_balance()
            if btc is not None and btc >= 0:
                checks['balance_btc'] = True
                logger.info(f"  ✅ BTC balance: {btc:.8f} BTC")
            else:
                logger.error(f"  ❌ BTC balance: Invalid balance returned")
        except Exception as e:
            logger.error(f"  ❌ BTC balance: {e}")

        # Check 3: Can fetch EUR balance
        try:
            eur = self.executor.get_available_balance("EUR")
            if eur is not None and eur >= 0:
                checks['balance_eur'] = True
                logger.info(f"  ✅ EUR balance: €{eur:,.2f}")
            else:
                logger.error(f"  ❌ EUR balance: Invalid balance returned")
        except Exception as e:
            logger.error(f"  ❌ EUR balance: {e}")

        # Summary
        passed = sum(checks.values())
        total = len(checks)

        if passed == total:
            logger.info(f"🛡️  Health check passed ({passed}/{total})")
            return True
        else:
            logger.error(f"🛡️  Health check FAILED ({passed}/{total})")
            return False

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
        Returns None if circuit breaker is open or API fails
        """
        try:
            price = self.executor.get_current_price()
            btc = self.executor.get_total_btc_balance()
            eur = self.executor.get_available_balance("EUR")

            # Handle circuit breaker open (returns None)
            if price is None or btc is None or eur is None:
                self.consecutive_failures += 1
                self.last_circuit_breaker_trigger = datetime.now()

                if self.consecutive_failures == 1:
                    logger.warning("⚠️ Circuit breaker triggered - API unavailable")
                logger.warning(f"⚠️ Consecutive failures: {self.consecutive_failures}")

                # Exponential backoff: wait longer each failure
                wait_time = min(60 * self.consecutive_failures, 300)
                logger.warning(f"💤 Waiting {wait_time}s for recovery...")
                time.sleep(wait_time)
                return None

            # Success - reset failure counter
            if self.consecutive_failures > 0:
                logger.info(f"✅ API recovered after {self.consecutive_failures} failures")
                self.consecutive_failures = 0

            logger.info(f"💰 BTC: {btc:.8f} | EUR: €{eur:,.2f}")
            logger.info(f"📊 Price: €{price:,.2f}")

            return {'price': price, 'btc': btc, 'eur': eur}

        except Exception as e:
            self.consecutive_failures += 1
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
    
    TRAILING_STOP_PCT = 0.10  # 10% trailing stop

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
        # Check trailing stop BEFORE strategy decision
        if state['btc'] > 0 and self.state.get_avg_buy_price() > 0:
            trailing_triggered = self._check_trailing_stop(state)
            if trailing_triggered:
                # Trailing stop overrides strategy
                return

        if decision.should_buy:
            logger.info(f"🟢 BUY: {decision.btc_amount:.8f} BTC @ €{decision.price:,.2f}")
            logger.info(f"   Reason: {decision.reasoning}")

            try:
                # Get optimal price from order book for better fill
                order_book = self.executor.get_order_book(depth=5)
                exec_price = self.executor.get_optimal_price(order_book, "buy") if order_book else decision.price

                logger.info(f"   📊 Order book price: €{exec_price:,.2f}")

                # Execute buy
                order_id = self.atomic.execute_buy(
                    volume=decision.btc_amount,
                    price=exec_price
                )
                if order_id:
                    logger.info(f"✅ Buy order placed: {order_id}")
            except Exception as e:
                logger.error(f"Failed to execute buy: {e}")

        elif decision.should_sell:
            logger.info(f"🔴 SELL: {decision.btc_amount:.8f} BTC @ €{decision.price:,.2f}")
            logger.info(f"   Reason: {decision.reasoning}")

            try:
                # Get optimal price from order book for better fill
                order_book = self.executor.get_order_book(depth=5)
                exec_price = self.executor.get_optimal_price(order_book, "sell") if order_book else decision.price

                logger.info(f"   📊 Order book price: €{exec_price:,.2f}")

                # Execute sell
                order_id = self.atomic.atomic_sell(
                    btc_amount=decision.btc_amount,
                    price=exec_price
                )
                if order_id:
                    logger.info(f"✅ Sell order placed: {order_id}")
            except Exception as e:
                logger.error(f"Failed to execute sell: {e}")
        
        else:
            logger.info(f"⏸️  HOLD - {decision.reasoning}")
    
    def _check_trailing_stop(self, state: dict) -> bool:
        """
        Check if trailing stop should trigger.

        TRAILING STOP LOGIC:
        - Track peak price since position opened
        - If price drops X% from peak, sell (stop loss)
        - This locks in profits while allowing upside

        Returns True if trailing stop triggered and sell executed
        """
        TRAILING_STOP_PCT = 0.15  # 15% drop from peak = sell

        current_price = state['price']
        avg_buy_price = self.state.get_avg_buy_price()
        btc_held = state['btc']

        if btc_held <= 0 or avg_buy_price <= 0:
            return False

        # Get peak price (highest price since we bought)
        peak_price = self.state.get_peak_price()

        # Update peak if current price is higher
        if current_price > peak_price:
            self.state.update_peak_price(current_price)
            peak_price = current_price

        # Calculate drop from peak
        if peak_price <= 0:
            return False

        drawdown_from_peak = (peak_price - current_price) / peak_price

        # Trigger trailing stop
        if drawdown_from_peak >= TRAILING_STOP_PCT:
            profit_from_buy = (current_price - avg_buy_price) / avg_buy_price

            logger.warning(f"🛡️  TRAILING STOP TRIGGERED!")
            logger.warning(f"   Peak: €{peak_price:,.2f} | Current: €{current_price:,.2f}")
            logger.warning(f"   Drawdown: {drawdown_from_peak:.1%} | Profit: {profit_from_buy:.1%}")

            try:
                # Execute sell at market (use current price for fast fill)
                order_id = self.atomic.atomic_sell(
                    btc_amount=btc_held,
                    price=current_price
                )
                if order_id:
                    logger.warning(f"✅ Trailing stop SELL executed: {order_id}")
                    return True
            except Exception as e:
                logger.error(f"❌ Trailing stop failed: {e}")

        return False

    def _check_trailing_stop(self, state: dict) -> bool:
        """
        Check if trailing stop should trigger.
        Protects profits by selling when price drops from peak.

        Trailing stop: If price rises X%, then drops Y%, sell.
        This lets profits run while protecting against reversals.
        """
        TRAILING_STOP_PERCENT = 0.10  # 10% trailing stop

        current_price = state['price']
        avg_buy = self.state.get_avg_buy_price()
        peak_price = self.state.get_peak_price()

        # No position or no profit yet
        if avg_buy <= 0 or current_price <= avg_buy:
            # Update peak price
            if current_price > peak_price:
                self.state.state['peak_price'] = current_price
                self.state.save_state()
            return False

        # Update peak if new high
        if current_price > peak_price:
            self.state.state['peak_price'] = current_price
            self.state.save_state()
            logger.debug(f"📈 New peak price: €{peak_price:,.2f}")
            return False

        # Calculate drawdown from peak
        if peak_price > 0:
            drawdown = (peak_price - current_price) / peak_price

            if drawdown >= TRAILING_STOP_PERCENT:
                profit_pct = (current_price - avg_buy) / avg_buy
                logger.warning(f"🛡️  TRAILING STOP: Price dropped {drawdown:.1%} from peak €{peak_price:,.2f}")
                logger.warning(f"   Current: €{current_price:,.2f} | Profit: {profit_pct:.1%}")

                # Execute trailing stop sell
                try:
                    order_id = self.atomic.atomic_sell(
                        btc_amount=state['btc'],
                        price=current_price
                    )
                    if order_id:
                        logger.warning(f"✅ Trailing stop executed: {order_id}")
                        return True
                except Exception as e:
                    logger.error(f"Failed to execute trailing stop: {e}")

        return False

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
