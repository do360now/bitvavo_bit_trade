#!/usr/bin/env python3
"""
Bitcoin Trading Bot - Macro-Aware Profit Engine

WHAT CHANGED (v2):
- Replaced accumulation-only CycleAwareTrading with active ProfitEngine
- Fetches OHLCV data for technical signal generation
- Fetches macro data (correlations, sentiment) when available
- Fixed: duplicate _health_check and _check_trailing_stop methods
- Fixed: atomic_sell → execute_sell (method didn't exist)
- Added adaptive trailing stop that works with the new signals
- Profit engine generates both buy AND sell signals

DESIGN (Ousterhout):
- Deep modules: ProfitEngine hides all signal complexity
- Main loop just coordinates: get data → decide → execute → log
- No temporal decomposition
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
    NEWS_CONFIG,
    CORRELATION_CONFIG,
    validate_config,
    print_config_summary,
)

from logger_config import logger
from bitvavo_api import EnhancedBitvavoAPI
from bot_state_manager import BotStateManager
from trade_executor import TradeExecutor
from order_manager import OrderManager
from atomic_trade_manager import AtomicTradeManager
from performance_tracker import PerformanceTracker

# Import engines (accumulation mode for long-term BTC conviction)
from accumulation_engine import AccumulationEngine
from profit_engine import ProfitEngine, TradingDecision


class TradingBot:
    """
    Main trading bot — coordinates data, strategy, and execution.

    v2 CHANGES:
    - Uses ProfitEngine instead of CycleAwareTrading
    - Passes OHLCV data to decide() for technical signals
    - Passes macro data to decide() for regime detection
    - Single trailing stop implementation (no duplicates)
    """

    def __init__(self):
        """Initialize bot components."""
        logger.info("🚀 Initializing Bitcoin Trading Bot v2 (Accumulation Engine)...")

        validate_config()
        print_config_summary()

        # Core components
        self.api = EnhancedBitvavoAPI(BITVAVO_API_KEY, BITVAVO_API_SECRET)
        self.state = BotStateManager()
        self.executor = TradeExecutor(self.api)
        self.executor.initialize()
        self.orders = OrderManager(self.api, self.executor)
        self.atomic = AtomicTradeManager(self.orders, self.state, self.executor)
        self.performance = PerformanceTracker(
            initial_btc_balance=0.0,
            initial_eur_balance=0.0,
            load_history=True,
        )

        # Trading strategy — AccumulationEngine: signal-timed BTC accumulation
        if CYCLE_CONFIG['ENABLED']:
            self.trading = AccumulationEngine(
                base_position_pct=CYCLE_CONFIG['BASE_POSITION_PCT'],
                min_eur_per_trade=15.0,
                total_capital=0.0,  # Will be set from first balance fetch
            )
            logger.info("✅ Accumulation Engine enabled (signal-timed BTC buying)")
        else:
            self.trading = None
            logger.info("ℹ️  Trading engine disabled")

        # Circuit breaker tracking
        self.iteration = 0
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3
        self.last_circuit_breaker_trigger = None

        # Bot control
        self.running = True

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)

        # Run startup health check
        if not self._health_check():
            logger.error("❌ Startup health check failed")
            logger.error("💡 Check API credentials and network connectivity")
            return

        logger.info("✅ Bot initialization complete!")

    def _health_check(self) -> bool:
        """
        Verify API connectivity and credentials before starting.
        Returns True if all checks pass.
        """
        logger.info("🔍 Running startup health check...")

        checks = {
            'price': False,
            'balance_btc': False,
            'balance_eur': False,
        }

        try:
            price = self.executor.get_current_price()
            if price and price > 0:
                checks['price'] = True
                logger.info(f"  ✅ Price API: €{price:,.2f}")
            else:
                logger.error("  ❌ Price API: Invalid price returned")
        except Exception as e:
            logger.error(f"  ❌ Price API: {e}")

        try:
            btc = self.executor.get_total_btc_balance()
            if btc is not None and btc >= 0:
                checks['balance_btc'] = True
                logger.info(f"  ✅ BTC balance: {btc:.8f} BTC")
            else:
                logger.error("  ❌ BTC balance: Invalid balance returned")
        except Exception as e:
            logger.error(f"  ❌ BTC balance: {e}")

        try:
            eur = self.executor.get_available_balance("EUR")
            if eur is not None and eur >= 0:
                checks['balance_eur'] = True
                logger.info(f"  ✅ EUR balance: €{eur:,.2f}")
            else:
                logger.error("  ❌ EUR balance: Invalid balance returned")
        except Exception as e:
            logger.error(f"  ❌ EUR balance: {e}")

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

        v2: Now fetches OHLCV and macro data for the profit engine.
        """
        logger.info("\n" + "🚀" * 35)
        logger.info("STARTING BITCOIN TRADING BOT v2 (ACCUMULATION ENGINE)")
        logger.info("🚀" * 35 + "\n")

        try:
            while self.running:
                self.iteration += 1
                self._log_iteration_start()

                # Step 1: Get market state (price, balances)
                state = self._get_market_state()
                if not state:
                    logger.error("Failed to get market state, sleeping 60s...")
                    time.sleep(60)
                    continue

                # Step 2: Fetch OHLCV for technical signals
                ohlcv = self._fetch_ohlcv()
                prices_1h = [c[4] for c in ohlcv] if ohlcv else None  # Close prices
                volumes_1h = [c[5] for c in ohlcv] if ohlcv else None  # Volumes

                # Step 3: Fetch macro data (best-effort, non-blocking)
                macro_data = self._fetch_macro_data()

                # Step 4: Process any filled orders
                self._process_orders()

                # Step 5: Update performance tracking
                self._update_performance(state)

                # Step 6: Make trading decision (THE KEY METHOD)
                if self.trading:
                    decision = self.trading.decide(
                        price=state['price'],
                        btc_held=state['btc'],
                        eur_available=state['eur'],
                        avg_buy_price=self.state.get_avg_buy_price(),
                        prices_1h=prices_1h,
                        volumes_1h=volumes_1h,
                        macro_data=macro_data,
                    )
                else:
                    decision = TradingDecision(
                        should_buy=False,
                        should_sell=False,
                        btc_amount=0.0,
                        price=state['price'],
                        reasoning="Trading engine disabled",
                        risk_level="UNKNOWN",
                    )

                # Step 7: Execute decision
                self._execute_decision(decision, state)

                # Step 8: Log summary
                self._log_iteration_summary(state, decision)

                # Step 9: Sleep
                logger.info(f"😴 Sleeping {SLEEP_DURATION // 60} minutes...\n")
                time.sleep(SLEEP_DURATION)

        except KeyboardInterrupt:
            logger.info("\n🛑 Stopped by user")
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}", exc_info=True)
        finally:
            self._shutdown()

    # ========================================================================
    # DATA FETCHING
    # ========================================================================

    def _get_market_state(self) -> dict:
        """Get current market state (price, BTC balance, EUR balance)."""
        try:
            price = self.executor.get_current_price()
            btc = self.executor.get_total_btc_balance()
            eur = self.executor.get_available_balance("EUR")

            if price is None or btc is None or eur is None:
                self.consecutive_failures += 1
                self.last_circuit_breaker_trigger = datetime.now()

                if self.consecutive_failures == 1:
                    logger.warning("⚠️ Circuit breaker triggered — API unavailable")
                logger.warning(f"⚠️ Consecutive failures: {self.consecutive_failures}")

                wait_time = min(60 * self.consecutive_failures, 300)
                logger.warning(f"💤 Waiting {wait_time}s for recovery...")
                time.sleep(wait_time)
                return None

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

    def _fetch_ohlcv(self) -> list:
        """
        Fetch hourly OHLCV candles for technical analysis.
        Returns list of [timestamp, open, high, low, close, volume].
        """
        try:
            candles = self.executor.get_ohlcv(interval=3600)  # 1h candles
            if candles and len(candles) > 0:
                logger.debug(f"📊 Fetched {len(candles)} hourly candles")
                return candles
            return []
        except Exception as e:
            logger.warning(f"Failed to fetch OHLCV: {e}")
            return []

    def _fetch_macro_data(self) -> dict:
        """
        Fetch macro data for regime detection.
        Best-effort: returns empty dict if anything fails.
        """
        macro = {}

        # News sentiment (if enabled)
        if NEWS_CONFIG.get('ENHANCED_NEWS_ENABLED'):
            try:
                from indicators import fetch_enhanced_news, calculate_enhanced_sentiment
                articles = fetch_enhanced_news(top_n=15)
                if articles:
                    sentiment = calculate_enhanced_sentiment(articles)
                    macro['sentiment'] = sentiment.get('sentiment', 0)
                    macro['risk_off_probability'] = sentiment.get('risk_off_probability', 0)
            except Exception as e:
                logger.debug(f"News sentiment unavailable: {e}")

        # Market correlations (if enabled)
        if CORRELATION_CONFIG.get('ENABLE_CORRELATION_MONITORING'):
            try:
                from indicators import get_market_correlations
                correlations = get_market_correlations()
                if correlations:
                    macro['correlations'] = correlations
            except Exception as e:
                logger.debug(f"Market correlations unavailable: {e}")

        return macro if macro else None

    # ========================================================================
    # ORDER PROCESSING
    # ========================================================================

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
                current_btc_price=state['price'],
            )
        except Exception as e:
            logger.error(f"Failed to update performance: {e}")

    # ========================================================================
    # EXECUTION
    # ========================================================================

    def _execute_decision(self, decision: TradingDecision, state: dict):
        """
        Execute a trading decision.

        v2 FIX: Uses execute_sell (not nonexistent atomic_sell)
        """
        # Check trailing stop BEFORE strategy decision
        if state['btc'] > 0 and self.state.get_avg_buy_price() > 0:
            trailing_triggered = self._check_trailing_stop(state)
            if trailing_triggered:
                return

        if decision.should_buy:
            logger.info(f"🟢 BUY: {decision.btc_amount:.8f} BTC @ €{decision.price:,.2f}")
            logger.info(f"   Reason: {decision.reasoning}")

            try:
                order_book = self.executor.get_order_book(depth=5)
                exec_price = self.executor.get_optimal_price(order_book, "buy") if order_book else decision.price
                logger.info(f"   📊 Order book price: €{exec_price:,.2f}")

                order_id = self.atomic.execute_buy(
                    volume=decision.btc_amount,
                    price=exec_price,
                )
                if order_id:
                    logger.info(f"✅ Buy order placed: {order_id}")
            except Exception as e:
                logger.error(f"Failed to execute buy: {e}")

        elif decision.should_sell:
            logger.info(f"🔴 SELL: {decision.btc_amount:.8f} BTC @ €{decision.price:,.2f}")
            logger.info(f"   Reason: {decision.reasoning}")

            try:
                order_book = self.executor.get_order_book(depth=5)
                exec_price = self.executor.get_optimal_price(order_book, "sell") if order_book else decision.price
                logger.info(f"   📊 Order book price: €{exec_price:,.2f}")

                # FIX: Use execute_sell (atomic_sell doesn't exist)
                order_id = self.atomic.execute_sell(
                    volume=decision.btc_amount,
                    price=exec_price,
                )
                if order_id:
                    logger.info(f"✅ Sell order placed: {order_id}")
            except Exception as e:
                logger.error(f"Failed to execute sell: {e}")

        else:
            logger.info(f"⏸️  HOLD — {decision.reasoning}")

    def _check_trailing_stop(self, state: dict) -> bool:
        """
        Check if trailing stop should trigger.

        v2: Single implementation (no duplicates), uses adaptive stop %.
        Only triggers when we're in profit.
        """
        TRAILING_STOP_PERCENT = 0.10  # 10% trailing stop from peak

        current_price = state['price']
        avg_buy = self.state.get_avg_buy_price()
        peak_price = self.state.get_peak_price()

        if avg_buy <= 0 or current_price <= avg_buy:
            # Not in profit — update peak but don't trigger
            if current_price > peak_price:
                self.state.state['peak_price'] = current_price
                self.state.save_state()
            return False

        # Update peak if new high
        if current_price > peak_price:
            self.state.state['peak_price'] = current_price
            self.state.save_state()
            logger.debug(f"📈 New peak price: €{current_price:,.2f}")
            return False

        # Calculate drawdown from peak
        if peak_price <= 0:
            return False

        drawdown = (peak_price - current_price) / peak_price

        if drawdown >= TRAILING_STOP_PERCENT:
            profit_pct = (current_price - avg_buy) / avg_buy
            logger.warning(f"🛡️  TRAILING STOP: Price dropped {drawdown:.1%} from peak €{peak_price:,.2f}")
            logger.warning(f"   Current: €{current_price:,.2f} | Profit: {profit_pct:.1%}")

            try:
                # FIX: Use execute_sell
                order_id = self.atomic.execute_sell(
                    volume=state['btc'],
                    price=current_price,
                )
                if order_id:
                    logger.warning(f"✅ Trailing stop executed: {order_id}")
                    return True
            except Exception as e:
                logger.error(f"Failed to execute trailing stop: {e}")

        return False

    # ========================================================================
    # LOGGING
    # ========================================================================

    def _log_iteration_start(self):
        """Log iteration header."""
        logger.info("=" * 70)
        logger.info(f"ITERATION #{self.iteration} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)

    def _log_iteration_summary(self, state: dict, decision: TradingDecision):
        """Log iteration summary."""
        logger.info("─" * 70)

        avg_buy = self.state.get_avg_buy_price()
        if avg_buy > 0:
            pnl = (state['price'] - avg_buy) / avg_buy
            logger.info(f"Position: Avg €{avg_buy:,.0f} | P&L: {pnl:+.1%}")

        action = "BUY" if decision.should_buy else "SELL" if decision.should_sell else "HOLD"
        logger.info(f"Action: {action}")
        logger.info(f"Risk: {decision.risk_level}")

        if decision.phase:
            logger.info(f"Phase: {decision.phase}")

        if decision.signals:
            composite = decision.signals.get('composite', None)
            if composite is not None:
                logger.info(f"Composite Signal: {composite:+.3f}")

        # Show accumulation tier info
        if hasattr(self, 'trading') and self.trading and hasattr(self.trading, 'get_accumulation_status'):
            acc = self.trading.get_accumulation_status(state['price'], state['eur'])
            logger.info(f"Tier: {acc['tier']} | Bullets: {acc['bullets_remaining']} | Reserve: {'✅' if acc['reserve_intact'] else '⚠️'}")

        logger.info("─" * 70)

        if self.iteration % 20 == 0:
            self._print_statistics()

    def _print_statistics(self):
        """Print comprehensive statistics."""
        logger.info("\n" + "=" * 70)
        logger.info("📈 PERFORMANCE STATISTICS")
        logger.info("=" * 70)

        self.state.print_statistics()
        self.performance.print_performance_summary()

        # Accumulation engine context
        if self.trading:
            price = self.executor.get_current_price()
            if price:
                context = self.trading.get_market_context(price)
                logger.info(f"Regime: {context['regime']}")
                logger.info(f"Cycle Phase: {context['cycle_phase']}")
                logger.info(f"Drawdown from ATH: {context['drawdown_from_ath']:.1%}")

                # Accumulation-specific status
                if hasattr(self.trading, 'get_accumulation_status'):
                    eur = self.executor.get_available_balance("EUR") or 0
                    acc_status = self.trading.get_accumulation_status(price, eur)
                    logger.info(f"Accumulation Tier: {acc_status['tier']} (weight: {acc_status['tier_weight']:.0%})")
                    logger.info(f"Dry Powder: €{acc_status['eur_after_reserve']:.0f} (reserve: €{acc_status['reserve']:.0f})")
                    logger.info(f"Bullets Remaining: {acc_status['bullets_remaining']}")

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
