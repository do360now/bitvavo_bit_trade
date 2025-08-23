# main.py
"""
Updated main entry point for the unified Bitcoin trading bot
"""

import time
import signal
import sys
import json
import os
from datetime import datetime


# Import the unified bot and supporting modules
from unified_bot import UnifiedTradingBot
from bitvavo_api import authenticate_exchange, test_connection

from tests.run_diagnostics import  test_performance_tracker
   

# Try to import from your actual file structure
try:
    from core.bot import BotConfiguration
except ImportError:
    from core.bot import BotConfiguration

try:
    from utils.config import PRICE_HISTORY_FILE, BOT_LOGS_FILE
except ImportError:
    # Fallback values if config import fails
    PRICE_HISTORY_FILE = "./price_history.json"
    BOT_LOGS_FILE = "./bot_logs.csv"

try:
    from utils.logger import logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class TradingBotManager:
    """Enhanced bot manager for the unified system"""

    def __init__(self):
        self.bot = None
        self.running = False
        self.iteration_count = 0
        self.start_time = time.time()
        self.bitvavo_api = None

    def initialize_bot(self):
        """Initialize the unified trading bot"""
        try:
            logger.info("🚀 Initializing Unified Bitcoin Trading Bot...")

            # Authenticate with Bitvavo
            logger.info("🔐 Authenticating with Bitvavo...")
            self.bitvavo_api = authenticate_exchange()

            # Test connection
            if not test_connection(self.bitvavo_api):
                logger.error("❌ Exchange connection test failed")
                return False

            # Create bot configuration
            config = BotConfiguration(
                enable_ml=True,
                enable_peak_detection=True,
                enable_onchain_analysis=True,
                enable_news_sentiment=True,
                max_daily_trades=8,
                base_position_size_pct=0.10,
                stop_loss_pct=0.03,
                take_profit_pct=0.10,
                min_confidence_threshold=0.30,  # CRITICAL: Was 0.6, now 0.35 to allow trades!
            )

            # Test performance tracker before starting
            logger.info("🧪 Testing performance tracker...")
            if test_performance_tracker():
                logger.info("✅ Performance tracker test passed")
            else:
                logger.warning("⚠️ Performance tracker test failed - will run without it")
            
            # Check if Bitcoin node is available
            onchain_enabled = self._check_bitcoin_node_available()

            # Initialize the unified bot
            logger.info("🤖 Creating unified trading bot...")
            self.bot = UnifiedTradingBot(
                bitvavo_api=self.bitvavo_api,
                config=config,
                enable_ml=True,
                enable_peak_detection=True,
                enable_advanced_strategies=True,
                enable_onchain_analysis=onchain_enabled
            )
            logger.info(f"Unified bot initialized (OnChain: {'✅' if onchain_enabled else '❌'})")

            # Load any existing state
            self.bot.load_state()

            # Test basic functionality
            logger.info("🧪 Testing bot functionality...")
            current_price, _ = self.bot.trade_executor.fetch_current_price()
            if not current_price:
                raise Exception("Failed to fetch current price")

            btc_balance = self.bot.trade_executor.get_total_btc_balance()
            eur_balance = self.bot.trade_executor.get_available_balance("EUR")

            logger.info("💰 Current Status:")
            logger.info(f"   BTC Price: €{current_price:.2f}")
            logger.info(f"₿🪙  BTC Balance: {btc_balance:.8f}")
            logger.info(f"💶  EUR Balance: €{eur_balance:.2f}")
            logger.info(
                f"   Total Value: €{eur_balance + (btc_balance * current_price):.2f}"
            )

            logger.info("✅ Unified trading bot initialized successfully")
        

            logger.info("🔍 Checking price data freshness...")
            if hasattr(self.bot, 'diagnose_data_freshness'):
                is_fresh = self.bot.diagnose_data_freshness()
                if not is_fresh:
                    logger.warning("⚠️  Price data appears stale - forcing update...")
                    self.bot._force_price_history_update()
            
            return True

        except Exception as e:
            logger.error(f"❌ Bot initialization failed: {e}", exc_info=True)
            return False
        
    def _check_bitcoin_node_available(self) -> bool:
        """Check if Bitcoin Core node is available"""
        try:
            from utils.config import RPC_HOST, RPC_PORT, RPC_USER, RPC_PASSWORD
            
            # Basic check - see if RPC credentials are configured
            if not all([RPC_HOST, RPC_PORT, RPC_USER, RPC_PASSWORD]):
                logger.info("Bitcoin node credentials not configured - OnChain disabled")
                return False
            
            # Try to create a quick connection test
            from bitcoinrpc.authproxy import AuthServiceProxy
            rpc_url = f"http://{RPC_USER}:{RPC_PASSWORD}@{RPC_HOST}:{RPC_PORT}"
            rpc = AuthServiceProxy(rpc_url, timeout=5)
            
            # Quick health check
            block_count = rpc.getblockcount()
            logger.info(f"Bitcoin node available - Block height: {block_count}")
            return True
            
        except ImportError:
            logger.info("Bitcoin RPC library not available - OnChain disabled")
            return False
        except Exception as e:
            logger.info(f"Bitcoin node not available: {e} - OnChain disabled")
        return False

    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""

        def signal_handler(sig, frame):
            logger.info(f"📡 Received signal {sig}, initiating graceful shutdown...")
            self.shutdown()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        logger.info("📡 Signal handlers configured")

    def run_main_loop(self):
        """Main trading loop"""
        self.running = True
        consecutive_errors = 0
        max_consecutive_errors = 5

        logger.info("🔄 Starting main trading loop...")

        while self.running:
            try:
                self.iteration_count += 1
                iteration_start = time.time()

                logger.info(f"\n{'='*60}")
                logger.info(
                    f"🔄 ITERATION #{self.iteration_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                logger.info(f"{'='*60}")

                # Execute the unified strategy
                self.bot.execute_unified_strategy()

                # Reset consecutive error counter on successful iteration
                consecutive_errors = 0

                # Log iteration performance
                iteration_time = time.time() - iteration_start
                uptime = time.time() - self.start_time
                logger.info(
                    f"⏱️ Iteration completed in {iteration_time:.2f}s (uptime: {uptime/3600:.1f}h)"
                )

                # Print comprehensive status every 4 iterations (1 hour with 15min intervals)
                if self.iteration_count % 4 == 0:
                    try:
                        self.bot.print_comprehensive_status()

                        # Cleanup and maintenance every 24 iterations (6 hours)
                        if self.iteration_count % 24 == 0:
                            self._periodic_maintenance()

                    except Exception as status_error:
                        logger.error(f"❌ Status reporting failed: {status_error}")

                # Calculate next run time (aligned to 15-minute intervals)
                self._sleep_until_next_interval()

            except KeyboardInterrupt:
                logger.info("⌨️ Keyboard interrupt received")
                break

            except Exception as e:
                consecutive_errors += 1
                logger.error(
                    f"❌ Main loop error #{consecutive_errors}: {e}", exc_info=True
                )

                # Save state before potential shutdown
                self._emergency_save_state()

                if consecutive_errors >= max_consecutive_errors:
                    logger.error(
                        f"💥 Too many consecutive errors ({consecutive_errors}), shutting down"
                    )
                    break

                # Progressive backoff on errors
                error_sleep = min(300, 30 * consecutive_errors)  # 30s to 5min max
                logger.info(f"⏳ Waiting {error_sleep}s before retry...")
                time.sleep(error_sleep)

        logger.info("🔄 Main trading loop ended")

    def _sleep_until_next_interval(self):
        """Sleep until the next 15-minute interval"""
        try:
            current_time = time.time()
            # Calculate next 15-minute boundary
            next_run = ((current_time // 900) + 1) * 900
            sleep_time = next_run - current_time

            if sleep_time > 0:
                next_run_str = datetime.fromtimestamp(next_run).strftime("%H:%M:%S")
                logger.info(f"😴 Sleeping for {sleep_time:.0f}s until {next_run_str}")
                time.sleep(sleep_time)

        except Exception as e:
            logger.error(f"Sleep calculation error: {e}")
            time.sleep(900)  # Default 15 minutes

    def _emergency_save_state(self):
        """Save critical state in case of emergency shutdown"""
        try:
            if self.bot:
                self.bot.save_state()
                logger.info("💾 Emergency state save completed")
        except Exception as e:
            logger.error(f"Emergency save failed: {e}")

    def _periodic_maintenance(self):
        """Perform periodic maintenance tasks"""
        try:
            logger.info("🧹 Performing periodic maintenance...")

            # Cleanup old orders
            if self.bot and self.bot.order_manager:
                self.bot.order_manager.cleanup_old_orders(days=30)

            # Force sync order state
            if self.bot and self.bot.order_manager:
                self.bot.order_manager.force_refresh_all_orders()

            # Save current state
            if self.bot:
                self.bot.save_state()

            logger.info("✅ Periodic maintenance completed")

        except Exception as e:
            logger.error(f"Periodic maintenance failed: {e}")

    def shutdown(self):
        """Graceful shutdown procedure"""
        logger.info("🛑 Initiating graceful shutdown...")

        self.running = False

        try:
            if self.bot:
                self.bot.shutdown()

            # Calculate total runtime
            total_runtime = time.time() - self.start_time
            logger.info(
                f"⏱️ Total runtime: {total_runtime/3600:.2f} hours ({self.iteration_count} iterations)"
            )

            logger.info("✅ Graceful shutdown completed")
            sys.exit(0)

        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)

    def run_diagnostic_check(self):
        """Run comprehensive diagnostic check"""
        logger.info("🔍 Running diagnostic check...")

        try:
            if not self.bot:
                logger.error("❌ Bot not initialized")
                return False

            # Test API connectivity
            current_price, volume = self.bot.trade_executor.fetch_current_price()
            if current_price:
                logger.info(f"✅ API connectivity: BTC price €{current_price:.2f}")
            else:
                logger.error("❌ API connectivity failed")
                return False

            # Test balances
            btc_balance = self.bot.trade_executor.get_total_btc_balance()
            eur_balance = self.bot.trade_executor.get_available_balance("EUR")
            logger.info(f"✅ Balances: {btc_balance:.8f} BTC, €{eur_balance:.2f} EUR")

            # Test order manager
            if self.bot.order_manager:
                pending = self.bot.order_manager.get_pending_orders()
                stats = self.bot.order_manager.get_order_statistics()
                logger.info(
                    f"✅ Order manager: {len(pending)} pending, {stats['fill_rate']:.1%} fill rate"
                )

            # Test data manager
            prices, volumes = self.bot.data_manager.load_price_history()
            logger.info(f"✅ Data manager: {len(prices)} price points loaded")

            # Test ML engine
            if self.bot.ml_engine:
                ml_status = (
                    "trained" if self.bot.ml_engine.is_trained else "not trained"
                )
                logger.info(f"✅ ML engine: {ml_status}")

            # Test strategies
            logger.info(f"✅ Strategies: {len(self.bot.strategies)} active")

            logger.info("✅ All diagnostic checks passed")
            return True

        except Exception as e:
            logger.error(f"❌ Diagnostic check failed: {e}", exc_info=True)
            return False


def run_status_check():
    """Standalone function to check bot status without running full bot"""
    try:
        logger.info("🔍 Running quick status check...")

        # Quick connectivity test
        bitvavo = authenticate_exchange()

        from trading.executor import TradeExecutor

        executor = TradeExecutor(bitvavo)

        current_price, _ = executor.fetch_current_price()
        btc_balance = executor.get_total_btc_balance()
        eur_balance = executor.get_available_balance("EUR")

        print(f"\n{'='*50}")
        print("📊 QUICK STATUS CHECK")
        print(f"{'='*50}")
        print(f"💰 BTC Price: €{current_price:.2f}")
        print(f"₿🪙 BTC Balance: {btc_balance:.8f}")
        print(f"💶 EUR Balance: €{eur_balance:.2f}")
        print(f"💰 Total Value: €{eur_balance + (btc_balance * current_price):.2f}")

        # Check for saved state
        if os.path.exists("./unified_bot_state.json"):
            with open("./unified_bot_state.json", "r") as f:
                state = json.load(f)
                print(f"📈 Session Trades: {state.get('total_trades', 0)}")
                print(f"💹 Session P&L: €{state.get('total_pnl', 0.0):.2f}")

        # Check for order history
        if os.path.exists("./order_history.json"):
            with open("./order_history.json", "r") as f:
                order_data = json.load(f)
                filled_count = len(order_data.get("filled_orders", {}))
                print(f"📋 Historical Orders: {filled_count} filled")

        print(f"{'='*50}\n")

    except Exception as e:
        logger.error(f"Status check failed: {e}")


def main():
    """Main entry point"""
    bot_manager = None

    try:
        # Create bot manager
        bot_manager = TradingBotManager()

        # Setup signal handlers for graceful shutdown
        bot_manager.setup_signal_handlers()

        # Initialize the unified bot
        if not bot_manager.initialize_bot():
            logger.error("💥 Bot initialization failed, exiting")
            sys.exit(1)

        # Run diagnostic check
        if not bot_manager.run_diagnostic_check():
            logger.error("💥 Diagnostic check failed, exiting")
            sys.exit(1)

        logger.info("🚀 Unified Bitcoin Trading Bot System is ready!")
        logger.info("📈 Starting automated trading...")

        # Run main trading loop
        bot_manager.run_main_loop()

    except KeyboardInterrupt:
        logger.info("⌨️ Keyboard interrupt in main")

    except Exception as e:
        logger.error(f"💥 Fatal error in main: {e}", exc_info=True)

    finally:
        # Ensure cleanup happens
        if bot_manager:
            bot_manager.shutdown()

        logger.info("👋 Unified Bitcoin Trading Bot System shutdown complete")


if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "status":
            run_status_check()
            sys.exit(0)
        elif sys.argv[1] == "help":
            print("Unified Bitcoin Trading Bot Commands:")
            print("  python main.py          - Run the full trading bot")
            print("  python main.py status   - Quick status check")
            print("  python main.py help     - Show this help")
            sys.exit(0)

    # Run the main bot
    main()
