#!/usr/bin/env python3
"""
Enhanced Bitcoin Trading Bot - Main Entry Point
Consolidated from multiple bot variants with Bitvavo API integration
"""

import time
import signal
import sys
import json
import os
import asyncio
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import our components
from bitvavo_api import authenticate_bitvavo, test_connection
from bot import EnhancedTradingBot, BotConfiguration


class TradingBotManager:
    """Enhanced bot manager for the unified system"""

    def __init__(self, mode="enhanced"):
        self.bot = None
        self.running = False
        self.iteration_count = 0
        self.start_time = time.time()
        self.bitvavo_api = None
        self.mode = mode

    def initialize_bot(self):
        """Initialize the trading bot"""
        try:
            logger.info(f"🚀 Initializing Bitcoin Trading Bot (Mode: {self.mode})...")

            # Authenticate with Bitvavo
            logger.info("🔐 Authenticating with Bitvavo...")
            self.bitvavo_api = authenticate_bitvavo()

            # Test connection
            if not test_connection(self.bitvavo_api):
                logger.error("❌ Exchange connection test failed")
                return False

            self._check_bitvavo_market_info()
            
            # Create bot configuration
            config = BotConfiguration(
                enable_ml=True,
                enable_peak_detection=True,
                enable_onchain_analysis=True,
                enable_news_sentiment=True,
                max_daily_trades=12,
                base_position_size_pct=0.08,
                stop_loss_pct=0.025,
                take_profit_pct=0.08,
                min_confidence_threshold=0.25,
            )

            # Initialize the enhanced bot
            logger.info("🤖 Creating enhanced trading bot...")
            self.bot = EnhancedTradingBot(self.bitvavo_api, config)

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
            logger.info(f"   💰 BTC Price: €{current_price:.2f}")
            logger.info(f"   ₿🪙 BTC Balance: {btc_balance:.8f}")
            logger.info(f"   💶 EUR Balance: €{eur_balance:.2f}")
            logger.info(f"   💼 Total Value: €{eur_balance + (btc_balance * current_price):.2f}")

            logger.info("✅ Enhanced trading bot initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Bot initialization failed: {e}", exc_info=True)
            return False

    
    def _check_bitvavo_market_info(self):
        """Check Bitvavo market information for BTC/EUR"""
        try:
            markets = self.bitvavo_api.load_markets()
            btc_eur = markets.get('BTC/EUR')
            
            logger.info("\n" + "="*70)
            logger.info("BTC/EUR Market Information")
            logger.info("="*70)
            logger.info(f"Precision - Price: {btc_eur.get('precision', {}).get('price')}")
            logger.info(f"Precision - Amount: {btc_eur.get('precision', {}).get('amount')}")
            logger.info(f"\nLimits:")
            limits = btc_eur.get('limits', {})
            logger.info(f"  Amount Min: {limits.get('amount', {}).get('min')}")
            logger.info(f"  Price Min: {limits.get('price', {}).get('min')}")
            logger.info(f"  Price Max: {limits.get('price', {}).get('max')}")
            logger.info(f"  Cost Min: {limits.get('cost', {}).get('min')}")
            logger.info("="*70)
            
        except Exception as e:
            logger.error(f"Error loading market info: {e}")
    
    
    
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

        logger.info("🔥 Starting main trading loop...")

        while self.running:
            try:
                self.iteration_count += 1
                iteration_start = time.time()

                logger.info(f"\n{'='*80}")
                logger.info(f"🔥 ITERATION #{self.iteration_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"{'='*80}")

                # Execute the enhanced strategy
                self.bot.execute_enhanced_strategy()

                # Reset consecutive error counter on successful iteration
                consecutive_errors = 0

                # Log iteration performance
                iteration_time = time.time() - iteration_start
                uptime = time.time() - self.start_time
                logger.info(f"⏱️ Iteration completed in {iteration_time:.2f}s (uptime: {uptime/3600:.1f}h)")

                # Print status every 4 iterations (1 hour with 15min intervals)
                if self.iteration_count % 4 == 0:
                    try:
                        self.bot.print_enhanced_status()

                        # Maintenance every 24 iterations (6 hours)
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
                logger.error(f"❌ Main loop error #{consecutive_errors}: {e}", exc_info=True)

                # Emergency save state
                self._emergency_save_state()

                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"💥 Too many consecutive errors ({consecutive_errors}), shutting down")
                    break

                # Progressive backoff on errors
                error_sleep = min(300, 30 * consecutive_errors)  # 30s to 5min max
                logger.info(f"⏳ Waiting {error_sleep}s before retry...")
                time.sleep(error_sleep)

        logger.info("🔥 Main trading loop ended")

    def _periodic_maintenance(self):
        """Periodic maintenance tasks"""
        try:
            logger.info("🔧 Performing periodic maintenance...")

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
            logger.error(f"❌ Periodic maintenance failed: {e}")

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

    def run_diagnostic_check(self):
        """Run diagnostic check"""
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
                logger.info(f"✅ Order manager: {len(pending)} pending, {stats['fill_rate']:.1%} fill rate")

            # Test data manager
            prices, volumes = self.bot.data_manager.load_price_history()
            logger.info(f"✅ Data manager: {len(prices)} price points loaded")

            # Test enhanced features
            features = self.bot.available_features
            for feature, available in features.items():
                status = "✅" if available else "⚠️"
                logger.info(f"{status} {feature.replace('_', ' ').title()}: {'Available' if available else 'Fallback mode'}")

            logger.info("✅ All diagnostic checks passed")
            return True

        except Exception as e:
            logger.error(f"❌ Diagnostic check failed: {e}", exc_info=True)
            return False

    def shutdown(self):
        """Graceful shutdown procedure"""
        logger.info("🛑 Initiating graceful shutdown...")

        self.running = False

        try:
            if self.bot:
                self.bot.shutdown()

            # Calculate total runtime
            total_runtime = time.time() - self.start_time
            logger.info(f"⏱️ Total runtime: {total_runtime/3600:.2f} hours ({self.iteration_count} iterations)")

            logger.info("✅ Graceful shutdown completed")
            sys.exit(0)

        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}", exc_info=True)


def run_status_check():
    """Quick status check"""
    try:
        logger.info("🔍 Running status check...")

        # Quick connectivity test
        bitvavo = authenticate_bitvavo()
        
        from bot import TradeExecutor
        executor = TradeExecutor(bitvavo)

        current_price, _ = executor.fetch_current_price()
        btc_balance = executor.get_total_btc_balance()
        eur_balance = executor.get_available_balance("EUR")

        print(f"\n{'='*70}")
        print("📊 BITCOIN TRADING BOT STATUS CHECK")
        print(f"{'='*70}")
        print(f"💰 BTC Price: €{current_price:.2f}")
        print(f"₿🪙 BTC Balance: {btc_balance:.8f}")
        print(f"💶 EUR Balance: €{eur_balance:.2f}")
        print(f"💼 Total Value: €{eur_balance + (btc_balance * current_price):.2f}")

        # Check for saved states
        if os.path.exists("./enhanced_bot_state.json"):
            with open("./enhanced_bot_state.json", "r") as f:
                state = json.load(f)
                print(f"📊 Session Trades: {state.get('total_trades', 0)}")
                print(f"🎯 Enhanced Decisions: {state.get('enhanced_decisions', 0)}")

        print("🔧 Enhanced Features: Available with risk management")
        print(f"{'='*70}\n")

    except Exception as e:
        logger.error(f"Status check failed: {e}")


def main():
    """Main entry point"""
    bot_manager = None

    try:
        # Check for command line arguments
        mode = "enhanced"
        if len(sys.argv) > 1:
            if sys.argv[1] == "status":
                run_status_check()
                sys.exit(0)
            elif sys.argv[1] == "help":
                print("Bitcoin Trading Bot Commands:")
                print("  python main.py          - Run the enhanced trading bot")
                print("  python main.py status   - Quick status check")
                print("  python main.py help     - Show this help")
                sys.exit(0)
            else:
                mode = sys.argv[1]

        # Create bot manager
        bot_manager = TradingBotManager(mode=mode)

        # Setup signal handlers for graceful shutdown
        bot_manager.setup_signal_handlers()

        # Initialize the bot
        if not bot_manager.initialize_bot():
            logger.error("💥 Bot initialization failed, exiting")
            sys.exit(1)

        # Run diagnostic check
        if not bot_manager.run_diagnostic_check():
            logger.error("💥 Diagnostic check failed, exiting")
            sys.exit(1)

        logger.info("🚀 Enhanced Bitcoin Trading Bot System is ready!")
        logger.info("🔥 Starting automated trading...")

        # Run main trading loop
        bot_manager.run_main_loop()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt in main")

    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)

    finally:
        # Ensure cleanup happens
        if bot_manager:
            bot_manager.shutdown()

        logger.info("Bitcoin Trading Bot System shutdown complete")


if __name__ == "__main__":
    main()