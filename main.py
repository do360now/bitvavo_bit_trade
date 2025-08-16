import time
import signal
import sys
import json
import traceback
from datetime import datetime

from trading_bot import TradingBot
from enhanced_trading_bot import enhance_existing_bot
from complete_integration import UltimateAdaptiveBot
from data_manager import DataManager
from trade_executor import TradeExecutor
from onchain_analyzer import OnChainAnalyzer
from bitvavo_api import authenticate_exchange
from config import BITVAVO_API_KEY, BITVAVO_API_SECRET, PRICE_HISTORY_FILE, BOT_LOGS_FILE
from logger_config import logger
from order_manager import OrderManager

class TradingBotManager:
    """Manager class to handle bot lifecycle and monitoring"""
    
    def __init__(self):
        self.bot = None
        self.enhanced_bot = None
        self.ultimate_bot = None
        self.running = False
        self.iteration_count = 0
        self.start_time = time.time()
        
    def initialize_bots(self):
        """Initialize all bot components with proper error handling"""
        try:
            logger.info("🚀 Initializing Bitcoin Trading Bot System...")
            
            # Authenticate with Bitvavo
            logger.info("🔐 Authenticating with Bitvavo...")
            bitvavo = authenticate_exchange()
            logger.info("✅ Bitvavo authentication successful")
            
            # Initialize core components
            logger.info("🔧 Initializing core components...")
            data_manager = DataManager(PRICE_HISTORY_FILE, BOT_LOGS_FILE)
            trade_executor = TradeExecutor(bitvavo)
            onchain_analyzer = OnChainAnalyzer()
            order_manager = OrderManager(bitvavo)
            
            # Test basic connectivity
            logger.info("🧪 Testing connectivity...")
            current_price, _ = trade_executor.fetch_current_price()
            if not current_price:
                raise Exception("Failed to fetch current price - check API connectivity")
            
            btc_balance = trade_executor.get_total_btc_balance()
            eur_balance = trade_executor.get_available_balance("EUR")
            logger.info(f"💰 Current balances: {btc_balance:.8f} BTC, €{eur_balance:.2f} EUR")
            
            # Initialize bots
            logger.info("🤖 Initializing trading bots...")
            self.bot = TradingBot(data_manager, trade_executor, onchain_analyzer, order_manager)
            
            # Enhanced bot with ML capabilities
            self.enhanced_bot = enhance_existing_bot(self.bot)
            
            # Ultimate adaptive bot
            self.ultimate_bot = UltimateAdaptiveBot(data_manager, trade_executor, onchain_analyzer, order_manager)
            
            logger.info("✅ All bots initialized successfully")
            
            # Print initial status
            self.bot.print_trading_status()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize bots: {e}", exc_info=True)
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
        """Main trading loop with comprehensive error handling"""
        self.running = True
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        logger.info("🔄 Starting main trading loop...")
        
        while self.running:
            try:
                self.iteration_count += 1
                iteration_start = time.time()
                
                logger.info(f"\n{'='*60}")
                logger.info(f"🔄 ITERATION #{self.iteration_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"{'='*60}")
                
                # Check pending orders first
                self.bot.check_pending_orders()
                
                # Execute strategy based on configuration
                try:
                    if hasattr(self, 'ultimate_bot') and self.ultimate_bot:
                        logger.info("🧠 Running Ultimate Adaptive Strategy...")
                        self.ultimate_bot.execute_ultimate_strategy()
                    elif hasattr(self, 'enhanced_bot') and self.enhanced_bot:
                        logger.info("📈 Running Enhanced Strategy...")
                        self.enhanced_bot.execute_enhanced_strategy()
                    else:
                        logger.info("⚡ Running Standard Strategy...")
                        self.bot.execute_strategy()
                        
                except Exception as strategy_error:
                    logger.error(f"❌ Strategy execution failed: {strategy_error}", exc_info=True)
                    # Continue with order checking even if strategy fails
                
                # Reset consecutive error counter on successful iteration
                consecutive_errors = 0
                
                # Log iteration performance
                iteration_time = time.time() - iteration_start
                uptime = time.time() - self.start_time
                logger.info(f"⏱️ Iteration completed in {iteration_time:.2f}s (uptime: {uptime/3600:.1f}h)")
                
                # Print detailed status every 4 iterations (1 hour with 15min intervals)
                if self.iteration_count % 4 == 0:
                    try:
                        if hasattr(self, 'ultimate_bot') and self.ultimate_bot:
                            self.ultimate_bot.print_comprehensive_status()
                        elif hasattr(self, 'enhanced_bot') and self.enhanced_bot:
                            self.enhanced_bot.print_enhanced_status()
                        else:
                            self.bot.print_trading_status()
                            
                        # Cleanup old data periodically
                        if self.iteration_count % 24 == 0:  # Every 6 hours
                            self._periodic_cleanup()
                            
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
                
                # Save state before potential shutdown
                self._emergency_save_state()
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"💥 Too many consecutive errors ({consecutive_errors}), shutting down")
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
                next_run_str = datetime.fromtimestamp(next_run).strftime('%H:%M:%S')
                logger.info(f"😴 Sleeping for {sleep_time:.0f}s until {next_run_str}")
                time.sleep(sleep_time)
            
        except Exception as e:
            logger.error(f"Sleep calculation error: {e}")
            time.sleep(900)  # Default 15 minutes
    
    def _emergency_save_state(self):
        """Save critical state in case of emergency shutdown"""
        try:
            if self.bot and hasattr(self.bot, 'order_manager') and self.bot.order_manager:
                self.bot.order_manager._save_order_history()
                logger.info("💾 Emergency state save completed")
        except Exception as e:
            logger.error(f"Emergency save failed: {e}")
    
    def _periodic_cleanup(self):
        """Perform periodic cleanup tasks"""
        try:
            logger.info("🧹 Performing periodic cleanup...")
            
            # Cleanup old orders
            if self.bot and hasattr(self.bot, 'order_manager') and self.bot.order_manager:
                self.bot.order_manager.cleanup_old_orders(days=30)
            
            # Force sync order state
            if self.bot and hasattr(self.bot, 'order_manager') and self.bot.order_manager:
                self.bot.order_manager.force_refresh_all_orders()
            
            logger.info("✅ Periodic cleanup completed")
            
        except Exception as e:
            logger.error(f"Periodic cleanup failed: {e}")
    
    def shutdown(self):
        """Graceful shutdown procedure"""
        logger.info("🛑 Initiating graceful shutdown...")
        
        self.running = False
        
        try:
            # Save all state
            if self.bot and hasattr(self.bot, 'order_manager') and self.bot.order_manager:
                logger.info("💾 Saving order history...")
                self.bot.order_manager._save_order_history()
            
            if self.bot and hasattr(self.bot, '_save_recent_buys'):
                logger.info("💾 Saving recent buys...")
                self.bot._save_recent_buys()
            
            if self.bot and hasattr(self.bot, '_save_trade_session'):
                logger.info("💾 Saving trade session...")
                self.bot._save_trade_session()
            
            # Print final status
            if self.bot:
                logger.info("📊 Final status report:")
                self.bot.print_trading_status()
            
            # Calculate total runtime
            total_runtime = time.time() - self.start_time
            logger.info(f"⏱️ Total runtime: {total_runtime/3600:.2f} hours ({self.iteration_count} iterations)")
            
            logger.info("✅ Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)
    
    def run_diagnostic_check(self):
        """Run comprehensive diagnostic check"""
        logger.info("🔍 Running diagnostic check...")
        
        try:
            # Test API connectivity
            if self.bot:
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
                
                # Test indicators
                if len(prices) > 50:
                    from indicators import calculate_rsi, fetch_enhanced_news
                    rsi = calculate_rsi(prices)
                    articles = fetch_enhanced_news(top_n=5)
                    logger.info(f"✅ Indicators: RSI {rsi:.1f}, {len(articles) if articles else 0} news articles")
                
                logger.info("✅ All diagnostic checks passed")
                return True
                
        except Exception as e:
            logger.error(f"❌ Diagnostic check failed: {e}", exc_info=True)
            return False

def main():
    """Main entry point with comprehensive error handling"""
    bot_manager = None
    
    try:
        # Create bot manager
        bot_manager = TradingBotManager()
        
        # Setup signal handlers for graceful shutdown
        bot_manager.setup_signal_handlers()
        
        # Initialize all components
        if not bot_manager.initialize_bots():
            logger.error("💥 Bot initialization failed, exiting")
            sys.exit(1)
        
        # Run diagnostic check
        if not bot_manager.run_diagnostic_check():
            logger.error("💥 Diagnostic check failed, exiting")
            sys.exit(1)
        
        logger.info("🚀 Bitcoin Trading Bot System is ready!")
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
        
        logger.info("👋 Bitcoin Trading Bot System shutdown complete")

def run_status_check():
    """Standalone function to check bot status without running full bot"""
    try:
        logger.info("🔍 Running status check...")
        
        # Quick connectivity test
        from bitvavo_api import authenticate_exchange
        from trade_executor import TradeExecutor
        
        bitvavo = authenticate_exchange()
        executor = TradeExecutor(bitvavo)
        
        current_price, _ = executor.fetch_current_price()
        btc_balance = executor.get_total_btc_balance()
        eur_balance = executor.get_available_balance("EUR")
        
        print(f"\n{'='*50}")
        print(f"📊 QUICK STATUS CHECK")
        print(f"{'='*50}")
        print(f"💰 BTC Price: €{current_price:.2f}")
        print(f"💎 BTC Balance: {btc_balance:.8f}")
        print(f"💶 EUR Balance: €{eur_balance:.2f}")
        print(f"💰 Total Value: €{eur_balance + (btc_balance * current_price):.2f}")
        
        # Check for order history
        if os.path.exists("./order_history.json"):
            with open("./order_history.json", 'r') as f:
                order_data = json.load(f)
                filled_count = len(order_data.get('filled_orders', {}))
                print(f"📋 Historical Orders: {filled_count} filled")
        
        # Check for recent session
        if os.path.exists("./trade_session.json"):
            with open("./trade_session.json", 'r') as f:
                session_data = json.load(f)
                daily_trades = session_data.get('daily_trade_count', 0)
                print(f"📊 Today's Trades: {daily_trades}")
        
        print(f"{'='*50}\n")
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")

if __name__ == "__main__":
    import os
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "status":
            run_status_check()
            sys.exit(0)
        elif sys.argv[1] == "help":
            print("Bitcoin Trading Bot Commands:")
            print("  python main.py          - Run the full trading bot")
            print("  python main.py status   - Quick status check")
            print("  python main.py help     - Show this help")
            sys.exit(0)
    
    # Run the main bot
    main()