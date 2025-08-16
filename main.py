import time
import signal
import sys
import json

from trading_bot import TradingBot
from enhanced_trading_bot import enhance_existing_bot
from complete_integration import UltimateAdaptiveBot
from data_manager import DataManager
from trade_executor import TradeExecutor
from onchain_analyzer import OnChainAnalyzer
from bitvavo_api import authenticate_exchange  # Fixed import
from config import BITVAVO_API_KEY, BITVAVO_API_SECRET, PRICE_HISTORY_FILE, BOT_LOGS_FILE
from logger_config import logger
from order_manager import OrderManager

def main():
    logger.info("Starting Bitcoin accumulation bot with Bitvavo...")

    # Initialize components
    bitvavo = authenticate_exchange()
    order_manager = OrderManager(bitvavo)
    data_manager = DataManager(PRICE_HISTORY_FILE, BOT_LOGS_FILE)
    trade_executor = TradeExecutor(bitvavo)  # Pass bitvavo instead of kraken
    onchain_analyzer = OnChainAnalyzer()
    bot = TradingBot(data_manager, trade_executor, onchain_analyzer, order_manager)
    # After initializing your original bot
    original_bot = TradingBot(data_manager, trade_executor, onchain_analyzer, order_manager)

    # Enhance it with adaptive learning
    enhanced_bot = enhance_existing_bot(original_bot)

    # Initialize the ultimate adaptive bot
    ultimate_bot = UltimateAdaptiveBot(data_manager, trade_executor, onchain_analyzer, order_manager)

    # Set up signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal, saving state...")
        order_manager._save_order_history()
        logger.info("Order history saved. Exiting.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run bot in a loop
    while True:
        try:
            bot.check_pending_orders()
            ultimate_bot.execute_ultimate_strategy()
            current_time = time.time()
            next_run = ((current_time // 900) + 1) * 900
            sleep_time = next_run - current_time
            logger.debug(f"Sleeping for {sleep_time:.2f} seconds until {time.ctime(next_run)}")
            time.sleep(sleep_time)

            # Get detailed performance summary
            summary = enhanced_bot.get_performance_summary()
            print(json.dumps(summary, indent=2))

            # Print real-time status
            enhanced_bot.print_enhanced_status()

        except Exception as e:
            logger.error(f"Main loop error: {e}", exc_info=True)
            logger.info("Saving order history before retrying...")
            order_manager._save_order_history()
            time.sleep(30)  # Wait before retrying

if __name__ == "__main__":
    main()