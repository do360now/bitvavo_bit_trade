# test_main.py
"""
Test main file for minimal enhanced bot
Use this to verify everything works before using full institutional features
"""

import time
import signal
import sys
from datetime import datetime

try:
    from minimal_enhanced_bot import create_minimal_enhanced_bot
    from bitvavo_api import authenticate_exchange, test_connection
    from core.bot import BotConfiguration
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

try:
    from utils.logger import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class TestBotManager:
    """Simple test bot manager"""

    def __init__(self):
        self.bot = None
        self.running = False
        self.iteration_count = 0

    def initialize_bot(self):
        """Initialize the test bot"""
        try:
            logger.info("🧪 Initializing Test Enhanced Bot...")

            # Authenticate with Bitvavo
            self.bitvavo_api = authenticate_exchange()

            # Test connection
            if not test_connection(self.bitvavo_api):
                logger.error("❌ Exchange connection test failed")
                return False

            # Create test bot
            self.bot = create_minimal_enhanced_bot(self.bitvavo_api)

            # Test basic functionality
            current_price, _ = self.bot.trade_executor.fetch_current_price()
            if not current_price:
                raise Exception("Failed to fetch current price")

            btc_balance = self.bot.trade_executor.get_total_btc_balance()
            eur_balance = self.bot.trade_executor.get_available_balance("EUR")

            logger.info("💰 Current Status:")
            logger.info(f"   💰 BTC Price: €{current_price:.2f}")
            logger.info(f"   ₿ BTC Balance: {btc_balance:.8f}")
            logger.info(f"   💶 EUR Balance: €{eur_balance:.2f}")
            logger.info(f"   💼 Total Value: €{eur_balance + (btc_balance * current_price):.2f}")

            logger.info("✅ Test bot initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Bot initialization failed: {e}", exc_info=True)
            return False

    def run_test_iterations(self, num_iterations=3):
        """Run a few test iterations"""
        logger.info(f"🧪 Running {num_iterations} test iterations...")

        for i in range(num_iterations):
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"🧪 TEST ITERATION #{i+1}/{num_iterations}")
                logger.info(f"{'='*60}")

                # Execute enhanced strategy
                self.bot.execute_enhanced_strategy()

                # Print status
                self.bot.print_enhanced_status()

                if i < num_iterations - 1:  # Don't sleep on last iteration
                    logger.info("⏳ Waiting 30 seconds before next test iteration...")
                    time.sleep(30)

            except Exception as e:
                logger.error(f"❌ Test iteration {i+1} failed: {e}", exc_info=True)

        logger.info("✅ Test iterations completed")

    def run_single_analysis(self):
        """Run single market analysis test"""
        try:
            logger.info("🔍 Running single market analysis test...")

            # Test enhanced market analysis
            indicators = self.bot.enhanced_market_analysis()

            logger.info("📊 Market Analysis Results:")
            logger.info(f"   Price: €{indicators.current_price:.2f}")
            logger.info(f"   RSI: {indicators.rsi:.1f} ({getattr(indicators, 'rsi_signal', 'unknown')})")
            logger.info(f"   Volatility: {indicators.volatility:.4f}")
            
            if hasattr(indicators, 'enhanced_volatility'):
                logger.info(f"   Enhanced Volatility: {indicators.enhanced_volatility:.4f}")
                logger.info(f"   Volatility Regime: {getattr(indicators, 'volatility_regime', 'unknown')}")
            
            if hasattr(indicators, 'volume_ratio'):
                logger.info(f"   Volume Ratio: {indicators.volume_ratio:.2f}x ({getattr(indicators, 'volume_signal', 'unknown')})")

            # Test signal generation
            signal = self.bot.generate_enhanced_signal(indicators)

            logger.info("🎯 Signal Results:")
            logger.info(f"   Action: {signal.action.value}")
            logger.info(f"   Confidence: {signal.confidence:.1%}")
            logger.info(f"   Volume: {signal.volume:.6f}")
            logger.info(f"   Reasoning: {signal.reasoning}")

            return True

        except Exception as e:
            logger.error(f"❌ Single analysis test failed: {e}", exc_info=True)
            return False


def main():
    """Test main function"""
    manager = TestBotManager()

    try:
        # Initialize bot
        if not manager.initialize_bot():
            logger.error("❌ Bot initialization failed")
            sys.exit(1)

        # Run tests based on command line argument
        if len(sys.argv) > 1:
            if sys.argv[1] == "analyze":
                manager.run_single_analysis()
            elif sys.argv[1] == "test":
                iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 3
                manager.run_test_iterations(iterations)
            else:
                print("Usage:")
                print("  python test_main.py analyze     - Run single analysis")
                print("  python test_main.py test [N]    - Run N test iterations (default 3)")
        else:
            # Default: run single analysis
            manager.run_single_analysis()

    except KeyboardInterrupt:
        logger.info("⌨️ Test interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
    finally:
        logger.info("👋 Test completed")


if __name__ == "__main__":
    main()