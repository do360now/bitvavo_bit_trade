#!/usr/bin/env python3
"""
Simple starter script for the unified Bitcoin trading bot
Use this for testing the consolidated system
"""

import sys
import logging

# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test if all imports are working"""
    try:
        logger.info("Testing imports...")

        # Test basic components
        from data_manager import DataManager
        from executor import TradeExecutor
        from order_manager import OrderManager

        logger.info("✅ Basic components imported successfully")

        # Test API authentication
        from bitvavo_api import authenticate_exchange

        logger.info("✅ API authentication module imported")

        # Test unified bot
        from unified_bot import UnifiedTradingBot

        logger.info("✅ Unified bot imported successfully")

        return True

    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False


def test_api_connection():
    """Test API connection without running full bot"""
    try:
        from bitvavo_api import authenticate_exchange, test_connection

        logger.info("Testing Bitvavo API connection...")
        exchange = authenticate_exchange()

        if test_connection(exchange):
            logger.info("✅ API connection successful")
            return True
        else:
            logger.error("❌ API connection failed")
            return False

    except Exception as e:
        logger.error(f"❌ API test error: {e}")
        return False


def main():
    """Main test function"""
    print("🧪 Bitcoin Trading Bot - System Test")
    print("=" * 50)

    # Test imports
    if not test_imports():
        print("❌ Import test failed - check your file structure and imports")
        return False

    # Test API connection
    if not test_api_connection():
        print("❌ API test failed - check your credentials and network")
        return False

    print("✅ All tests passed! Your system is ready.")
    print("\nTo run the full bot:")
    print("  python main.py")
    print("\nTo check status:")
    print("  python main.py status")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
