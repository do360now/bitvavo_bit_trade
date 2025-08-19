#!/usr/bin/env python3
"""
Migration script to help consolidate the Bitcoin trading bot
This script helps you complete the refactoring process
"""

import os
import shutil
import sys
from pathlib import Path

def create_directory_structure():
    """Create the recommended directory structure"""
    dirs = [
        "bitcoin_bot",
        "bitcoin_bot/core",
        "bitcoin_bot/trading", 
        "bitcoin_bot/analysis",
        "bitcoin_bot/utils"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

def move_files_to_structure():
    """Move files to the recommended structure"""
    file_moves = {
        # Core files
        "core/bot.py": "bitcoin_bot/core/bot.py",
        "core/data_manager.py": "bitcoin_bot/core/data_manager.py",
        
        # Trading files  
        "trading/executor.py": "bitcoin_bot/trading/executor.py",
        "trading/order_manager.py": "bitcoin_bot/trading/order_manager.py",
        "trading/strategies.py": "bitcoin_bot/trading/strategies.py",

        # Analysis files
        "analysis/ml_engine.py": "bitcoin_bot/analysis/ml_engine.py",
        "analysis/peak_detection.py": "bitcoin_bot/analysis/peak_detection.py",
        "analysis/indicators.py": "bitcoin_bot/analysis/indicators.py",
        
        # Utils
        "utils/config.py": "bitcoin_bot/utils/config.py",
        "utils/logger.py": "bitcoin_bot/utils/logger.py",
        
        # Root level
        "unified_bot.py": "bitcoin_bot/unified_bot.py",
        "bitvavo_api.py": "bitcoin_bot/bitvavo_api.py"
    }
    
    for src, dst in file_moves.items():
        if os.path.exists(src):
            # Create destination directory if it doesn't exist
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            print(f"✅ Copied {src} -> {dst}")
        else:
            print(f"⚠️  File not found: {src}")

def create_init_files():
    """Create __init__.py files for proper Python package structure"""
    init_files = [
        "bitcoin_bot/__init__.py",
        "bitcoin_bot/core/__init__.py", 
        "bitcoin_bot/trading/__init__.py",
        "bitcoin_bot/analysis/__init__.py",
        "bitcoin_bot/utils/__init__.py"
    ]
    
    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write('"""Bitcoin Trading Bot Package"""\n')
        print(f"✅ Created {init_file}")

def update_import_statements():
    """Generate corrected import statements"""
    import_fixes = """
# For unified_bot.py - use these imports:
from bitcoin_bot.core.bot import TradingBot as CoreBot, BotConfiguration, MarketIndicators, TradingSignal, TradingAction, RiskLevel
from bitcoin_bot.core.data_manager import DataManager
from bitcoin_bot.trading.executor import TradeExecutor
from bitcoin_bot.trading.order_manager import OrderManager
from bitcoin_bot.analysis.ml_engine import MLEngine, MLConfig
from bitcoin_bot.analysis.peak_detection import PeakAvoidanceSystem, PeakDetectionConfig
from bitcoin_bot.trading.strategies import StrategyFactory, StrategyType, StrategyConfig

# For main.py - use these imports:
from bitcoin_bot.unified_bot import UnifiedTradingBot
from bitcoin_bot.bitvavo_api import authenticate_exchange, test_connection, OnChainAnalyzer
from bitcoin_bot.core.bot import BotConfiguration
from bitcoin_bot.utils.config import PRICE_HISTORY_FILE, BOT_LOGS_FILE
from bitcoin_bot.utils.logger import logger

# For other files - update relative imports to absolute imports using bitcoin_bot prefix
"""
    
    with open("import_fixes_reference.txt", 'w') as f:
        f.write(import_fixes)
    
    print("✅ Created import_fixes_reference.txt with corrected import statements")

def create_simple_starter():
    """Create a simplified starter script for testing"""
    starter_content = '''#!/usr/bin/env python3
"""
Simple starter script for the unified Bitcoin trading bot
Use this for testing the consolidated system
"""

import sys
import os
import logging

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
    print("\\nTo run the full bot:")
    print("  python main.py")
    print("\\nTo check status:")
    print("  python main.py status")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
    
    with open("test_system.py", 'w') as f:
        f.write(starter_content)
    
    print("✅ Created test_system.py for testing your setup")

def generate_requirements():
    """Generate requirements.txt based on your codebase"""
    requirements = """# Bitcoin Trading Bot Requirements
ccxt>=4.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
scipy>=1.7.0
python-dotenv>=0.19.0
requests>=2.26.0
nltk>=3.6
yfinance>=0.1.70
tenacity>=8.0.0

# Optional ML libraries
xgboost>=1.5.0
lightgbm>=3.3.0

# Development
pytest>=6.0.0
"""
    
    with open("requirements.txt", 'w') as f:
        f.write(requirements)
    
    print("✅ Created requirements.txt")

def create_env_template():
    """Create .env template file"""
    env_template = """# Bitcoin Trading Bot Environment Variables

# Bitvavo API Credentials (REQUIRED)
BITVAVO_API_KEY=your_api_key_here
BITVAVO_API_SECRET=your_api_secret_here

# News API (Optional - for sentiment analysis)
NEWS_API_KEY=your_news_api_key_here

# Trading Parameters
ALLOC_HODL=0.7
ALLOC_YIELD=0.2
ALLOC_TRADING=0.1
TOTAL_BTC=0.0
MIN_TRADE_VOLUME=0.0001
GLOBAL_TRADE_COOLDOWN=180
SLEEP_DURATION=900

# Risk Management
USE_STOP_LOSS=true
STOP_LOSS_PERCENT=0.03
USE_TAKE_PROFIT=true
TAKE_PROFIT_PERCENT=0.08

# Logging
LOG_LEVEL=INFO
LOG_FILE=trading_bot.log
"""
    
    with open(".env.template", 'w') as f:
        f.write(env_template)
    
    print("✅ Created .env.template (copy to .env and fill in your credentials)")

def print_next_steps():
    """Print what to do next"""
    next_steps = """
🚀 NEXT STEPS FOR CONSOLIDATION:

1. Review the file structure created in bitcoin_bot/ directory
2. Copy .env.template to .env and add your API credentials
3. Update imports in your files using import_fixes_reference.txt
4. Test the system: python test_system.py
5. If tests pass, run: python main.py status
6. Finally, start trading: python main.py

📁 NEW FILE STRUCTURE:
bitcoin_bot/
├── __init__.py
├── unified_bot.py          # Main unified bot
├── bitvavo_api.py          # API authentication
├── main.py                 # Entry point (update this)
├── core/
│   ├── __init__.py
│   ├── bot.py              # Core bot logic
│   └── data_manager.py     # Data management
├── trading/
│   ├── __init__.py
│   ├── executor.py         # Trade execution
│   ├── order_manager.py    # Order management
│   └── strategies.py       # Trading strategies
├── analysis/
│   ├── __init__.py
│   ├── ml_engine.py        # Machine learning
│   ├── peak_detection.py   # Peak detection
│   └── indicators.py       # Technical indicators
└── utils/
    ├── __init__.py
    ├── config.py           # Configuration
    └── logger.py           # Logging setup

🔧 WHAT'S BEEN AUTOMATED:
✅ Directory structure created
✅ Files copied to new structure
✅ __init__.py files created
✅ Import reference guide created
✅ Test script created
✅ Requirements.txt generated
✅ .env template created

⚠️  MANUAL STEPS REQUIRED:
1. Update imports in your files
2. Set up your .env file with API credentials
3. Test the system
4. Update main.py to use the new unified_bot

💡 TIP: Start with test_system.py to verify everything works before running the full bot
"""
    
    print(next_steps)

def main():
    """Main migration function"""
    print("🔄 Bitcoin Trading Bot - Migration Script")
    print("=" * 60)
    
    try:
        print("1. Creating directory structure...")
        create_directory_structure()
        
        print("\n2. Moving files to new structure...")
        move_files_to_structure()
        
        print("\n3. Creating package __init__.py files...")
        create_init_files()
        
        print("\n4. Generating import fixes reference...")
        update_import_statements()
        
        print("\n5. Creating test script...")
        create_simple_starter()
        
        print("\n6. Generating requirements.txt...")
        generate_requirements()
        
        print("\n7. Creating .env template...")
        create_env_template()
        
        print("\n" + "=" * 60)
        print("✅ MIGRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print_next_steps()
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Migration complete! Check the next steps above.")
    else:
        print("\n💥 Migration failed. Check the errors above.")
    
    sys.exit(0 if success else 1)
