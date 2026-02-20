"""
Enhanced Configuration with Bitcoin Cycle Awareness

All prices in EUR for BTC-EUR trading on Bitvavo
"""

import os
from dotenv import load_dotenv


# Load environment variables from the .env file
load_dotenv()


# ============================================================================
# API CREDENTIALS
# ============================================================================

# Get API credentials from environment variables
BITVAVO_API_KEY = os.getenv("BITVAVO_API_KEY") or os.getenv("API_KEY")
BITVAVO_API_SECRET = os.getenv("BITVAVO_API_SECRET") or os.getenv("API_SECRET")

# Ensure critical environment variables are set
if not BITVAVO_API_KEY or not BITVAVO_API_SECRET:
    raise ValueError(
        "BITVAVO_API_KEY or BITVAVO_API_SECRET (or API_KEY/API_SECRET) is missing. "
        "Please check your .env file."
    )

# API-related constants
API_DOMAIN = os.getenv("API_DOMAIN", "https://api.bitvavo.com")


# ============================================================================
# PORTFOLIO ALLOCATION
# ============================================================================

# Allocation strategy for portfolio management
ALLOCATIONS = {
    'HODL': float(os.getenv("ALLOC_HODL", "0.9")),      # 90% long-term hold
    'YIELD': float(os.getenv("ALLOC_YIELD", "0.0")),    # 0% staking/yield
    'TRADING': float(os.getenv("ALLOC_TRADING", "0.1")), # 10% active trading
}

# Initial BTC balance (if tracking from specific starting point)
TOTAL_BTC = float(os.getenv("TOTAL_BTC", "0.0"))


# ============================================================================
# TRADING LIMITS & CONSTRAINTS
# ============================================================================

# Minimum trading volume to avoid very small trades
MIN_TRADE_VOLUME = float(os.getenv("MIN_TRADE_VOLUME", "0.0001"))  # 0.0001 BTC

# Minimum EUR value for trade (Bitvavo minimum is €5)
MIN_EUR_FOR_TRADE = float(os.getenv("MIN_EUR_FOR_TRADE", "15.0"))  # €15 minimum

# Maximum cash allocation (% of portfolio in EUR)
MAX_CASH_ALLOCATION = float(os.getenv("MAX_CASH_ALLOCATION", "0.8"))  # 80% max

# Maximum sell allocation (% of BTC to sell at once)
MAX_SELL_ALLOCATION = float(os.getenv("MAX_SELL_ALLOCATION", "0.5"))  # 50% max

# Minimum profit margin for sell (legacy - overridden by cycle strategy)
MIN_PROFIT_MARGIN = float(os.getenv("MIN_PROFIT_MARGIN", "0.05"))  # 5%


# ============================================================================
# TIMING & COOLDOWNS
# ============================================================================

# Cooldown period in seconds between trades (prevent overtrading)
GLOBAL_TRADE_COOLDOWN = int(os.getenv("GLOBAL_TRADE_COOLDOWN", "180"))  # 3 minutes

# Main loop sleep duration in seconds
SLEEP_DURATION = int(os.getenv("SLEEP_DURATION", "900"))  # 15 minutes

# Balance cache duration (avoid excessive API calls)
BALANCE_CACHE_DURATION = int(os.getenv("BALANCE_CACHE_DURATION", "900"))  # 15 minutes

# On-chain data cache duration
ONCHAIN_CACHE_DURATION = int(os.getenv("ONCHAIN_CACHE_DURATION", "60"))  # 1 minute


# ============================================================================
# BITCOIN NODE (Optional - for on-chain analysis)
# ============================================================================

RPC_USER = os.getenv("RPC_USER", "")
RPC_PASSWORD = os.getenv("RPC_PASSWORD", "")
RPC_HOST = os.getenv("RPC_HOST", "127.0.0.1")
RPC_PORT = os.getenv("RPC_PORT", "8332")

# Known exchange addresses for on-chain tracking
EXCHANGE_ADDRESSES = {
    "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa": "Coinbase",
    "3EktnHQD7RiAE6uzMj2ZifT9YgRrkSgzQX": "Binance1",
    "34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo": "Binance2",
    "3FrSzikNqBgikWgTHixywhXcx57q6H6rHC": "Binance3",
    "3D2oetdNuZUqQHPJmcMDDHYoqkyNVsFk9r": "Bitfinex",
    "1AnwDVbwsLBVwRfqN2x9Eo4YEJSPXo2cwG": "Kraken"
}


# ============================================================================
# FILE PATHS
# ============================================================================

PRICE_HISTORY_FILE = os.getenv("PRICE_HISTORY_FILE", "./price_history.json")
BOT_LOGS_FILE = os.getenv("BOT_LOGS_FILE", "./bot_logs.csv")
BOT_STATE_FILE = os.getenv("BOT_STATE_FILE", "./bot_state.json")
ORDER_HISTORY_FILE = os.getenv("ORDER_HISTORY_FILE", "./order_history.json")
PERFORMANCE_FILE = os.getenv("PERFORMANCE_FILE", "./performance_history.json")
PENDING_UPDATES_FILE = os.getenv("PENDING_UPDATES_FILE", "./pending_updates.json")


# ============================================================================
# LEGACY TRADING PARAMETERS (Overridden by Cycle Strategy if enabled)
# ============================================================================

TRADING_PARAMS = {
    'USE_STOP_LOSS': os.getenv("USE_STOP_LOSS", "true").lower() == "true",
    'STOP_LOSS_PERCENT': float(os.getenv("STOP_LOSS_PERCENT", "0.05")),  # 5% for bear market
    'USE_TAKE_PROFIT': os.getenv("USE_TAKE_PROFIT", "true").lower() == "true",
    'TAKE_PROFIT_PERCENT': float(os.getenv("TAKE_PROFIT_PERCENT", "0.20")),  # 20% for bear market
}


# ============================================================================
# CYCLE-AWARE TRADING CONFIGURATION (NEW!)
# ============================================================================

CYCLE_CONFIG = {
    # Master switch - set to False to use legacy strategy
    'ENABLED': os.getenv("CYCLE_AWARE_ENABLED", "true").lower() == "true",
    
    # Current Bitcoin halving cycle (4 = 2024 halving, 3.125 BTC rewards)
    'CURRENT_CYCLE': int(os.getenv("CURRENT_CYCLE", "4")),
    
    # Base position size as % of available capital (before cycle adjustments)
    # Cycle multiplier will adjust this (0.2x to 3.0x based on phase)
    'BASE_POSITION_PCT': float(os.getenv("BASE_POSITION_PCT", "0.10")),  # 10%
    
    # Allow cycle detector to dynamically adjust position sizes
    'CYCLE_ADJUSTMENTS': os.getenv("CYCLE_ADJUSTMENTS", "true").lower() == "true",
    
    # Minimum drawdown from ATH required before buying (safety filter)
    # 0.30 = only buy if price is 30%+ down from all-time high
    'MIN_DRAWDOWN_FOR_BUY': float(os.getenv("MIN_DRAWDOWN_FOR_BUY", "0.30")),  # 30%
    
    # Drawdown threshold for aggressive accumulation
    # 0.70 = buy aggressively when 70%+ down (historical bottom zone)
    'AGGRESSIVE_ACCUMULATION_THRESHOLD': float(
        os.getenv("AGGRESSIVE_ACCUMULATION_THRESHOLD", "0.70")
    ),  # 70%
    
    # Cycle-specific EUR values (for BTC-EUR on Bitvavo)
    # These override if set, otherwise detector uses built-in values
    'CYCLE_ATH_EUR': float(os.getenv("CYCLE_ATH_EUR", "114395")),  # Oct 2025 peak
    'CYCLE_LOW_EUR': float(os.getenv("CYCLE_LOW_EUR", "14069")),   # Nov 2022 bottom
    
    # Risk management overrides
    'STOP_LOSS_OVERRIDE': os.getenv("STOP_LOSS_OVERRIDE", "false").lower() == "true",
    'FIXED_STOP_LOSS_PCT': float(os.getenv("FIXED_STOP_LOSS_PCT", "0.05")),  # 5%
    
    # Print detailed cycle analysis every N iterations
    'PRINT_CYCLE_ANALYSIS_EVERY': int(os.getenv("PRINT_CYCLE_ANALYSIS_EVERY", "10")),
}


# ============================================================================
# ENHANCED RISK MANAGEMENT (Optional - Advanced Features)
# ============================================================================

ENHANCED_RISK_PARAMS = {
    'ENABLE_DYNAMIC_STOPS': os.getenv("ENABLE_DYNAMIC_STOPS", "true").lower() == "true",
    'BASE_STOP_LOSS_PCT': float(os.getenv("BASE_STOP_LOSS_PCT", "0.03")),  # 3%
    'MAX_RISK_OFF_THRESHOLD': float(os.getenv("MAX_RISK_OFF_THRESHOLD", "0.6")),  # 60%
    'HIGH_VOLATILITY_THRESHOLD': float(os.getenv("HIGH_VOLATILITY_THRESHOLD", "0.05")),  # 5%
    'LIQUIDATION_CASCADE_THRESHOLD': float(os.getenv("LIQUIDATION_CASCADE_THRESHOLD", "0.5")),  # 50%
    'MIN_CONFIDENCE_THRESHOLD': float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "60.0")),  # 60%
}


# ============================================================================
# NEWS MONITORING (Optional - requires API keys)
# ============================================================================

NEWS_CONFIG = {
    'ENHANCED_NEWS_ENABLED': os.getenv("ENHANCED_NEWS_ENABLED", "false").lower() == "true",
    'NEWS_CACHE_MINUTES': int(os.getenv("NEWS_CACHE_MINUTES", "30")),
    'MAX_NEWS_ARTICLES': int(os.getenv("MAX_NEWS_ARTICLES", "20")),
    'RISK_OFF_WEIGHT': float(os.getenv("RISK_OFF_WEIGHT", "2.0")),
    'MACRO_NEWS_WEIGHT': float(os.getenv("MACRO_NEWS_WEIGHT", "2.0")),
}


# ============================================================================
# MARKET CORRELATION MONITORING (Optional)
# ============================================================================

CORRELATION_CONFIG = {
    'ENABLE_CORRELATION_MONITORING': os.getenv("ENABLE_CORRELATION_MONITORING", "false").lower() == "true",
    'CORRELATION_LOOKBACK_DAYS': int(os.getenv("CORRELATION_LOOKBACK_DAYS", "30")),
    'HIGH_CORRELATION_THRESHOLD': float(os.getenv("HIGH_CORRELATION_THRESHOLD", "0.7")),
    'CORRELATION_CACHE_MINUTES': int(os.getenv("CORRELATION_CACHE_MINUTES", "15")),
}


# ============================================================================
# POSITION SIZING (Legacy - overridden by cycle strategy)
# ============================================================================

POSITION_SIZING = {
    'BASE_POSITION_PCT': float(os.getenv("BASE_POSITION_PCT", "0.1")),  # 10%
    'MAX_POSITION_PCT': float(os.getenv("MAX_POSITION_PCT", "0.15")),  # 15%
    'MIN_POSITION_PCT': float(os.getenv("MIN_POSITION_PCT", "0.02")),  # 2%
    'RISK_REDUCTION_FACTOR': float(os.getenv("RISK_REDUCTION_FACTOR", "0.5")),
}


# ============================================================================
# LLM CONFIGURATION (Optional - for AI-powered decisions)
# ============================================================================

LLM_CONFIG = {
    'OLLAMA_URL': os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"),
    'MODEL_NAME': os.getenv("MODEL_NAME", "gemma3:4b"),
    'LLM_TIMEOUT_SECONDS': int(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
    'FALLBACK_TO_SIMPLE_LOGIC': os.getenv("FALLBACK_TO_SIMPLE_LOGIC", "true").lower() == "true",
}


# ============================================================================
# MARKET DATA SOURCES (Optional - for correlation analysis)
# ============================================================================

MARKET_DATA_CONFIG = {
    'YAHOO_FINANCE_ENABLED': os.getenv("YAHOO_FINANCE_ENABLED", "false").lower() == "true",
    'BITCOIN_PAIR': 'BTC-USD',
    'SPY_TICKER': 'SPY',
    'DXY_TICKER': 'DXY=X',
    'GOLD_TICKER': 'GC=F',
}


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "trading_bot.log")


# ============================================================================
# DISPLAY CONFIGURATION ON STARTUP
# ============================================================================

def print_config_summary():
    """Print configuration summary on startup"""
    from logger_config import logger
    
    logger.info("=" * 70)
    logger.info("BITCOIN TRADING BOT CONFIGURATION")
    logger.info("=" * 70)
    logger.info(f"API Domain: {API_DOMAIN}")
    logger.info(f"Trading Market: BTC-EUR (Bitvavo)")
    logger.info("")
    logger.info("Portfolio Allocation:")
    logger.info(f"  HODL: {ALLOCATIONS['HODL']:.0%}")
    logger.info(f"  Yield: {ALLOCATIONS['YIELD']:.0%}")
    logger.info(f"  Trading: {ALLOCATIONS['TRADING']:.0%}")
    logger.info("")
    logger.info("Trading Limits:")
    logger.info(f"  Min Trade Volume: {MIN_TRADE_VOLUME:.8f} BTC")
    logger.info(f"  Min Trade Value: €{MIN_EUR_FOR_TRADE:.2f} EUR")
    logger.info(f"  Max Cash Allocation: {MAX_CASH_ALLOCATION:.0%}")
    logger.info(f"  Max Sell Allocation: {MAX_SELL_ALLOCATION:.0%}")
    logger.info("")
    logger.info("Cycle-Aware Trading:")
    if CYCLE_CONFIG['ENABLED']:
        logger.info(f"  ✅ ENABLED - Cycle {CYCLE_CONFIG['CURRENT_CYCLE']}")
        logger.info(f"  Base Position: {CYCLE_CONFIG['BASE_POSITION_PCT']:.0%}")
        logger.info(f"  Min Drawdown for Buy: {CYCLE_CONFIG['MIN_DRAWDOWN_FOR_BUY']:.0%}")
        logger.info(f"  Aggressive Threshold: {CYCLE_CONFIG['AGGRESSIVE_ACCUMULATION_THRESHOLD']:.0%}")
        logger.info(f"  Cycle ATH: €{CYCLE_CONFIG['CYCLE_ATH_EUR']:,.0f} EUR")
        logger.info(f"  Cycle Low: €{CYCLE_CONFIG['CYCLE_LOW_EUR']:,.0f} EUR")
    else:
        logger.info(f"  ❌ DISABLED - Using legacy strategy")
        logger.info(f"  Stop Loss: {TRADING_PARAMS['STOP_LOSS_PERCENT']:.0%}")
        logger.info(f"  Take Profit: {TRADING_PARAMS['TAKE_PROFIT_PERCENT']:.0%}")
    logger.info("")
    logger.info("Timing:")
    logger.info(f"  Loop Interval: {SLEEP_DURATION//60} minutes")
    logger.info(f"  Trade Cooldown: {GLOBAL_TRADE_COOLDOWN//60} minutes")
    logger.info("")
    logger.info("Log Level: " + LOG_LEVEL)
    logger.info("=" * 70)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration values"""
    errors = []
    
    # Validate allocations sum to 1.0
    total_allocation = sum(ALLOCATIONS.values())
    if not (0.99 <= total_allocation <= 1.01):
        errors.append(f"Allocations sum to {total_allocation:.2%}, should be 100%")
    
    # Validate percentages are in valid range
    if not (0 <= CYCLE_CONFIG['BASE_POSITION_PCT'] <= 1):
        errors.append(f"BASE_POSITION_PCT must be between 0 and 1")
    
    if not (0 <= CYCLE_CONFIG['MIN_DRAWDOWN_FOR_BUY'] <= 1):
        errors.append(f"MIN_DRAWDOWN_FOR_BUY must be between 0 and 1")
    
    # Validate positive values
    if MIN_EUR_FOR_TRADE < 5.0:
        errors.append(f"MIN_EUR_FOR_TRADE must be at least €5 (Bitvavo minimum)")
    
    if errors:
        from logger_config import logger
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        raise ValueError("Invalid configuration")
    
    return True


if __name__ == "__main__":
    # Validate and print config when run directly
    validate_config()
    print_config_summary()