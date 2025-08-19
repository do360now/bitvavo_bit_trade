import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Bitvavo API credentials
BITVAVO_API_KEY = os.getenv("BITVAVO_API_KEY")
BITVAVO_API_SECRET = os.getenv("BITVAVO_API_SECRET")

# Ensure critical environment variables are set
if not BITVAVO_API_KEY or not BITVAVO_API_SECRET:
    raise ValueError(
        "BITVAVO_API_KEY or BITVAVO_API_SECRET is missing. Please check your environment variables."
    )

# API-related constants for Bitvavo
API_DOMAIN = "https://api.bitvavo.com"

# Allocation strategy for portfolio management
ALLOCATIONS = {
    "HODL": float(os.getenv("ALLOC_HODL", "0.7")),
    "YIELD": float(os.getenv("ALLOC_YIELD", "0.2")),
    "TRADING": float(os.getenv("ALLOC_TRADING", "0.1")),
}

# Initial BTC balance
TOTAL_BTC = float(os.getenv("TOTAL_BTC", "0.0"))

# Minimum trading volume to avoid very small trades
MIN_TRADE_VOLUME = float(os.getenv("MIN_TRADE_VOLUME", "0.0001"))

# Cooldown period in seconds between trades
GLOBAL_TRADE_COOLDOWN = int(os.getenv("GLOBAL_TRADE_COOLDOWN", "180"))  # 3 minutes

SLEEP_DURATION = int(os.getenv("SLEEP_DURATION", "900"))  # 15 minutes

# RPC settings for Bitcoin node (if used)
RPC_USER = os.getenv("RPC_USER")
RPC_PASSWORD = os.getenv("RPC_PASSWORD")
RPC_HOST = os.getenv("RPC_HOST", "localhost")
RPC_PORT = os.getenv("RPC_PORT", "8332")

# Exchange addresses for on-chain analysis
EXCHANGE_ADDRESSES = {
    "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa": "Coinbase",
    "3EktnHQD7RiAE6uzMj2ZifT9YgRrkSgzQX": "Binance1",
    "34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo": "Binance2",
    "3FrSzikNqBgikWgTHixywhXcx57q6H6rHC": "Binance3",
    "3D2oetdNuZUqQHPJmcMDDHYoqkyNVsFk9r": "Bitfinex",
    "1AnwDVbwsLBVwRfqN2x9Eo4YEJSPXo2cwG": "Kraken",
    # Add Bitvavo addresses if known
    "3BitvavoAddressHere": "Bitvavo",
}

# File paths
PRICE_HISTORY_FILE = "./price_history.json"
BOT_LOGS_FILE = "./bot_logs.csv"

# Trading thresholds
MIN_BTC = 10  # Minimum BTC amount in satoshis
MIN_EUR_FOR_TRADE = 15.0  # Minimum EUR amount for trades
MAX_CASH_ALLOCATION = 0.8  # Maximum cash allocation
MAX_SELL_ALLOCATION = 0.5  # Sell up to 50% of BTC holdings
MIN_PROFIT_MARGIN = 0.05  # Minimum 5% profit for sell

# Cache durations
BALANCE_CACHE_DURATION = 900  # 15 minutes
ONCHAIN_CACHE_DURATION = 60  # 1 minute

# Trading parameters
TRADING_PARAMS = {
    "USE_STOP_LOSS": os.getenv("USE_STOP_LOSS", "true").lower() == "true",
    "STOP_LOSS_PERCENT": float(os.getenv("STOP_LOSS_PERCENT", "0.03")),
    "USE_TAKE_PROFIT": os.getenv("USE_TAKE_PROFIT", "true").lower() == "true",
    "TAKE_PROFIT_PERCENT": float(os.getenv("TAKE_PROFIT_PERCENT", "0.08")),
}

# Enhanced risk management parameters
ENHANCED_RISK_PARAMS = {
    "ENABLE_DYNAMIC_STOPS": os.getenv("ENABLE_DYNAMIC_STOPS", "true").lower() == "true",
    "BASE_STOP_LOSS_PCT": float(os.getenv("BASE_STOP_LOSS_PCT", "0.03")),  # 3%
    "MAX_RISK_OFF_THRESHOLD": float(os.getenv("MAX_RISK_OFF_THRESHOLD", "0.6")),  # 60%
    "HIGH_VOLATILITY_THRESHOLD": float(
        os.getenv("HIGH_VOLATILITY_THRESHOLD", "0.05")
    ),  # 5%
    "LIQUIDATION_CASCADE_THRESHOLD": float(
        os.getenv("LIQUIDATION_CASCADE_THRESHOLD", "0.5")
    ),  # 50%
    "MIN_CONFIDENCE_THRESHOLD": float(
        os.getenv("MIN_CONFIDENCE_THRESHOLD", "60.0")
    ),  # 60%
}

# Enhanced news monitoring
NEWS_CONFIG = {
    "ENHANCED_NEWS_ENABLED": os.getenv("ENHANCED_NEWS_ENABLED", "true").lower()
    == "true",
    "NEWS_CACHE_MINUTES": int(os.getenv("NEWS_CACHE_MINUTES", "30")),
    "MAX_NEWS_ARTICLES": int(os.getenv("MAX_NEWS_ARTICLES", "20")),
    "RISK_OFF_WEIGHT": float(os.getenv("RISK_OFF_WEIGHT", "2.0")),
    "MACRO_NEWS_WEIGHT": float(os.getenv("MACRO_NEWS_WEIGHT", "2.0")),
}

# Market correlation monitoring
CORRELATION_CONFIG = {
    "ENABLE_CORRELATION_MONITORING": os.getenv(
        "ENABLE_CORRELATION_MONITORING", "true"
    ).lower()
    == "true",
    "CORRELATION_LOOKBACK_DAYS": int(os.getenv("CORRELATION_LOOKBACK_DAYS", "30")),
    "HIGH_CORRELATION_THRESHOLD": float(os.getenv("HIGH_CORRELATION_THRESHOLD", "0.7")),
    "CORRELATION_CACHE_MINUTES": int(os.getenv("CORRELATION_CACHE_MINUTES", "15")),
}

# Enhanced position sizing
POSITION_SIZING = {
    "BASE_POSITION_PCT": float(
        os.getenv("BASE_POSITION_PCT", "0.1")
    ),  # 10% of portfolio
    "MAX_POSITION_PCT": float(os.getenv("MAX_POSITION_PCT", "0.15")),  # 15% max
    "MIN_POSITION_PCT": float(os.getenv("MIN_POSITION_PCT", "0.02")),  # 2% min
    "RISK_REDUCTION_FACTOR": float(os.getenv("RISK_REDUCTION_FACTOR", "0.5")),
}

# LLM Configuration
LLM_CONFIG = {
    "OLLAMA_URL": os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"),
    "MODEL_NAME": os.getenv("MODEL_NAME", "gemma3:4b"),
    "LLM_TIMEOUT_SECONDS": int(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
    "FALLBACK_TO_SIMPLE_LOGIC": os.getenv("FALLBACK_TO_SIMPLE_LOGIC", "true").lower()
    == "true",
}

# Market data sources
MARKET_DATA_CONFIG = {
    "YAHOO_FINANCE_ENABLED": True,  # For correlation data
    "BITCOIN_PAIR": "BTC/EUR",  # Changed from BTC-EUR to match Bitvavo
    "SPY_TICKER": "SPY",
    "DXY_TICKER": "DXY=X",
    "GOLD_TICKER": "GC=F",
}

# Bitvavo specific settings
BITVAVO_CONFIG = {
    "TRADING_PAIR": "BTC/EUR",
    "BASE_CURRENCY": "EUR",
    "QUOTE_CURRENCY": "BTC",
    "MIN_ORDER_SIZE": 0.0001,  # Minimum BTC order size on Bitvavo
    "MAKER_FEE": 0.0015,  # 0.15% maker fee
    "TAKER_FEE": 0.0025,  # 0.25% taker fee
}

# Backwards compatibility (in case some code still references these)
API_KEY = BITVAVO_API_KEY
API_SECRET = BITVAVO_API_SECRET
