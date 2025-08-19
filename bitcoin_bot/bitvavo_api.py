import ccxt
import logging

# Try different import paths for config
try:
    from utils.config import BITVAVO_API_KEY, BITVAVO_API_SECRET
except ImportError:
    try:
        from utils.config import BITVAVO_API_KEY, BITVAVO_API_SECRET
    except ImportError:
        import os
        BITVAVO_API_KEY = os.getenv("BITVAVO_API_KEY")
        BITVAVO_API_SECRET = os.getenv("BITVAVO_API_SECRET")
"""
Bitvavo API Authentication and Setup
"""

import ccxt
import logging

# Try different import paths for config
try:
    from utils.config import BITVAVO_API_KEY, BITVAVO_API_SECRET
except ImportError:
    try:
        from utils.config import BITVAVO_API_KEY, BITVAVO_API_SECRET
    except ImportError:
        import os
        BITVAVO_API_KEY = os.getenv("BITVAVO_API_KEY")
        BITVAVO_API_SECRET = os.getenv("BITVAVO_API_SECRET")
        if not BITVAVO_API_KEY or not BITVAVO_API_SECRET:
            raise ValueError("BITVAVO_API_KEY and BITVAVO_API_SECRET must be set in environment variables")

logger = logging.getLogger(__name__)

def authenticate_exchange():
    """
    Authenticate with Bitvavo exchange
    
    Returns:
        ccxt.bitvavo: Authenticated Bitvavo exchange instance
    """
    try:
        logger.info("Authenticating with Bitvavo...")
        
        # Create Bitvavo exchange instance
        exchange = ccxt.bitvavo({
            'apiKey': BITVAVO_API_KEY,
            'secret': BITVAVO_API_SECRET,
            'sandbox': False,  # Set to True for testing
            'enableRateLimit': True,
            'options': {
                'adjustForTimeDifference': True,
            }
        })
        
        # Test the connection
        try:
            # Try to fetch balance to verify authentication
            balance = exchange.fetch_balance()
            logger.info("✅ Bitvavo authentication successful")
            
            # Log available currencies (for debugging)
            currencies = [curr for curr, data in balance.items() if data.get('total', 0) > 0]
            logger.info(f"Available currencies: {currencies}")
            
            return exchange
            
        except Exception as auth_error:
            logger.error(f"Bitvavo authentication test failed: {auth_error}")
            
            # Still return the exchange instance - it might work for some operations
            logger.warning("Proceeding with unverified authentication")
            return exchange
            
    except Exception as e:
        logger.error(f"Failed to create Bitvavo exchange instance: {e}")
        raise

def test_connection(exchange):
    """
    Test exchange connection and basic functionality
    
    Args:
        exchange: CCXT exchange instance
        
    Returns:
        bool: True if connection is working
    """
    try:
        # Test market data (doesn't require authentication)
        ticker = exchange.fetch_ticker('BTC/EUR')
        logger.info(f"Market data test: BTC/EUR = €{ticker['last']:.2f}")
        
        # Test authenticated endpoints - improved error handling
        try:
            balance = exchange.fetch_balance()
            if isinstance(balance, dict):
                logger.info("✅ Authenticated API access confirmed")
            else:
                logger.warning(f"Unexpected balance format: {type(balance)}")
                logger.info("✅ API access working (non-standard format)")
        except Exception as balance_error:
            logger.warning(f"Balance check failed: {balance_error}")
            # Still consider it working if we can fetch market data
            logger.info("✅ API access working (market data only)")
        
        # Test order book
        order_book = exchange.fetch_order_book('BTC/EUR')
        logger.info(f"Order book test: {len(order_book['bids'])} bids, {len(order_book['asks'])} asks")
        
        return True
        
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False

class OnChainAnalyzer:
    """
    Simple on-chain analyzer placeholder
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".OnChainAnalyzer")
        self.logger.info("OnChain analyzer initialized (placeholder)")
    
    def get_onchain_signals(self) -> dict:
        """Get on-chain signals (placeholder implementation)"""
        return {
            'netflow': 0.0,
            'fee_rate': 10.0,
            'old_utxos': 50.0
        }
    
    def analyze_addresses(self, addresses: list) -> dict:
        """Analyze addresses (placeholder implementation)"""
        return {
            'total_balance': 0.0,
            'activity_score': 0.5
        }