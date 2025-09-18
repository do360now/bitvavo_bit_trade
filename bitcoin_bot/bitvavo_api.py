import requests
import time
import hmac
import hashlib
import json
from typing import Optional, List, Dict, Any
import logging
from tenacity import retry, wait_exponential, stop_after_attempt
from utils.config import BITVAVO_API_KEY, BITVAVO_API_SECRET, API_DOMAIN
import ccxt

logger = logging.getLogger(__name__)

class BitvavoAPI:
    def __init__(self, api_key: str, api_secret: str, api_domain: str = "https://api.bitvavo.com/v2"):
        self.api_key = api_key
        self.api_secret = api_secret.encode('utf-8')
        self.api_domain = api_domain
        self.last_request_time = 0
        self.request_interval = 0.1  # Bitvavo allows higher frequency

    def _get_bitvavo_signature(self, timestamp: str, method: str, url_path: str, data: Dict = None) -> str:
        body = '' if data is None else json.dumps(data, separators=(',', ':'))
        message = timestamp + method + url_path + body
        signature = hmac.new(
            self.api_secret,
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(5))
    def query_public(self, endpoint: str, params: Dict = None) -> Dict:
        try:
            url = f'{self.api_domain}/{endpoint}'
            time_since_last = time.time() - self.last_request_time
            if time_since_last < self.request_interval:
                time.sleep(self.request_interval - time_since_last)
            self.last_request_time = time.time()
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Bitvavo public API request failed: {e}")
            return {'error': str(e)}

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(5))
    def query_private(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        try:
            timestamp = str(int(time.time() * 1000))
            url = f'{self.api_domain}/{endpoint}'

            headers = {
                'Bitvavo-Access-Key': self.api_key,
                'Bitvavo-Access-Signature': self._get_bitvavo_signature(timestamp, method, f'/{endpoint}', data),
                'Bitvavo-Access-Timestamp': timestamp,
                'Content-Type': 'application/json'
            }

            time_since_last = time.time() - self.last_request_time
            if time_since_last < self.request_interval:
                time.sleep(self.request_interval - time_since_last)
            
            self.last_request_time = time.time()

            if method == 'GET':
                response = requests.get(url, headers=headers, params=data)
            else:
                response = requests.post(url, headers=headers, json=data)
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Bitvavo private API request failed: {e}")
            return {'error': str(e)}

    def get_ohlc_data(self, pair: str = "BTC-EUR", interval: str = "15m") -> Optional[List[List]]:
        try:
            params = {
                "market": pair,
                "interval": interval
            }
            result = self.query_public("candles", params)
            if isinstance(result, list):
                return [[
                    float(candle[0]),  # timestamp
                    float(candle[1]),  # open
                    float(candle[2]),  # high
                    float(candle[3]),  # low
                    float(candle[4]),  # close
                    float(candle[5])   # volume
                ] for candle in result]
            return []
        except Exception as e:
            logger.error(f"Failed to fetch OHLC data: {e}")
            return []

    def get_btc_order_book(self) -> Optional[Dict]:
        result = self.query_public("BTC-EUR/book")
        if isinstance(result, dict):
            return result
        return None

    def get_optimal_price(self, order_book: Dict, side: str, buffer: float = 0.05, decimals: int = 1) -> Optional[float]:
        if not order_book:
            return None
        try:
            best_bid = float(order_book['bids'][0][0]) if order_book.get('bids') else None
            best_ask = float(order_book['asks'][0][0]) if order_book.get('asks') else None
            if side == "buy":
                if best_ask is None:
                    return None
                optimal_price = best_ask + buffer
            elif side == "sell":
                if best_bid is None:
                    return None
                optimal_price = best_bid - buffer
            else:
                return None
            return round(optimal_price, decimals)
        except Exception as e:
            logger.error(f"Failed to calculate optimal price: {e}")
            return None

    def get_btc_price(self) -> Optional[float]:
        result = self.query_public("ticker/price", {"market": "BTC-EUR"})
        if isinstance(result, dict) and 'price' in result:
            return float(result['price'])
        return None

    def get_market_volume(self, pair: str = "BTC-EUR") -> Optional[float]:
        result = self.query_public("ticker/24h", {"market": pair})
        if isinstance(result, dict) and 'volume' in result:
            return float(result['volume'])
        return None

    def get_available_balance(self, asset: str) -> Optional[float]:
        result = self.query_private("GET", "balance")
        if isinstance(result, list):
            for balance in result:
                if balance['symbol'] == asset:
                    return float(balance['available'])
        return 0.0

    def get_total_btc_balance(self) -> Optional[float]:
        return self.get_available_balance('BTC')

    def place_order(self, pair: str, side: str, ordertype: str, amount: float, price: float = None) -> Dict:
        data = {
            'market': pair,
            'side': side,
            'orderType': ordertype,
            'amount': str(amount)
        }
        if price is not None:
            data['price'] = str(price)
        return self.query_private('POST', 'order', data)

    def cancel_order(self, order_id: str, pair: str) -> Dict:
        return self.query_private('DELETE', f'order', {'orderId': order_id, 'market': pair})

    def get_open_orders(self, pair: str = None) -> Dict:
        params = {'market': pair} if pair else None
        return self.query_private('GET', 'orders', params)

    def get_order_info(self, order_id: str) -> Dict:
        return self.query_private('GET', f'order', {'orderId': order_id})


def authenticate_bitvavo():
    """
    Authenticate with Bitvavo exchange with improved error handling

    Returns:
        ccxt.bitvavo: Authenticated Bitvavo exchange instance
    """
    try:
        logger.info("Authenticating with Bitvavo...")
        
        api_key, api_secret = load_api_credentials()

        # Create Bitvavo exchange instance
        exchange = ccxt.bitvavo({
            "apiKey": api_key,
            "secret": api_secret,
            "sandbox": False,
            "enableRateLimit": True,
            "options": {
                "adjustForTimeDifference": True,
            },
        })

        # Test the connection
        if test_connection(exchange):
            logger.info("✅ Bitvavo authentication successful")
            return exchange
        else:
            logger.warning("Authentication test failed, proceeding with unverified connection")
            return exchange

    except Exception as e:
        logger.error(f"Failed to create Bitvavo exchange instance: {e}")
        raise


def test_connection(exchange):
    """
    Test exchange connection and basic functionality with comprehensive checks

    Args:
        exchange: CCXT exchange instance

    Returns:
        bool: True if connection is working
    """
    try:
        # Test market data (doesn't require authentication)
        ticker = exchange.fetch_ticker("BTC/EUR")
        logger.info(f"Market data test: BTC/EUR = €{ticker['last']:.2f}")

        # Test authenticated endpoints with improved error handling
        try:
            balance = exchange.fetch_balance()
            if isinstance(balance, dict):
                logger.info("✅ Authenticated API access confirmed")
                return True
            else:
                logger.warning(f"Unexpected balance format: {type(balance)}")
                return True  # Still consider working
                
        except Exception as balance_error:
            logger.warning(f"Balance check failed: {balance_error}")
            # Check if it's just a permission issue vs connection issue
            try:
                # Try a simpler authenticated call
                exchange.fetch_trading_fees()
                logger.info("✅ Basic authenticated access working")
                return True
            except Exception:
                logger.error("All authenticated calls failed")
                return False

    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False
    

def load_api_credentials():
    """
    Centralized credential loading with fallback mechanisms
    """
    # Try different import paths for config
    api_key = None
    api_secret = None
    
    try:
        from utils.config import BITVAVO_API_KEY, BITVAVO_API_SECRET
        api_key = BITVAVO_API_KEY
        api_secret = BITVAVO_API_SECRET
    except ImportError:
        # Fallback to environment variables
        api_key = os.getenv("BITVAVO_API_KEY")
        api_secret = os.getenv("BITVAVO_API_SECRET")
    
    if not api_key or not api_secret:
        raise ValueError(
            "BITVAVO_API_KEY and BITVAVO_API_SECRET must be set in config file or environment variables"
        )
    
    return api_key, api_secret
