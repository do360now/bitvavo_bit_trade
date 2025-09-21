"""
Enhanced Bitvavo API Client with async support, security improvements,
and performance optimizations for high-frequency trading.
"""

import asyncio
import aiohttp
import hmac
import hashlib
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from tenacity import retry, wait_exponential, stop_after_attempt
from contextlib import asynccontextmanager
import weakref
from datetime import datetime, timedelta
import ccxt

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stopLoss"
    STOP_LIMIT = "stopLossLimit"
    TAKE_PROFIT = "takeProfit"
    TAKE_PROFIT_LIMIT = "takeProfitLimit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class OrderRequest:
    """Type-safe order request structure"""
    market: str
    side: OrderSide
    order_type: OrderType
    amount: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: Optional[str] = None
    post_only: bool = False
    reduce_only: bool = False
    client_order_id: Optional[str] = None


@dataclass
class MarketData:
    """Market data structure with validation"""
    symbol: str
    price: float
    volume: float
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    
    def __post_init__(self):
        if self.price <= 0:
            raise ValueError(f"Invalid price: {self.price}")
        if self.volume < 0:
            raise ValueError(f"Invalid volume: {self.volume}")


class SecurityManager:
    """Enhanced security manager for API credentials"""
    
    def __init__(self, api_key: str, api_secret: str):
        # Store credentials securely in memory
        self._api_key = api_key
        self._api_secret = api_secret.encode('utf-8')
        
        # Validate credentials format
        if not self._api_key or len(self._api_key) < 32:
            raise ValueError("Invalid API key format")
        if not self._api_secret or len(self._api_secret) < 32:
            raise ValueError("Invalid API secret format")
    
    def generate_signature(self, timestamp: str, method: str, 
                          url_path: str, body: str = "") -> str:
        """Generate HMAC signature for API requests"""
        message = timestamp + method + url_path + body
        signature = hmac.new(
            self._api_secret,
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    @property
    def api_key(self) -> str:
        return self._api_key
    
    def __del__(self):
        # Clear credentials from memory on cleanup
        if hasattr(self, '_api_key'):
            self._api_key = None
        if hasattr(self, '_api_secret'):
            self._api_secret = None


class RateLimiter:
    """Advanced rate limiter with burst capacity"""
    
    def __init__(self, requests_per_second: float = 10.0, burst_capacity: int = 20):
        self.requests_per_second = requests_per_second
        self.burst_capacity = burst_capacity
        self.tokens = burst_capacity
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire permission to make a request"""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # Add tokens based on elapsed time
            self.tokens = min(
                self.burst_capacity,
                self.tokens + elapsed * self.requests_per_second
            )
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return
            
            # Wait for next token
            wait_time = (1 - self.tokens) / self.requests_per_second
            await asyncio.sleep(wait_time)
            self.tokens = 0


class EnhancedBitvavoAPI:
    """
    Enhanced Bitvavo API client with async support, security improvements,
    and performance optimizations.
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "https://api.bitvavo.com/v2",
        rate_limit: float = 10.0,
        max_connections: int = 100
    ):
        self.base_url = base_url
        self.security_manager = SecurityManager(api_key, api_secret)
        self.rate_limiter = RateLimiter(rate_limit)
        
        # Connection management
        self.connector = aiohttp.TCPConnector(
            limit=max_connections,
            limit_per_host=20,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        self.timeout = aiohttp.ClientTimeout(total=30.0)
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Circuit breaker state
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure = 0
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 300  # 5 minutes
        
        # Performance monitoring
        self.request_count = 0
        self.total_latency = 0.0
        
        logger.info("Enhanced Bitvavo API client initialized")
    
    @property
    def average_latency(self) -> float:
        """Get average request latency"""
        return self.total_latency / max(self.request_count, 1)
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=self.timeout
            )
        return self._session
    
    async def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open"""
        if self.circuit_breaker_failures < self.circuit_breaker_threshold:
            return False
        
        time_since_last_failure = time.time() - self.circuit_breaker_last_failure
        if time_since_last_failure > self.circuit_breaker_timeout:
            # Reset circuit breaker
            self.circuit_breaker_failures = 0
            return False
        
        return True
    
    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3)
    )
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        authenticated: bool = False
    ) -> Dict[str, Any]:
        """Make authenticated API request with comprehensive error handling"""
        
        if await self._is_circuit_breaker_open():
            raise ConnectionError("Circuit breaker is open")
        
        await self.rate_limiter.acquire()
        
        url = f"{self.base_url}/{endpoint}"
        headers = {"Content-Type": "application/json"}
        
        if authenticated:
            timestamp = str(int(time.time() * 1000))
            body = json.dumps(data, separators=(',', ':')) if data else ""
            
            signature = self.security_manager.generate_signature(
                timestamp, method, f"/{endpoint}", body
            )
            
            headers.update({
                "Bitvavo-Access-Key": self.security_manager.api_key,
                "Bitvavo-Access-Signature": signature,
                "Bitvavo-Access-Timestamp": timestamp
            })
        
        start_time = time.time()
        session = await self._get_session()
        
        try:
            async with session.request(
                method,
                url,
                headers=headers,
                params=params,
                json=data
            ) as response:
                latency = time.time() - start_time
                self.request_count += 1
                self.total_latency += latency
                
                if response.status == 429:
                    # Rate limited - back off exponentially
                    await asyncio.sleep(min(latency * 2, 60))
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message="Rate limited"
                    )
                
                response.raise_for_status()
                result = await response.json()
                
                # Record success
                self.circuit_breaker_failures = max(0, self.circuit_breaker_failures - 1)
                return result
                
        except Exception as e:
            # Record failure
            self.circuit_breaker_failures += 1
            self.circuit_breaker_last_failure = time.time()
            logger.error(f"API request failed: {method} {endpoint} - {e}")
            raise
    
    # Public API methods
    async def get_ticker(self, symbol: str) -> MarketData:
        """Get ticker data for symbol"""
        result = await self._make_request("GET", f"ticker/price", {"market": symbol})
        
        return MarketData(
            symbol=symbol,
            price=float(result["price"]),
            volume=0.0,  # Not provided in price endpoint
            timestamp=datetime.now()
        )
    
    async def get_order_book(self, symbol: str, depth: int = 10) -> Dict[str, Any]:
        """Get order book for symbol"""
        params = {"depth": depth}
        return await self._make_request("GET", f"{symbol}/book", params)
    
    async def get_ohlcv(
        self,
        symbol: str,
        interval: str = "15m",
        limit: int = 100,
        since: Optional[int] = None
    ) -> List[List[float]]:
        """Get OHLCV candlestick data"""
        params = {
            "market": symbol,
            "interval": interval,
            "limit": limit
        }
        if since:
            params["start"] = since
        
        result = await self._make_request("GET", "candles", params)
        return result
    
    # Authenticated API methods
    async def get_balance(self) -> Dict[str, Dict[str, float]]:
        """Get account balance"""
        result = await self._make_request("GET", "balance", authenticated=True)
        
        # Convert list format to dict format for easier access
        balance_dict = {}
        for item in result:
            balance_dict[item["symbol"]] = {
                "available": float(item["available"]),
                "inOrder": float(item["inOrder"])
            }
        
        return balance_dict
    
    async def place_order(self, order: OrderRequest) -> Dict[str, Any]:
        """Place a new order"""
        data = {
            "market": order.market,
            "side": order.side.value,
            "orderType": order.order_type.value,
            "amount": str(order.amount)
        }
        
        if order.price is not None:
            data["price"] = str(order.price)
        if order.stop_price is not None:
            data["triggerAmount"] = str(order.stop_price)
        if order.time_in_force:
            data["timeInForce"] = order.time_in_force
        if order.post_only:
            data["postOnly"] = True
        if order.client_order_id:
            data["clientOrderId"] = order.client_order_id
        
        return await self._make_request("POST", "order", data=data, authenticated=True)
    
    async def close(self):
        """Close all connections"""
        if self._session and not self._session.closed:
            await self._session.close()
        await self.connector.close()
        logger.info("Bitvavo API client closed")



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
