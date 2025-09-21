# bitvavo_api_async.py
import asyncio
import aiohttp
import time
import hmac
import hashlib
import json
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager
import logging
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

logger = logging.getLogger(__name__)

@dataclass
class RateLimitInfo:
    """Rate limiting information."""
    requests_per_second: float = 10.0
    burst_size: int = 20
    last_request_time: float = 0.0
    request_count: int = 0
    window_start: float = 0.0

class AsyncBitvavoAPI:
    """Async Bitvavo API client with enhanced error handling and rate limiting."""
    
    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://api.bitvavo.com/v2"):
        self.api_key = api_key
        self.api_secret = api_secret.encode('utf-8')
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit = RateLimitInfo()
        self._lock = asyncio.Lock()
    
    async def __aenter__(self):
        await self.start_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_session()
    
    async def start_session(self):
        """Initialize HTTP session with optimized settings."""
        connector = aiohttp.TCPConnector(
            limit=10,
            limit_per_host=5,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'BitcoinTradingBot/1.0',
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }
        )
    
    async def close_session(self):
        """Clean up HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    def _generate_signature(self, timestamp: str, method: str, url_path: str, body: str = "") -> str:
        """Generate API signature."""
        message = timestamp + method + url_path + body
        signature = hmac.new(
            self.api_secret,
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    async def _rate_limit_check(self):
        """Enforce rate limiting with token bucket algorithm."""
        async with self._lock:
            current_time = time.time()
            
            # Reset window if needed
            if current_time - self.rate_limit.window_start >= 1.0:
                self.rate_limit.window_start = current_time
                self.rate_limit.request_count = 0
            
            # Check if we can make request
            if self.rate_limit.request_count >= self.rate_limit.burst_size:
                sleep_time = 1.0 - (current_time - self.rate_limit.window_start)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    self.rate_limit.window_start = time.time()
                    self.rate_limit.request_count = 0
            
            self.rate_limit.request_count += 1
            self.rate_limit.last_request_time = current_time
    
    @retry(
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
)
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        signed: bool = False
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic."""
        await self._rate_limit_check()
        
        url = f"{self.base_url}/{endpoint}"
        headers = {}
        body = ""
        
        if signed:
            timestamp = str(int(time.time() * 1000))
            if data:
                body = json.dumps(data, separators=(',', ':'))
            
            # Fix: Include the API version in the signature path to match the actual URL
            signature_path = f"/v2/{endpoint}"
            signature = self._generate_signature(timestamp, method, signature_path, body)
            
            headers.update({
                'Bitvavo-Access-Key': self.api_key,
                'Bitvavo-Access-Signature': signature,
                'Bitvavo-Access-Timestamp': timestamp,
            })
        
        try:
            async with self.session.request(
                method=method,
                url=url,
                params=params,
                data=body if body else None,
                headers=headers
            ) as response:
                response.raise_for_status()
                return await response.json()
                
        except aiohttp.ClientResponseError as e:
            if e.status == 429:
                # Rate limited, wait longer
                await asyncio.sleep(5)
                raise
            elif e.status >= 500:
                # Server error, retry
                logger.warning(f"Server error {e.status}, retrying...")
                raise
            else:
                # Client error, don't retry
                logger.error(f"Client error {e.status}: {e.message}")
                raise ValueError(f"API error: {e.status} - {e.message}")
    
    async def get_ticker(self, symbol: str = "BTC-EUR") -> Dict[str, Any]:
        """Get current ticker information."""
        return await self._make_request("GET", f"ticker/price", {"market": symbol})
    
            
    async def get_order_book(self, symbol: str = "BTC-EUR", depth: int = 25) -> Dict[str, Any]:
        """Get order book with specified depth."""
        params = {"depth": depth}
        # Use market-specific endpoint: {symbol}/book
        return await self._make_request("GET", f"{symbol}/book", params)

    async def get_ohlcv(
        self, 
        symbol: str = "BTC-EUR", 
        interval: str = "15m", 
        limit: int = 100,
        since: Optional[int] = None
    ) -> List[List]:
        """Get OHLCV data with improved error handling."""
        params = {
            "interval": interval,
            "limit": min(limit, 1440)  # API limit
        }
        
        if since:
            params["start"] = since
        
        # Use market-specific endpoint: {symbol}/candles
        result = await self._make_request("GET", f"{symbol}/candles", params)
        
        # Validate and process response
        if not isinstance(result, list):
            raise ValueError(f"Unexpected OHLCV response format: {type(result)}")
        
        # Format data consistently
        formatted_data = []
        for candle in result:
            try:
                formatted_data.append([
                    int(candle[0]) // 1000,  # timestamp in seconds
                    float(candle[1]),        # open
                    float(candle[2]),        # high
                    float(candle[3]),        # low
                    float(candle[4]),        # close
                    float(candle[5])         # volume
                ])
            except (ValueError, IndexError) as e:
                logger.warning(f"Skipping invalid candle data: {candle} - {e}")
                continue
        
        return formatted_data

    # Note: get_ticker() is already correct and doesn't need changes

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all open orders."""
        params = {}
        if symbol:  # Only add market param if symbol is provided and not None
            params["market"] = symbol
        return await self._make_request("GET", "orders", params, signed=True)
    
    async def get_balance(self) -> Dict[str, Any]:
        """Get account balance."""
        return await self._make_request("GET", "balance", signed=True)
    
    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
        time_in_force: str = "GTC"
    ) -> Dict[str, Any]:
        """Create a new order with enhanced validation."""
        # Validate inputs
        if side not in ["buy", "sell"]:
            raise ValueError(f"Invalid side: {side}")
        
        if order_type not in ["market", "limit"]:
            raise ValueError(f"Invalid order type: {order_type}")
        
        if amount <= 0:
            raise ValueError(f"Invalid amount: {amount}")
        
        if order_type == "limit" and (not price or price <= 0):
            raise ValueError(f"Invalid price for limit order: {price}")
        
        data = {
            "market": symbol,
            "side": side,
            "orderType": order_type,
            "amount": str(amount),
            "timeInForce": time_in_force
        }
        
        if price:
            data["price"] = str(price)
        
        return await self._make_request("POST", "order", data=data, signed=True)
    
    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Cancel an existing order."""
        data = {"orderId": order_id, "market": symbol}
        return await self._make_request("DELETE", "order", data=data, signed=True)
    
    async def get_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Get order status."""
        params = {"orderId": order_id, "market": symbol}
        return await self._make_request("GET", "order", params, signed=True)
    
    

# Connection pool manager
class APIConnectionPool:
    """Manage multiple API connections for improved throughput."""
    
    def __init__(self, api_key: str, api_secret: str, pool_size: int = 3):
        self.api_key = api_key
        self.api_secret = api_secret
        self.pool_size = pool_size
        self.connections: List[AsyncBitvavoAPI] = []
        self.connection_queue = asyncio.Queue()
        self._initialized = False
    
    async def initialize(self):
        """Initialize connection pool."""
        if self._initialized:
            return
        
        for _ in range(self.pool_size):
            conn = AsyncBitvavoAPI(self.api_key, self.api_secret)
            await conn.start_session()
            self.connections.append(conn)
            await self.connection_queue.put(conn)
        
        self._initialized = True
        logger.info(f"Initialized API connection pool with {self.pool_size} connections")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool."""
        if not self._initialized:
            await self.initialize()
        
        conn = await self.connection_queue.get()
        try:
            yield conn
        finally:
            await self.connection_queue.put(conn)
    
    async def close_all(self):
        """Close all connections in the pool."""
        for conn in self.connections:
            await conn.close_session()
        self.connections.clear()
        self._initialized = False