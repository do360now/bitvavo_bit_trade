"""
Bitvavo Trade Executor - handles balance checking and price fetching
"""

import asyncio
from typing import Optional, Dict, List
from logger_config import logger


class TradeExecutor:
    """Executes trades on Bitvavo BTC/EUR market"""

    def __init__(self, bitvavo_api):
        self.bitvavo_api = bitvavo_api
        self.market = "BTC-EUR"
        self.min_order_size = 0.00001  # Bitvavo minimum

    async def get_total_btc_balance(self) -> Optional[float]:
        """Get total BTC balance with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                balance = await self.bitvavo_api.get_balance_async()

                if balance:
                    logger.debug(f"Balance data keys: {list(balance.keys())}")
                    if "BTC" in balance:
                        btc_data = balance["BTC"]
                        logger.debug(f"BTC data: {btc_data}")
                        # Use available + inOrder for total tradable BTC
                        btc_balance = float(btc_data.get("available", 0)) + float(btc_data.get("inOrder", 0))
                        logger.info(f"Total BTC balance: {btc_balance}")
                        return btc_balance
                    else:
                        logger.warning(f"BTC not in balance keys: {list(balance.keys())}")
                        return 0.0

                logger.warning("Balance is None")
                return 0.0

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Balance fetch attempt {attempt + 1} failed: {e}, retrying...")
                    await asyncio.sleep(1)  # Wait before retry
                else:
                    logger.error(f"Failed to fetch BTC balance after {max_retries} attempts: {e}")
                    return 0.0

        return 0.0

    async def get_available_balance(self, asset: str = "EUR") -> Optional[float]:
        """Get available balance for an asset with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                balance = await self.bitvavo_api.get_balance_async()

                if balance and asset in balance:
                    available = float(balance[asset].get("available", 0))
                    logger.info(f"Available {asset} balance: {available}")
                    return available

                logger.warning(f"{asset} asset not found in balance")
                return 0.0

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Balance fetch attempt {attempt + 1} failed: {e}, retrying...")
                    await asyncio.sleep(1)  # Wait before retry
                else:
                    logger.error(f"Failed to fetch {asset} balance after {max_retries} attempts: {e}")
                    return 0.0

        return 0.0

    async def get_current_price(self) -> Optional[float]:
        """Get current BTC/EUR price"""
        try:
            ticker = await self.bitvavo_api.get_ticker_async(self.market)

            if ticker and "price" in ticker:
                price = float(ticker["price"])
                logger.debug(f"Current BTC price: €{price}")
                return price

            logger.error("Failed to get current price")
            return None

        except Exception as e:
            logger.error(f"Failed to fetch current price: {e}")
            return None

    async def get_order_book(self, depth: int = 10) -> Optional[Dict]:
        """Get order book data"""
        try:
            order_book = await self.bitvavo_api.get_order_book_async(self.market, depth)

            if order_book and "bids" in order_book and "asks" in order_book:
                logger.debug(
                    f"Fetched order book with {len(order_book['bids'])} bids and {len(order_book['asks'])} asks"
                )
                return order_book

            logger.error("Failed to get order book")
            return None

        except Exception as e:
            logger.error(f"Failed to fetch order book: {e}")
            return None

    async def get_ohlcv(self, interval: int = 3600) -> Optional[List]:
        """Get OHLCV candle data"""
        try:
            # Bitvavo returns candles in specific format
            candles = await self.bitvavo_api.get_ohlcv_async(self.market, interval)

            if candles:
                logger.debug(f"Fetched {len(candles)} OHLCV candles")
                return candles

            return []

        except Exception as e:
            logger.error(f"Failed to fetch OHLCV data: {e}")
            return None

    def calculate_position_size(
        self, available_eur: float, price: float, risk_percent: float = 12.5
    ) -> float:
        """
        Calculate position size based on available EUR and risk percentage

        Args:
            available_eur: Available EUR balance
            price: Current BTC price
            risk_percent: Risk percentage (default 1% of balance)

        Returns:
            Position size in BTC
        """
        if price <= 0 or available_eur <= 0:
            return 0

        trade_amount_eur = available_eur * (risk_percent / 100)
        position_size = trade_amount_eur / price

        # Ensure minimum order size
        if position_size < self.min_order_size:
            return 0

        return round(position_size, 8)

    async def get_optimal_price(self, order_book: Dict, side: str) -> Optional[float]:
        """
        Get optimal entry price based on order book

        Args:
            order_book: Order book data
            side: 'buy' or 'sell'

        Returns:
            Recommended price
        """
        try:
            if not order_book:
                return None

            if side == "buy":
                # For buying, use best ask (top of ask side)
                asks = order_book.get("asks", [])
                if asks:
                    best_ask = float(asks[0][0])
                    # Place buy order slightly below best ask
                    return round(best_ask * 0.999, 2)

            elif side == "sell":
                # For selling, use best bid (top of bid side)
                bids = order_book.get("bids", [])
                if bids:
                    best_bid = float(bids[0][0])
                    # Place sell order slightly above best bid
                    return round(best_bid * 1.001, 2)

            return None

        except Exception as e:
            logger.error(f"Failed to calculate optimal price: {e}")
            return None
