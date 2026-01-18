"""
Bitvavo Trade Executor - FIXED with proper tick size handling
"""

import asyncio
import time
from typing import Optional, Dict, List
from logger_config import logger


class TradeExecutor:
    """Executes trades on Bitvavo BTC/EUR market with proper precision"""

    def __init__(self, bitvavo_api):
        self.bitvavo_api = bitvavo_api
        self.market = "BTC-EUR"
        self.min_order_size = 0.0001  # Bitvavo minimum BTC
        
        # Market specs (fetched once at startup)
        self.tick_size = 1.0
        self.min_order_value = 5.0  # Bitvavo minimum is €5
        self.amount_decimals = 8  # BTC precision
        
    def initialize(self):
        """Fetch market specifications on startup"""
        try:
            market_info = self.bitvavo_api._make_request(
                "GET", "markets", {"market": self.market}
            )
            
            if market_info:
                info = market_info[0] if isinstance(market_info, list) else market_info
                
                self.tick_size = float(info.get('tickSize', 1.0))
                self.min_order_value = float(info.get('minOrderInQuoteAsset', 5.0))
                self.amount_decimals = int(info.get('baseDecimals', 8))
                
                logger.info(f"Market specs loaded: tick_size=€{self.tick_size}, min_order=€{self.min_order_value}")
            else:
                logger.warning(f"Could not fetch market info, using defaults")
                self.tick_size = 1.0  # Default to €1 increments for BTC
                
        except Exception as e:
            logger.error(f"Failed to fetch market specs: {e}")
            self.tick_size = 1.0  # Safe default

    def round_to_tick(self, price: float) -> float:
        """Round price to nearest valid tick size"""
        if not self.tick_size:
            return round(price, 2)  # Fallback
        
        rounded = round(price / self.tick_size) * self.tick_size
        
        # Format based on tick size
        if self.tick_size >= 1:
            # Whole numbers (e.g., €77800)
            return round(rounded)
        elif self.tick_size >= 0.1:
            # One decimal (e.g., €77800.0)
            return round(rounded, 1)
        else:
            # Two decimals (e.g., €77800.12)
            return round(rounded, 2)

    def get_total_btc_balance(self) -> Optional[float]:
        """Get total BTC balance with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                balance = self.bitvavo_api.get_balance_async()

                if balance:
                    if "BTC" in balance:
                        btc_data = balance["BTC"]
                        # Use available + inOrder for total tradable BTC
                        btc_balance = float(btc_data.get("available", 0)) + float(btc_data.get("inOrder", 0))
                        logger.debug(f"Total BTC balance: {btc_balance:.8f}")
                        return btc_balance
                    else:
                        logger.warning(f"BTC not in balance keys: {list(balance.keys())}")
                        return 0.0

                logger.warning("Balance is None")
                return 0.0

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Balance fetch attempt {attempt + 1} failed: {e}, retrying...")
                    time.sleep(1)
                else:
                    logger.error(f"Failed to fetch BTC balance after {max_retries} attempts: {e}")
                    return 0.0

        return 0.0

    def get_available_balance(self, asset: str = "EUR") -> Optional[float]:
        """Get available balance for an asset with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                balance = self.bitvavo_api.get_balance_async()

                if balance and asset in balance:
                    available = float(balance[asset].get("available", 0))
                    logger.debug(f"Available {asset} balance: {available:.2f}")
                    return available

                logger.warning(f"{asset} asset not found in balance")
                return 0.0

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Balance fetch attempt {attempt + 1} failed: {e}, retrying...")
                    time.sleep(1)
                else:
                    logger.error(f"Failed to fetch {asset} balance after {max_retries} attempts: {e}")
                    return 0.0

        return 0.0

    def get_current_price(self) -> Optional[float]:
        """Get current BTC/EUR price"""
        try:
            ticker = self.bitvavo_api.get_ticker_async(self.market)

            if ticker and "price" in ticker:
                price = float(ticker["price"])
                logger.debug(f"Current BTC price: €{price:,.2f}")
                return price

            logger.error("Failed to get current price")
            return None

        except Exception as e:
            logger.error(f"Failed to fetch current price: {e}")
            return None

    def get_order_book(self, depth: int = 10) -> Optional[Dict]:
        """Get order book data"""
        try:
            order_book = self.bitvavo_api.get_order_book_async(self.market, depth)

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

    def get_ohlcv(self, interval: int = 3600) -> Optional[List]:
        """Get OHLCV candle data"""
        try:
            candles = self.bitvavo_api.get_ohlcv_async(self.market, interval)

            if candles:
                logger.debug(f"Fetched {len(candles)} OHLCV candles")
                return candles

            return []

        except Exception as e:
            logger.error(f"Failed to fetch OHLCV data: {e}")
            return None

    def calculate_position_size(
        self, available_eur: float, price: float, risk_percent: float = 100.0
    ) -> float:
        """
        Calculate position size based on available EUR and risk percentage

        Args:
            available_eur: Available EUR balance
            price: Current BTC price in EUR
            risk_percent: Risk percentage of balance
                - Can be decimal (0.01 = 1%) or percentage (1.0 = 1%, 100.0 = 100%)
                - Auto-detects format based on value

        Returns:
            Position size in BTC (0 if below minimum order size)
        """
        if price <= 0 or available_eur <= 0:
            logger.debug(f"Cannot calculate position: price={price}, eur={available_eur}")
            return 0

        # Auto-detect risk_percent format
        # If risk_percent < 1, assume it's in decimal format (0.01 = 1%)
        # If risk_percent >= 1, assume it's in percentage format (1.0 = 1%, 100.0 = 100%)
        if risk_percent < 1:
            # Decimal format: multiply by 100 to convert to percentage
            risk_percentage = risk_percent * 100
        else:
            # Percentage format: use as-is
            risk_percentage = risk_percent

        # Use a percentage of available balance
        trade_amount_eur = available_eur * (risk_percentage / 100)
        
        # Check minimum order value
        if trade_amount_eur < self.min_order_value:
            logger.warning(
                f"Trade amount €{trade_amount_eur:.2f} below minimum €{self.min_order_value}"
            )
            return 0
        
        position_size = trade_amount_eur / price
        
        logger.debug(
            f"Position calc: €{available_eur:.2f} × {risk_percentage}% = €{trade_amount_eur:.2f} "
            f"→ {position_size:.8f} BTC (min: {self.min_order_size})"
        )

        # Ensure minimum order size
        if position_size < self.min_order_size:
            logger.warning(
                f"Position size {position_size:.8f} BTC below minimum {self.min_order_size} BTC "
                f"(need €{self.min_order_size * price:.2f} EUR minimum)"
            )
            return 0

        return round(position_size, self.amount_decimals)

    def get_optimal_price(self, order_book: Dict, side: str) -> Optional[float]:
        """
        Get optimal entry price based on order book, rounded to valid tick size

        Args:
            order_book: Order book data
            side: 'buy' or 'sell'

        Returns:
            Recommended price (rounded to tick size)
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
                    target_price = best_ask * 0.999
                    # Round to valid tick
                    return self.round_to_tick(target_price)

            elif side == "sell":
                # For selling, use best bid (top of bid side)
                bids = order_book.get("bids", [])
                if bids:
                    best_bid = float(bids[0][0])
                    # Place sell order slightly above best bid
                    target_price = best_bid * 1.001
                    # Round to valid tick
                    return self.round_to_tick(target_price)

            return None

        except Exception as e:
            logger.error(f"Failed to calculate optimal price: {e}")
            return None