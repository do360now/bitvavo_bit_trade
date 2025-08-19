import time
from datetime import datetime
from typing import Optional, Dict, List
from utils.logger import logger


class TradeExecutor:
    def __init__(self, bitvavo_api):
        self.bitvavo_api = bitvavo_api
        self.pair = "BTC/EUR"  # Bitvavo pair format

    def fetch_current_price(self) -> tuple[Optional[float], float]:
        """Fetch current BTC/EUR price and volume from Bitvavo"""
        try:
            # Get OHLCV data (1m timeframe for recent data)
            ohlcv = self.bitvavo_api.fetch_ohlcv(self.pair, "15m", limit=10)
            logger.debug(f"Raw OHLCV response: {ohlcv}")

            if ohlcv and len(ohlcv) > 0:
                # OHLCV format: [timestamp, open, high, low, close, volume]
                latest_candle = ohlcv[-1]
                close_price = float(latest_candle[4])  # Close price
                volume = float(latest_candle[5])  # Volume

                logger.debug(f"Fetched price: €{close_price:.2f}, volume: {volume:.2f}")
                return close_price, volume
            else:
                logger.warning("No OHLCV data available")
                return None, 0.0

        except Exception as e:
            logger.error(f"Failed to fetch current price: {e}")
            return None, 0.0

    def get_btc_order_book(self) -> Optional[Dict]:
        """Get BTC/EUR order book from Bitvavo"""
        try:
            order_book = self.bitvavo_api.fetch_order_book(self.pair)
            logger.debug(
                f"Order book fetched: bids={len(order_book.get('bids', []))}, asks={len(order_book.get('asks', []))}"
            )
            return order_book
        except Exception as e:
            logger.error(f"Failed to fetch order book: {e}")
            return None

    def get_optimal_price(self, order_book: Dict, side: str) -> Optional[float]:
        """Calculate optimal price from order book"""
        try:
            if side == "buy":
                # For buying, use the best ask price (slightly lower to get filled)
                asks = order_book.get("asks", [])
                if asks:
                    best_ask = float(asks[0][0])
                    # Place buy order slightly below best ask to ensure fill
                    optimal_price = best_ask * 0.9999  # 0.0% below best ask
                    logger.debug(
                        f"Optimal buy price: €{optimal_price:.2f} (best ask: €{best_ask:.2f})"
                    )
                    return optimal_price
            else:  # sell
                # For selling, use the best bid price (slightly higher to get filled)
                bids = order_book.get("bids", [])
                if bids:
                    best_bid = float(bids[0][0])
                    # Place sell order slightly above best bid to ensure fill
                    optimal_price = best_bid * 1.0001  # 0.01% above best bid
                    logger.debug(
                        f"Optimal sell price: €{optimal_price:.2f} (best bid: €{best_bid:.2f})"
                    )
                    return optimal_price

            logger.warning(f"No {side} price available in order book")
            return None

        except Exception as e:
            logger.error(f"Failed to calculate optimal price: {e}")
            return None

    def execute_trade(self, volume: float, side: str, price: float) -> bool:
        """Execute a limit order on Bitvavo"""
        try:
            if side not in ["buy", "sell"]:
                logger.error(f"Invalid trade side: {side}")
                return False

            # Check minimum order requirements
            try:
                markets = self.bitvavo_api.load_markets()
                market_info = markets.get(self.pair)

                if market_info:
                    limits = market_info.get("limits", {})
                    min_amount = limits.get("amount", {}).get("min", 0)
                    min_cost = limits.get("cost", {}).get("min", 0)

                    order_value = volume * price

                    if min_amount and volume < min_amount:
                        logger.error(
                            f"Order amount {volume:.8f} BTC below minimum {min_amount:.8f} BTC"
                        )
                        return False

                    if min_cost and order_value < min_cost:
                        logger.error(
                            f"Order value €{order_value:.2f} below minimum €{min_cost:.2f}"
                        )
                        return False

                    logger.debug(
                        f"Order meets requirements: {volume:.8f} BTC (min: {min_amount:.8f}), €{order_value:.2f} (min: €{min_cost:.2f})"
                    )
            except Exception as e:
                logger.warning(f"Could not check order limits: {e}")

            logger.debug(f"Executing {side} order: {volume:.8f} BTC at €{price:.2f}")

            # Create limit order with error handling
            try:
                import time

                operator_id = int(time.time() * 1000)  # Generate unique operatorId

                order = self.bitvavo_api.create_limit_order(
                    symbol=self.pair,
                    side=side,
                    amount=volume,
                    price=price,
                    params={"operatorId": operator_id},
                )
            except Exception as e:
                logger.error(f"create_limit_order failed: {e}")
                # Try alternative method
                try:
                    import time

                    operator_id = int(time.time() * 1000) + 1  # Different ID for retry

                    order = self.bitvavo_api.create_order(
                        symbol=self.pair,
                        type="limit",
                        side=side,
                        amount=volume,
                        price=price,
                        params={"operatorId": operator_id},
                    )
                except Exception as e2:
                    logger.error(f"create_order also failed: {e2}")
                    return False

            if order and order.get("id"):
                logger.info(
                    f"✅ {side.upper()} order placed: {volume:.8f} BTC at €{price:.2f}, Order ID: {order['id']}"
                )
                return True
            else:
                logger.error(f"Failed to place {side} order: {order}")
                return False

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return False

    def get_total_btc_balance(self) -> Optional[float]:
        """Get total BTC balance (available + in orders)"""
        try:
            balance = self.bitvavo_api.fetch_balance()
            btc_balance = balance.get("BTC", {})

            # Get total (free + used)
            total = btc_balance.get("total", 0)
            free = btc_balance.get("free", 0)
            used = btc_balance.get("used", 0)

            logger.debug(
                f"BTC balance - Total: {total:.8f}, Free: {free:.8f}, Used: {used:.8f}"
            )
            return float(total) if total else 0.0

        except Exception as e:
            logger.error(f"Failed to fetch BTC balance: {e}")
            return None

    def get_available_balance(self, currency: str) -> Optional[float]:
        """Get available balance for a specific currency"""
        try:
            balance = self.bitvavo_api.fetch_balance()
            currency_balance = balance.get(currency, {})

            # Get free balance (available for trading)
            free = currency_balance.get("free", 0)

            logger.debug(f"{currency} available balance: {free:.2f}")
            return float(free) if free else 0.0

        except Exception as e:
            logger.error(f"Failed to fetch {currency} balance: {e}")
            return None

    def get_ohlc_data(
        self,
        pair: str = None,
        interval: str = "15m",
        since: int = None,
        limit: int = 100,
    ):
        """Get OHLC data for backtesting and analysis with improved timestamp handling"""
        try:
            if pair is None:
                pair = self.pair

            # Convert Kraken interval format to Bitvavo format if needed
            interval_map = {
                1: "1m",
                5: "5m",
                15: "15m",
                30: "30m",
                60: "1h",
                240: "4h",
                1440: "1d",
            }

            if isinstance(interval, int):
                interval = interval_map.get(interval, "15m")

            # For first run, get more historical data
            if since is None:
                # Get last 7 days of data for initial setup
                since = int(time.time() - (7 * 24 * 3600))

            logger.info(
                f"Fetching OHLC data since {datetime.fromtimestamp(since)} for pair {pair}"
            )

            # Fetch OHLCV data
            ohlcv = self.bitvavo_api.fetch_ohlcv(
                symbol=pair,
                timeframe=interval,
                since=since * 1000 if since else None,  # Convert to milliseconds
                limit=limit,
            )

            if ohlcv:
                # Convert to format expected by existing code
                # [timestamp, open, high, low, close, volume]
                formatted_data = []
                current_time = int(time.time())

                logger.info(f"Processing {len(ohlcv)} raw OHLC candles")

                for i, candle in enumerate(ohlcv):
                    try:
                        # Bitvavo returns timestamps in milliseconds
                        timestamp_ms = int(candle[0])
                        timestamp = timestamp_ms // 1000  # Convert to seconds

                        # Debug first few timestamps
                        if i < 5:
                            logger.info(
                                f"Candle {i}: Raw timestamp {timestamp_ms} -> {timestamp} ({datetime.fromtimestamp(timestamp)})"
                            )

                        # Validate timestamp is reasonable (not too far in past or future)
                        min_valid = current_time - (365 * 24 * 3600)  # 1 year ago
                        max_valid = current_time + (24 * 3600)  # 1 day in future

                        if timestamp < min_valid:
                            logger.debug(
                                f"Skipping old timestamp: {timestamp} ({datetime.fromtimestamp(timestamp)})"
                            )
                            continue

                        if timestamp > max_valid:
                            logger.debug(
                                f"Skipping future timestamp: {timestamp} ({datetime.fromtimestamp(timestamp)})"
                            )
                            continue

                        # Validate OHLC data
                        open_price = float(candle[1])
                        high_price = float(candle[2])
                        low_price = float(candle[3])
                        close_price = float(candle[4])
                        volume = float(candle[5])

                        # Basic validation
                        if not (
                            0 < low_price <= high_price
                            and low_price <= open_price <= high_price
                            and low_price <= close_price <= high_price
                            and volume >= 0
                        ):
                            logger.debug(f"Skipping invalid OHLC values: {candle}")
                            continue

                        # Price sanity check for Bitcoin
                        if not (1000 < close_price < 1000000):
                            logger.debug(f"Skipping unrealistic price: {close_price}")
                            continue

                        formatted_data.append(
                            [
                                timestamp,  # timestamp in seconds
                                open_price,  # open
                                high_price,  # high
                                low_price,  # low
                                close_price,  # close
                                volume,  # volume
                            ]
                        )

                    except (ValueError, TypeError, IndexError) as e:
                        logger.debug(f"Skipping invalid candle {i}: {candle} - {e}")
                        continue

                if formatted_data:
                    logger.info(
                        f"Successfully formatted {len(formatted_data)} OHLC candles"
                    )
                    logger.info(
                        f"Time range: {datetime.fromtimestamp(formatted_data[0][0])} to {datetime.fromtimestamp(formatted_data[-1][0])}"
                    )
                    logger.info(
                        f"Price range: €{min(c[4] for c in formatted_data):.2f} to €{max(c[4] for c in formatted_data):.2f}"
                    )
                else:
                    logger.warning("No valid OHLC candles after processing")

                return formatted_data

            logger.warning("No OHLC data received from API")
            return []

        except Exception as e:
            logger.error(f"Failed to fetch OHLC data: {e}")
            return []

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order"""
        try:
            import time

            operator_id = int(time.time() * 1000)  # Generate unique operatorId
            result = self.bitvavo_api.cancel_order(
                order_id, self.pair, params={"operatorId": operator_id}
            )
            logger.info(
                f"Order {order_id} cancelled successfully with operatorId {operator_id}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get status of a specific order"""
        try:
            order = self.bitvavo_api.fetch_order(order_id, self.pair)
            return order
        except Exception as e:
            logger.error(f"Failed to fetch order status for {order_id}: {e}")
            return None

    def get_open_orders(self) -> List[Dict]:
        """Get all open orders"""
        try:
            orders = self.bitvavo_api.fetch_open_orders(self.pair)
            logger.debug(f"Fetched {len(orders)} open orders")
            return orders
        except Exception as e:
            logger.error(f"Failed to fetch open orders: {e}")
            return []
