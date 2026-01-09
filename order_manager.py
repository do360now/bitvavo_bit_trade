"""
Bitvavo Order Manager - handles order placement, tracking, and cancellation
"""

import time
from typing import Dict, List, Optional
from logger_config import logger


class OrderManager:
    """Manages BTC/EUR orders on Bitvavo"""

    def __init__(self, bitvavo_api):
        self.bitvavo_api = bitvavo_api
        self.pending_orders = {}  # order_id -> order_info
        self.filled_orders = {}
        self.cancelled_orders = {}
        self.order_history_file = "./order_history.json"
        self._load_order_history()

    def _load_order_history(self):
        """Load order history from file"""
        try:
            import json
            import os

            if os.path.exists(self.order_history_file):
                with open(self.order_history_file, "r") as f:
                    data = json.load(f)
                    self.filled_orders = data.get("filled", {})
                    self.cancelled_orders = data.get("cancelled", {})
                    logger.info(
                        f"Loaded {len(self.filled_orders)} filled orders and {len(self.cancelled_orders)} cancelled orders"
                    )
        except Exception as e:
            logger.error(f"Failed to load order history: {e}")

    def _save_order_history(self):
        """Save order history to file"""
        try:
            import json

            data = {"filled": self.filled_orders, "cancelled": self.cancelled_orders}
            with open(self.order_history_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save order history: {e}")

    async def place_limit_order(
        self, volume: float, side: str, price: float, market: str = "BTC-EUR"
    ) -> Optional[str]:
        """
        Place a limit order on Bitvavo

        Args:
            volume: Amount of BTC to trade
            side: 'buy' or 'sell'
            price: Limit price in EUR
            market: Market pair (default: BTC-EUR)

        Returns:
            Order ID if successful, None otherwise
        """
        try:
            logger.info(f"Placing {side} order: {volume} BTC at €{price}")

            order_data = {
                "market": market,
                "side": side,
                "orderType": "limit",
                "amount": str(volume),
                "price": str(price),
                "postOnly": True,
            }

            # Place order via Bitvavo API
            response = await self.bitvavo_api.place_order_async(order_data)

            if response and "orderId" in response:
                order_id = response["orderId"]

                self.pending_orders[order_id] = {
                    "timestamp": time.time(),
                    "side": side,
                    "volume": volume,
                    "price": price,
                    "market": market,
                    "status": "pending",
                }

                logger.info(f"Order placed successfully: {order_id}")
                return order_id
            else:
                logger.error(f"Failed to place order: {response}")
                return None

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None

    async def check_order_status(self, order_id: str) -> Optional[Dict]:
        """Check status of a specific order"""
        try:
            # Get order details from Bitvavo
            response = await self.bitvavo_api.get_order_async(order_id)

            if response:
                status = response.get("status", "unknown")

                if status == "closed":
                    # Order filled
                    filled_volume = float(response.get("filledAmount", 0))
                    if filled_volume > 0:
                        self.filled_orders[order_id] = response
                        if order_id in self.pending_orders:
                            del self.pending_orders[order_id]
                        logger.info(f"Order {order_id} FILLED: {filled_volume} BTC")
                        return {"status": "filled", "data": response}

                elif status == "canceled":
                    self.cancelled_orders[order_id] = response
                    if order_id in self.pending_orders:
                        del self.pending_orders[order_id]
                    logger.info(f"Order {order_id} CANCELLED")
                    return {"status": "cancelled", "data": response}

                else:
                    return {"status": status, "data": response}

            return None

        except Exception as e:
            logger.error(f"Failed to check order status: {e}")
            return None

    async def cancel_order(self, order_id: str, market: str = "BTC-EUR") -> bool:
        """Cancel an open order"""
        try:
            response = await self.bitvavo_api.cancel_order_async(market, order_id)

            if response:
                logger.info(f"Order {order_id} cancelled successfully")
                self.cancelled_orders[order_id] = response
                if order_id in self.pending_orders:
                    del self.pending_orders[order_id]
                self._save_order_history()
                return True
            else:
                logger.error(f"Failed to cancel order {order_id}")
                return False

        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    async def check_and_update_orders(self) -> Dict[str, List[str]]:
        """Check all pending orders and update status"""
        results = {"filled": [], "cancelled": [], "pending": []}

        try:
            orders_to_check = list(self.pending_orders.keys())

            for order_id in orders_to_check:
                status_info = await self.check_order_status(order_id)

                if status_info:
                    status = status_info["status"]

                    if status == "filled":
                        results["filled"].append(order_id)
                    elif status == "cancelled":
                        results["cancelled"].append(order_id)
                    else:
                        results["pending"].append(order_id)

            self._save_order_history()
            return results

        except Exception as e:
            logger.error(f"Error checking orders: {e}")
            return results

    def get_order_statistics(self) -> Dict:
        """Calculate order statistics"""
        total_filled = len(self.filled_orders)
        total_cancelled = len(self.cancelled_orders)

        if total_filled == 0:
            fill_rate = 0.0
        else:
            fill_rate = (
                total_filled / (total_filled + total_cancelled) * 100
                if total_cancelled > 0
                else 100.0
            )

        return {
            "fill_rate": fill_rate,
            "filled_orders": total_filled,
            "cancelled_orders": total_cancelled,
            "pending_orders": len(self.pending_orders),
        }
