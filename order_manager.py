"""
Order Manager - Enhanced with Validation
"""

import time
import json
import os
from typing import Dict, List, Optional
from logger_config import logger
from validators import TradingValidator, ValidationError  # NEW IMPORT


class OrderManager:
    """Manages BTC/EUR orders on Bitvavo (sync) with validation"""

    def __init__(self, bitvavo_api, trade_executor=None):
        self.bitvavo_api = bitvavo_api
        self.trade_executor = trade_executor  # NEW: Need this for tick rounding
        self.pending_orders = {}
        self.filled_orders = {}
        self.cancelled_orders = {}
        self.order_history_file = "./order_history.json"
        self._load_order_history()

    def place_limit_order(
        self, volume: float, side: str, price: float, market: str = "BTC-EUR"
    ) -> Optional[str]:
        """
        Place a limit order with validation
        
        Args:
            volume: Amount of BTC to trade
            side: 'buy' or 'sell'
            price: Limit price in EUR
            market: Trading pair
        
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            # === VALIDATION ===
            try:
                volume, price, side = TradingValidator.validate_order(
                    volume, price, side, asset="BTC"
                )
            except ValidationError as e:
                logger.error(f"❌ Order validation failed: {e}")
                return None
            
            # === TICK SIZE ROUNDING ===
            if self.trade_executor:
                price = self.trade_executor.round_to_tick(price)
            else:
                # Fallback to integer rounding if no executor
                price = round(price)
            
            # === PRECISION ROUNDING ===
            volume = round(volume, 8)
            
            # === DOUBLE CHECK (paranoid mode) ===
            order_value = volume * price
            if order_value < 5.0:
                logger.error(f"❌ Order value €{order_value:.2f} below minimum €5.00")
                return None

            logger.info(f"📝 Placing {side.upper()} order: {volume:.8f} BTC @ €{price:,.2f}")

            # === PREPARE ORDER DATA ===
            order_data = {
                "market": market,
                "side": side,
                "orderType": "limit",
                "amount": f"{volume:.8f}",
                "price": str(int(price)),  # Bitvavo wants price as string integer
                "operatorId": "",
            }

            # === PLACE ORDER ===
            response = self.bitvavo_api.place_order_async(order_data)

            if response and "orderId" in response:
                order_id = response["orderId"]

                # Track the order
                self.pending_orders[order_id] = {
                    "timestamp": time.time(),
                    "side": side,
                    "volume": volume,
                    "price": price,
                    "market": market,
                    "status": "pending",
                }

                logger.info(f"✅ Order placed successfully: {order_id}")
                return order_id
            else:
                logger.error(f"❌ Order placement failed - API response: {response}")
                return None

        except Exception as e:
            logger.error(f"❌ Exception placing order: {e}", exc_info=True)
            return None
    

    def _load_order_history(self):
        """Load order history from file"""
        try:
            if os.path.exists(self.order_history_file):
                with open(self.order_history_file, "r") as f:
                    data = json.load(f)
                    self.filled_orders = data.get("filled", {})
                    self.cancelled_orders = data.get("cancelled", {})
                    logger.info(
                        f"Loaded {len(self.filled_orders)} filled and {len(self.cancelled_orders)} cancelled orders"
                    )
        except Exception as e:
            logger.error(f"Failed to load order history: {e}")

    def _save_order_history(self):
        """Save order history to file"""
        try:
            data = {"filled": self.filled_orders, "cancelled": self.cancelled_orders}
            with open(self.order_history_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save order history: {e}")

    # def place_limit_order(
    #     self, volume: float, side: str, price: float, market: str = "BTC-EUR"
    # ) -> Optional[str]:
    #     """Place a limit order (sync)"""
    #     try:
    #         # Round to appropriate precision
    #         volume = round(volume, 8)
    #         price = round(float(price))
            
    #         # Validate
    #         if volume < 0.0001:
    #             logger.error(f"Volume {volume:.8f} BTC below minimum")
    #             return None
            
    #         if price <= 0:
    #             logger.error(f"Invalid price: €{price}")
    #             return None
            
    #         order_value = volume * price
    #         if order_value < 5.0:
    #             logger.error(f"Order value €{order_value:.2f} below minimum €5.00")
    #             return None

    #         logger.info(f"Placing {side.upper()} order: {volume:.8f} BTC @ €{price:.2f}")

    #         # Prepare order data
    #         order_data = {
    #             "market": market,
    #             "side": side,
    #             "orderType": "limit",
    #             "amount": f"{volume:.8f}",
    #             "price": str(int(price)),
    #             "operatorId": "",
    #         }

    #         # Place order (sync call despite _async name)
    #         response = self.bitvavo_api.place_order_async(order_data)

    #         if response and "orderId" in response:
    #             order_id = response["orderId"]

    #             # Track the order
    #             self.pending_orders[order_id] = {
    #                 "timestamp": time.time(),
    #                 "side": side,
    #                 "volume": volume,
    #                 "price": price,
    #                 "market": market,
    #                 "status": "pending",
    #             }

    #             logger.info(f"✅ Order placed successfully: {order_id}")
    #             return order_id
    #         else:
    #             logger.error(f"❌ Failed to place order - API response: {response}")
    #             return None

    #     except Exception as e:
    #         logger.error(f"❌ Exception placing order: {e}", exc_info=True)
    #         return None

    def check_order_status(self, order_id: str) -> Optional[Dict]:
        """Check status of a specific order (sync)"""
        try:
            response = self.bitvavo_api.get_order_async(order_id)

            if response:
                status = response.get("status", "unknown")

                if status == "filled":
                    filled_volume = float(response.get("filledAmount", 0))
                    if filled_volume > 0:
                        self.filled_orders[order_id] = response
                        if order_id in self.pending_orders:
                            del self.pending_orders[order_id]
                        logger.info(f"✅ Order {order_id[:8]}... FILLED: {filled_volume:.8f} BTC")
                        return {"status": "filled", "data": response}

                elif status == "canceled":
                    self.cancelled_orders[order_id] = response
                    if order_id in self.pending_orders:
                        del self.pending_orders[order_id]
                    logger.info(f"⚠️  Order {order_id[:8]}... CANCELLED")
                    return {"status": "cancelled", "data": response}

                else:
                    return {"status": status, "data": response}

            return None

        except Exception as e:
            logger.error(f"Failed to check order status for {order_id}: {e}")
            return None

    def cancel_order(self, order_id: str, market: str = "BTC-EUR") -> bool:
        """Cancel an open order (sync)"""
        try:
            response = self.bitvavo_api.cancel_order_async(market, order_id)

            if response:
                logger.info(f"✅ Order {order_id[:8]}... cancelled")
                self.cancelled_orders[order_id] = response
                if order_id in self.pending_orders:
                    del self.pending_orders[order_id]
                self._save_order_history()
                return True
            else:
                logger.error(f"❌ Failed to cancel order {order_id[:8]}...")
                return False

        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    def check_and_update_orders(self) -> Dict[str, List[str]]:
        """Check all pending orders and update status (sync)"""
        results = {"filled": [], "cancelled": [], "pending": []}

        try:
            orders_to_check = list(self.pending_orders.keys())

            for order_id in orders_to_check:
                status_info = self.check_order_status(order_id)

                if status_info:
                    status = status_info["status"]

                    if status == "filled":
                        results["filled"].append(order_id)
                    elif status == "cancelled":
                        results["cancelled"].append(order_id)
                    else:
                        results["pending"].append(order_id)

            if results["filled"] or results["cancelled"]:
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