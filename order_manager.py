import time
import json
import os
from typing import Dict, List, Optional, Tuple
from logger_config import logger

class OrderManager:
    def __init__(self, bitvavo_api):
        self.bitvavo_api = bitvavo_api
        self.pair = "BTC/EUR"
        self.pending_orders = {}  # order_id -> order_info
        self.filled_orders = {}   # order_id -> order_info
        self.cancelled_orders = {} # order_id -> order_info
        self.order_history_file = "./order_history.json"
        self._load_order_history()
        
        # CRITICAL FIX: Sync with exchange on startup
        self._sync_with_exchange_on_startup()

    def _sync_with_exchange_on_startup(self):
        """
        CRITICAL: Sync local state with exchange on startup to avoid stale pending orders
        """
        logger.info("🔄 Syncing order state with exchange on startup...")
        
        try:
            # Get all currently open orders from exchange
            exchange_open_orders = self.bitvavo_api.fetch_open_orders(self.pair)
            exchange_order_ids = {order['id'] for order in exchange_open_orders}
            
            logger.info(f"Found {len(exchange_open_orders)} open orders on exchange")
            
            # Check our pending orders against exchange reality
            stale_orders = []
            for order_id in list(self.pending_orders.keys()):
                if order_id not in exchange_order_ids:
                    logger.warning(f"⚠️ Order {order_id} in pending but not on exchange - marking as filled/cancelled")
                    stale_orders.append(order_id)
                    
                    # Try to get final status from exchange
                    try:
                        final_status = self.bitvavo_api.fetch_order(order_id, self.pair)
                        if final_status:
                            status = final_status.get('status', '').lower()
                            if status in ['closed', 'filled']:
                                # Move to filled orders
                                order_info = self.pending_orders[order_id].copy()
                                order_info.update({
                                    'status': status,
                                    'executed_volume': final_status.get('filled', 0),
                                    'average_price': final_status.get('average', order_info['price']),
                                    'fee': final_status.get('fee', {}).get('cost', 0),
                                    'filled_at': time.time(),
                                    'raw_order': final_status
                                })
                                self.filled_orders[order_id] = order_info
                                logger.info(f"✅ Recovered filled order: {order_id}")
                            else:
                                # Move to cancelled orders
                                order_info = self.pending_orders[order_id].copy()
                                order_info.update({
                                    'status': status,
                                    'cancelled_at': time.time(),
                                    'raw_order': final_status
                                })
                                self.cancelled_orders[order_id] = order_info
                                logger.info(f"❌ Recovered cancelled order: {order_id}")
                    except Exception as e:
                        logger.warning(f"Could not fetch final status for {order_id}: {e}")
                        # Default to cancelled if we can't determine status
                        order_info = self.pending_orders[order_id].copy()
                        order_info.update({
                            'status': 'cancelled',
                            'cancelled_at': time.time(),
                            'reason': 'Stale order - not found on exchange'
                        })
                        self.cancelled_orders[order_id] = order_info
            
            # Remove stale orders from pending
            for order_id in stale_orders:
                self.pending_orders.pop(order_id, None)
            
            # Update pending orders with current exchange data
            current_pending = {}
            for order in exchange_open_orders:
                order_id = order['id']
                if order_id in self.pending_orders:
                    # Update existing tracked order
                    order_info = self.pending_orders[order_id].copy()
                    order_info.update({
                        'status': order.get('status', 'open'),
                        'executed_volume': order.get('filled', 0),
                        'average_price': order.get('average', order_info['price']),
                        'raw_order': order
                    })
                    current_pending[order_id] = order_info
                else:
                    # Found untracked order - this shouldn't happen normally
                    logger.warning(f"Found untracked open order on exchange: {order_id}")
                    # Add it to our tracking
                    order_info = {
                        'id': order_id,
                        'side': order['side'],
                        'volume': order['amount'],
                        'price': order['price'],
                        'timestamp': order.get('timestamp', time.time()) / 1000,  # Convert from ms
                        'timeout': 300,  # Default timeout
                        'status': order.get('status', 'open'),
                        'executed_volume': order.get('filled', 0),
                        'average_price': order.get('average', order['price']),
                        'fee': order.get('fee', {}).get('cost', 0),
                        'raw_order': order
                    }
                    current_pending[order_id] = order_info
            
            self.pending_orders = current_pending
            
            if stale_orders:
                logger.info(f"🧹 Cleaned up {len(stale_orders)} stale orders")
                self._save_order_history()
            
            logger.info(f"✅ Sync complete: {len(self.pending_orders)} orders currently pending")
            
        except Exception as e:
            logger.error(f"Failed to sync with exchange on startup: {e}")

    def _load_order_history(self):
        """Load order history from file"""
        try:
            if os.path.exists(self.order_history_file):
                with open(self.order_history_file, 'r') as f:
                    data = json.load(f)
                    self.filled_orders = data.get('filled_orders', {})
                    self.cancelled_orders = data.get('cancelled_orders', {})
                    # Don't load pending orders as they might be stale - we'll sync with exchange
                    logger.info(f"Loaded {len(self.filled_orders)} filled orders and {len(self.cancelled_orders)} cancelled orders")
            else:
                logger.info("No existing order history found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load order history: {e}")

    def _save_order_history(self):
        """Save order history to file"""
        try:
            data = {
                'filled_orders': self.filled_orders,
                'cancelled_orders': self.cancelled_orders,
                'last_updated': time.time()
            }
            with open(self.order_history_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved order history: {len(self.filled_orders)} filled, {len(self.cancelled_orders)} cancelled")
        except Exception as e:
            logger.error(f"Failed to save order history: {e}")

    def place_limit_order_with_timeout(self, volume: float, side: str, price: float, 
                                     timeout: int = 300, post_only: bool = False) -> Optional[str]:
        """Place a limit order with timeout handling"""
        try:
            logger.info(f"Placing {side} order: {volume:.8f} BTC at €{price:.2f}")
            
            # Create the order using CCXT with operatorId
            try:
                # Bitvavo requires operatorId - generate a unique one
                import time
                operator_id = int(time.time() * 1000)  # Use timestamp as operatorId
                
                order = self.bitvavo_api.create_limit_order(
                    symbol=self.pair,
                    side=side,
                    amount=volume,
                    price=price,
                    params={
                        'operatorId': operator_id,
                        'postOnly': post_only
                    } if post_only else {'operatorId': operator_id}
                )
            except Exception as order_error:
                logger.error(f"CCXT order creation failed: {order_error}")
                
                # Try alternative method if the first fails
                try:
                    import time
                    operator_id = int(time.time() * 1000) + 1  # Different ID for retry
                    
                    order = self.bitvavo_api.create_order(
                        symbol=self.pair,
                        type='limit',
                        side=side,
                        amount=volume,
                        price=price,
                        params={'operatorId': operator_id}
                    )
                except Exception as alt_error:
                    logger.error(f"Alternative order creation also failed: {alt_error}")
                    raise order_error
            
            if not order or not order.get('id'):
                logger.error(f"Failed to create order: {order}")
                return None
            
            order_id = order['id']
            
            # Store order info for tracking
            order_info = {
                'id': order_id,
                'side': side,
                'volume': volume,
                'price': price,
                'timestamp': time.time(),
                'timeout': timeout,
                'status': order.get('status', 'open'),
                'executed_volume': order.get('filled', 0),
                'average_price': order.get('average', price),
                'fee': order.get('fee', {}).get('cost', 0),
                'raw_order': order
            }
            
            self.pending_orders[order_id] = order_info
            logger.info(f"✅ {side.upper()} order placed successfully: {order_id}")
            return order_id
            
        except Exception as e:
            logger.error(f"Failed to place {side} order: {e}")
            return None

    def check_and_update_orders(self) -> Dict[str, List[str]]:
        """Check status of pending orders and update accordingly"""
        results = {
            'filled': [],
            'cancelled': [],
            'partial': [],
            'still_pending': []
        }
        
        if not self.pending_orders:
            return results
        
        try:
            current_time = time.time()
            orders_to_remove = []
            
            for order_id, order_info in self.pending_orders.items():
                try:
                    # Check if order has timed out
                    if current_time - order_info['timestamp'] > order_info['timeout']:
                        logger.info(f"⏰ Order {order_id} timed out, cancelling...")
                        if self._cancel_order(order_id):
                            results['cancelled'].append(order_id)
                            orders_to_remove.append(order_id)
                        continue
                    
                    # Fetch current order status from Bitvavo
                    try:
                        order_status = self.bitvavo_api.fetch_order(order_id, self.pair)
                    except Exception as e:
                        error_msg = str(e).lower()
                        if "no active order found" in error_msg or "order not found" in error_msg:
                            # Order doesn't exist on exchange - likely filled or cancelled
                            logger.warning(f"Order {order_id} not found on exchange - assuming filled/cancelled")
                            # Try to determine final state or default to cancelled
                            order_info['cancelled_at'] = current_time
                            self.cancelled_orders[order_id] = order_info.copy()
                            results['cancelled'].append(order_id)
                            orders_to_remove.append(order_id)
                            continue
                        else:
                            logger.warning(f"Could not fetch order {order_id} status: {e}")
                            continue
                    
                    if not order_status:
                        logger.warning(f"No status returned for order {order_id}")
                        continue
                    
                    status = order_status.get('status', '').lower()
                    filled_amount = order_status.get('filled', 0)
                    average_price = order_status.get('average', order_info['price'])
                    fee_info = order_status.get('fee', {})
                    
                    # Update order info
                    order_info.update({
                        'status': status,
                        'executed_volume': filled_amount,
                        'average_price': average_price,
                        'fee': fee_info.get('cost', 0),
                        'raw_order': order_status
                    })
                    
                    if status == 'closed' or status == 'filled':
                        # Order fully filled
                        order_info['filled_at'] = current_time
                        self.filled_orders[order_id] = order_info.copy()
                        results['filled'].append(order_id)
                        orders_to_remove.append(order_id)
                        logger.info(f"✅ Order {order_id} FILLED: {filled_amount:.8f} BTC @ €{average_price:.2f}")
                        
                    elif status == 'canceled' or status == 'cancelled':
                        # Order cancelled
                        order_info['cancelled_at'] = current_time
                        self.cancelled_orders[order_id] = order_info.copy()
                        results['cancelled'].append(order_id)
                        orders_to_remove.append(order_id)
                        logger.info(f"❌ Order {order_id} CANCELLED")
                        
                    elif filled_amount > 0 and status == 'open':
                        # Partially filled
                        results['partial'].append(order_id)
                        logger.info(f"📊 Order {order_id} PARTIAL: {filled_amount:.8f}/{order_info['volume']:.8f} BTC")
                        
                    else:
                        # Still pending
                        results['still_pending'].append(order_id)
                        
                except Exception as e:
                    logger.error(f"Error checking order {order_id}: {e}")
                    continue
            
            # Remove processed orders from pending
            for order_id in orders_to_remove:
                self.pending_orders.pop(order_id, None)
            
            # Save order history
            if results['filled'] or results['cancelled']:
                self._save_order_history()
            
            return results
            
        except Exception as e:
            logger.error(f"Error in check_and_update_orders: {e}")
            return results

    def _cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order"""
        try:
            import time
            operator_id = int(time.time() * 1000)  # Generate unique operatorId (milliseconds timestamp)
            result = self.bitvavo_api.cancel_order(
                order_id,
                self.pair,
                params={'operatorId': operator_id}
            )
            logger.info(f"Order {order_id} cancellation requested with operatorId {operator_id}")
            return True
        except Exception as e:
            error_msg = str(e).lower()
            if "no active order found" in error_msg or "order not found" in error_msg:
                logger.warning(f"Order {order_id} already cancelled/filled on exchange")
                return True  # Consider this successful since order is no longer active
            else:
                logger.error(f"Failed to cancel order {order_id}: {e}")
                return False

    def force_refresh_pending_orders(self):
        """Force refresh all pending orders by fetching from exchange"""
        try:
            logger.info("🔄 Force refreshing pending orders from exchange...")
            
            # Get all open orders from Bitvavo
            open_orders = self.bitvavo_api.fetch_open_orders(self.pair)
            
            # Update our pending orders with current data
            current_pending = {}
            for order in open_orders:
                order_id = order['id']
                
                # If we're already tracking this order, update it
                if order_id in self.pending_orders:
                    order_info = self.pending_orders[order_id].copy()
                    order_info.update({
                        'status': order.get('status', 'open'),
                        'executed_volume': order.get('filled', 0),
                        'average_price': order.get('average', order_info['price']),
                        'raw_order': order
                    })
                    current_pending[order_id] = order_info
                else:
                    # New order we weren't tracking (shouldn't happen normally)
                    logger.warning(f"Found untracked open order: {order_id}")
                    
            self.pending_orders = current_pending
            logger.info(f"✅ Refreshed pending orders: {len(current_pending)} currently open")
            
        except Exception as e:
            logger.error(f"Failed to refresh pending orders: {e}")

    def clear_stale_pending_orders(self):
        """
        Emergency function to clear stale pending orders.
        Call this if you suspect the bot has stale order state.
        """
        logger.warning("🧹 Clearing all pending orders and resyncing with exchange...")
        
        # Move all current pending to cancelled (they're likely stale)
        current_time = time.time()
        for order_id, order_info in self.pending_orders.items():
            order_info.update({
                'status': 'cancelled',
                'cancelled_at': current_time,
                'reason': 'Cleared as stale order'
            })
            self.cancelled_orders[order_id] = order_info
        
        # Clear pending
        self.pending_orders = {}
        
        # Resync with exchange
        self._sync_with_exchange_on_startup()
        
        # Save state
        self._save_order_history()
        
        logger.info("✅ Stale order cleanup complete")

    # ... (rest of the methods remain the same)
    def get_pending_orders(self) -> Dict:
        """Get all pending orders"""
        return self.pending_orders.copy()

    def get_filled_orders(self, hours: int = 24) -> Dict:
        """Get filled orders from the last N hours"""
        cutoff_time = time.time() - (hours * 3600)
        recent_filled = {}
        
        for order_id, order_info in self.filled_orders.items():
            filled_at = order_info.get('filled_at', order_info.get('timestamp', 0))
            if filled_at > cutoff_time:
                recent_filled[order_id] = order_info
                
        return recent_filled

    def get_order_statistics(self) -> Dict:
        """Get order statistics"""
        total_filled = len(self.filled_orders)
        total_cancelled = len(self.cancelled_orders)
        total_orders = total_filled + total_cancelled
        
        fill_rate = total_filled / total_orders if total_orders > 0 else 0
        
        # Calculate average time to fill
        fill_times = []
        for order_info in self.filled_orders.values():
            filled_at = order_info.get('filled_at', 0)
            timestamp = order_info.get('timestamp', 0)
            if filled_at > 0 and timestamp > 0:
                fill_times.append(filled_at - timestamp)
        
        avg_time_to_fill = sum(fill_times) / len(fill_times) if fill_times else 0
        
        # Calculate total fees
        total_fees = sum(
            order_info.get('fee', 0) for order_info in self.filled_orders.values()
        )
        
        return {
            'total_filled_orders': total_filled,
            'total_cancelled_orders': total_cancelled,
            'fill_rate': fill_rate,
            'avg_time_to_fill': avg_time_to_fill,
            'total_fees_paid': total_fees
        }

    def cleanup_old_orders(self, days: int = 30):
        """Remove old order history to keep file size manageable"""
        cutoff_time = time.time() - (days * 24 * 3600)
        
        # Clean filled orders
        old_filled = [
            order_id for order_id, order_info in self.filled_orders.items()
            if order_info.get('filled_at', order_info.get('timestamp', 0)) < cutoff_time
        ]
        
        for order_id in old_filled:
            self.filled_orders.pop(order_id, None)
        
        # Clean cancelled orders
        old_cancelled = [
            order_id for order_id, order_info in self.cancelled_orders.items()
            if order_info.get('cancelled_at', order_info.get('timestamp', 0)) < cutoff_time
        ]
        
        for order_id in old_cancelled:
            self.cancelled_orders.pop(order_id, None)
        
        if old_filled or old_cancelled:
            logger.info(f"Cleaned up {len(old_filled)} old filled and {len(old_cancelled)} old cancelled orders")
            self._save_order_history()

    def get_order_by_id(self, order_id: str) -> Optional[Dict]:
        """Get order info by ID from any category"""
        if order_id in self.pending_orders:
            return self.pending_orders[order_id]
        elif order_id in self.filled_orders:
            return self.filled_orders[order_id]
        elif order_id in self.cancelled_orders:
            return self.cancelled_orders[order_id]
        else:
            return None