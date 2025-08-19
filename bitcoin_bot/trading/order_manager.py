import time
import json
import os
from typing import Dict, List, Optional, Tuple
from utils.logger import logger
from datetime import datetime, timedelta

class OrderManager:
    """Enhanced Order Manager with better state tracking and error handling"""
    
    def __init__(self, bitvavo_api):
        self.bitvavo_api = bitvavo_api
        self.pair = "BTC/EUR"
        self.pending_orders = {}  # order_id -> order_info
        self.filled_orders = {}   # order_id -> order_info
        self.cancelled_orders = {} # order_id -> order_info
        self.order_history_file = "./order_history.json"
        self.last_sync_time = 0
        self.sync_interval = 300  # Sync with exchange every 5 minutes
        
        # Load existing history and sync with exchange
        self._load_order_history()
        self._sync_with_exchange_on_startup()

    def _sync_with_exchange_on_startup(self):
        """CRITICAL: Sync local state with exchange on startup"""
        logger.info("🔄 Syncing order state with exchange on startup...")
        
        try:
            # Get all open orders from exchange
            exchange_orders = self.bitvavo_api.fetch_open_orders(self.pair)
            exchange_order_ids = {order['id'] for order in exchange_orders}
            
            logger.info(f"Found {len(exchange_orders)} open orders on exchange")
            
            # Check for stale pending orders
            stale_orders = []
            for order_id in list(self.pending_orders.keys()):
                if order_id not in exchange_order_ids:
                    logger.warning(f"⚠️ Order {order_id} is pending locally but not on exchange")
                    stale_orders.append(order_id)
                    
                    # Try to get final status
                    final_status = self._get_final_order_status(order_id)
                    if final_status:
                        self._process_completed_order(order_id, final_status)
                    else:
                        # Default to cancelled if we can't determine status
                        self._mark_order_as_cancelled(order_id, "Not found on exchange during sync")
            
            # Update pending orders with current exchange data
            current_pending = {}
            for order in exchange_orders:
                order_id = order['id']
                
                if order_id in self.pending_orders:
                    # Update existing tracked order
                    order_info = self.pending_orders[order_id].copy()
                    order_info.update({
                        'status': order.get('status', 'open'),
                        'executed_volume': order.get('filled', 0),
                        'average_price': order.get('average', order_info['price']),
                        'raw_order': order,
                        'last_updated': time.time()
                    })
                    current_pending[order_id] = order_info
                else:
                    # Found untracked order
                    logger.warning(f"Found untracked open order: {order_id}")
                    order_info = self._create_order_info_from_exchange(order)
                    current_pending[order_id] = order_info
            
            self.pending_orders = current_pending
            self.last_sync_time = time.time()
            
            if stale_orders:
                logger.info(f"🧹 Cleaned up {len(stale_orders)} stale orders")
                self._save_order_history()
            
            logger.info(f"✅ Sync complete: {len(self.pending_orders)} orders currently pending")
            
        except Exception as e:
            logger.error(f"Failed to sync with exchange: {e}")

    def _get_final_order_status(self, order_id: str) -> Optional[Dict]:
        """Try to get the final status of an order"""
        try:
            return self.bitvavo_api.fetch_order(order_id, self.pair)
        except Exception as e:
            logger.debug(f"Could not fetch final status for order {order_id}: {e}")
            return None

    def _process_completed_order(self, order_id: str, final_status: Dict):
        """Process an order that has been completed"""
        order_info = self.pending_orders.get(order_id, {}).copy()
        status = final_status.get('status', '').lower()
        
        order_info.update({
            'status': status,
            'executed_volume': final_status.get('filled', 0),
            'average_price': final_status.get('average', order_info.get('price', 0)),
            'fee': final_status.get('fee', {}).get('cost', 0),
            'raw_order': final_status,
            'completed_at': time.time()
        })
        
        if status in ['closed', 'filled']:
            order_info['filled_at'] = time.time()
            self.filled_orders[order_id] = order_info
            logger.info(f"✅ Recovered filled order: {order_id}")
        else:
            order_info['cancelled_at'] = time.time()
            self.cancelled_orders[order_id] = order_info
            logger.info(f"❌ Recovered cancelled order: {order_id}")

    def _mark_order_as_cancelled(self, order_id: str, reason: str):
        """Mark an order as cancelled"""
        order_info = self.pending_orders.get(order_id, {}).copy()
        order_info.update({
            'status': 'cancelled',
            'cancelled_at': time.time(),
            'reason': reason
        })
        self.cancelled_orders[order_id] = order_info

    def _create_order_info_from_exchange(self, order: Dict) -> Dict:
        """Create order info structure from exchange order data"""
        return {
            'id': order['id'],
            'side': order['side'],
            'volume': order['amount'],
            'price': order['price'],
            'timestamp': order.get('timestamp', time.time() * 1000) / 1000,
            'timeout': 300,  # Default timeout
            'status': order.get('status', 'open'),
            'executed_volume': order.get('filled', 0),
            'average_price': order.get('average', order['price']),
            'fee': order.get('fee', {}).get('cost', 0),
            'raw_order': order,
            'last_updated': time.time()
        }

    def place_limit_order_with_timeout(self, volume: float, side: str, price: float, 
                                     timeout: int = 300, post_only: bool = False) -> Optional[str]:
        """Place a limit order with enhanced tracking"""
        try:
            logger.info(f"Placing {side} order: {volume:.8f} BTC at €{price:.2f}")
            
            # Generate unique operator ID
            operator_id = int(time.time() * 1000)
            
            # Create the order
            try:
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
                logger.error(f"Order creation failed: {order_error}")
                # Try alternative method
                operator_id = int(time.time() * 1000) + 1
                try:
                    order = self.bitvavo_api.create_order(
                        symbol=self.pair,
                        type='limit',
                        side=side,
                        amount=volume,
                        price=price,
                        params={'operatorId': operator_id}
                    )
                except Exception as alt_error:
                    logger.error(f"Alternative order creation failed: {alt_error}")
                    return None
            
            if not order or not order.get('id'):
                logger.error(f"Invalid order response: {order}")
                return None
            
            order_id = order['id']
            
            # Store comprehensive order info
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
                'raw_order': order,
                'operator_id': operator_id,
                'last_updated': time.time()
            }
            
            self.pending_orders[order_id] = order_info
            
            # Immediate status check to catch quick fills
            time.sleep(0.5)  # Brief pause to allow order processing
            self._check_single_order(order_id)
            
            logger.info(f"✅ {side.upper()} order placed: {order_id}")
            return order_id
            
        except Exception as e:
            logger.error(f"Failed to place {side} order: {e}")
            return None

    def check_and_update_orders(self) -> Dict[str, List[str]]:
        """Enhanced order checking with better error handling"""
        results = {
            'filled': [],
            'cancelled': [],
            'partial': [],
            'still_pending': []
        }
        
        if not self.pending_orders:
            return results
        
        # Periodic full sync with exchange
        if time.time() - self.last_sync_time > self.sync_interval:
            self._periodic_sync_check()
        
        current_time = time.time()
        orders_to_remove = []
        
        for order_id in list(self.pending_orders.keys()):
            try:
                order_info = self.pending_orders[order_id]
                
                # Check timeout
                if current_time - order_info['timestamp'] > order_info['timeout']:
                    logger.info(f"⏰ Order {order_id} timed out, cancelling...")
                    if self._cancel_order(order_id):
                        results['cancelled'].append(order_id)
                        orders_to_remove.append(order_id)
                    continue
                
                # Check order status
                status_result = self._check_single_order(order_id)
                if status_result:
                    results[status_result].append(order_id)
                    if status_result in ['filled', 'cancelled']:
                        orders_to_remove.append(order_id)
                else:
                    results['still_pending'].append(order_id)
                    
            except Exception as e:
                logger.error(f"Error checking order {order_id}: {e}")
                continue
        
        # Remove processed orders
        for order_id in orders_to_remove:
            self.pending_orders.pop(order_id, None)
        
        # Save if there were changes
        if results['filled'] or results['cancelled']:
            self._save_order_history()
        
        return results

    def _check_single_order(self, order_id: str) -> Optional[str]:
        """Check status of a single order"""
        try:
            order_status = self.bitvavo_api.fetch_order(order_id, self.pair)
            
            if not order_status:
                return None
            
            order_info = self.pending_orders[order_id]
            status = order_status.get('status', '').lower()
            filled_amount = order_status.get('filled', 0)
            
            # Update order info
            order_info.update({
                'status': status,
                'executed_volume': filled_amount,
                'average_price': order_status.get('average', order_info['price']),
                'fee': order_status.get('fee', {}).get('cost', 0),
                'raw_order': order_status,
                'last_updated': time.time()
            })
            
            if status in ['closed', 'filled']:
                order_info['filled_at'] = time.time()
                self.filled_orders[order_id] = order_info.copy()
                logger.info(f"✅ Order {order_id} FILLED: {filled_amount:.8f} BTC @ €{order_info['average_price']:.2f}")
                return 'filled'
                
            elif status in ['canceled', 'cancelled']:
                order_info['cancelled_at'] = time.time()
                self.cancelled_orders[order_id] = order_info.copy()
                logger.info(f"❌ Order {order_id} CANCELLED")
                return 'cancelled'
                
            elif filled_amount > 0 and status == 'open':
                logger.info(f"📊 Order {order_id} PARTIAL: {filled_amount:.8f}/{order_info['volume']:.8f} BTC")
                return 'partial'
            
            return None  # Still pending
            
        except Exception as e:
            error_msg = str(e).lower()
            if "no active order found" in error_msg or "order not found" in error_msg:
                # Order no longer exists on exchange
                logger.warning(f"Order {order_id} not found on exchange")
                order_info = self.pending_orders[order_id]
                order_info.update({
                    'status': 'not_found',
                    'cancelled_at': time.time(),
                    'reason': 'Order not found on exchange'
                })
                self.cancelled_orders[order_id] = order_info.copy()
                return 'cancelled'
            else:
                logger.warning(f"Could not check order {order_id}: {e}")
                return None

    def _periodic_sync_check(self):
        """Periodic sync to catch any missed updates"""
        try:
            exchange_orders = self.bitvavo_api.fetch_open_orders(self.pair)
            exchange_order_ids = {order['id'] for order in exchange_orders}
            
            # Check for orders that are pending locally but not on exchange
            for order_id in list(self.pending_orders.keys()):
                if order_id not in exchange_order_ids:
                    logger.info(f"🔍 Order {order_id} no longer on exchange, investigating...")
                    final_status = self._get_final_order_status(order_id)
                    if final_status:
                        self._process_completed_order(order_id, final_status)
                        self.pending_orders.pop(order_id, None)
            
            self.last_sync_time = time.time()
            
        except Exception as e:
            logger.error(f"Periodic sync check failed: {e}")

    def _cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order"""
        try:
            operator_id = int(time.time() * 1000)
            result = self.bitvavo_api.cancel_order(
                order_id,
                self.pair,
                params={'operatorId': operator_id}
            )
            logger.info(f"Order {order_id} cancellation requested")
            return True
        except Exception as e:
            error_msg = str(e).lower()
            if "no active order found" in error_msg or "order not found" in error_msg:
                logger.warning(f"Order {order_id} already completed")
                return True
            else:
                logger.error(f"Failed to cancel order {order_id}: {e}")
                return False

    def _load_order_history(self):
        """Load order history from file"""
        try:
            if os.path.exists(self.order_history_file):
                with open(self.order_history_file, 'r') as f:
                    data = json.load(f)
                    self.filled_orders = data.get('filled_orders', {})
                    self.cancelled_orders = data.get('cancelled_orders', {})
                    # Don't load pending orders - they'll be synced with exchange
                logger.info(f"Loaded order history: {len(self.filled_orders)} filled, {len(self.cancelled_orders)} cancelled")
        except Exception as e:
            logger.error(f"Failed to load order history: {e}")

    def _save_order_history(self):
        """Save order history to file"""
        try:
            data = {
                'filled_orders': self.filled_orders,
                'cancelled_orders': self.cancelled_orders,
                'last_updated': time.time(),
                'last_sync': self.last_sync_time
            }
            with open(self.order_history_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved order history")
        except Exception as e:
            logger.error(f"Failed to save order history: {e}")

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
        """Get comprehensive order statistics"""
        total_filled = len(self.filled_orders)
        total_cancelled = len(self.cancelled_orders)
        total_orders = total_filled + total_cancelled
        
        fill_rate = total_filled / total_orders if total_orders > 0 else 0
        
        # Calculate metrics from filled orders
        fill_times = []
        total_fees = 0
        total_volume = 0
        
        for order_info in self.filled_orders.values():
            # Fill time calculation
            filled_at = order_info.get('filled_at', 0)
            timestamp = order_info.get('timestamp', 0)
            if filled_at > 0 and timestamp > 0:
                fill_times.append(filled_at - timestamp)
            
            # Fee and volume calculation
            total_fees += order_info.get('fee', 0)
            total_volume += order_info.get('executed_volume', 0)
        
        avg_time_to_fill = sum(fill_times) / len(fill_times) if fill_times else 0
        
        return {
            'total_filled_orders': total_filled,
            'total_cancelled_orders': total_cancelled,
            'total_orders': total_orders,
            'fill_rate': fill_rate,
            'avg_time_to_fill': avg_time_to_fill,
            'total_fees_paid': total_fees,
            'total_volume_traded': total_volume
        }

    def cleanup_old_orders(self, days: int = 30):
        """Remove old order history to keep files manageable"""
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

    def force_refresh_all_orders(self):
        """Emergency function to refresh all order states"""
        logger.warning("🔄 Force refreshing all order states...")
        
        # Clear current state
        self.pending_orders = {}
        
        # Re-sync with exchange
        self._sync_with_exchange_on_startup()
        
        logger.info("✅ Force refresh complete")

    def get_recent_trade_count(self, hours: int = 24) -> int:
        """Get count of recent trades"""
        recent_filled = self.get_filled_orders(hours)
        return len(recent_filled)

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