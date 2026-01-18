"""
Atomic Trade Manager - Ensures state updates match actual order fills
"""

import time
import json
from typing import Optional, Dict
from logger_config import logger


class AtomicTradeManager:
    """
    Manages trades atomically - only updates state after confirming order fills
    """
    
    def __init__(self, order_manager, state_manager, trade_executor):
        self.order_manager = order_manager
        self.state_manager = state_manager
        self.trade_executor = trade_executor
        
        # Track pending state updates
        self.pending_updates = {}
        self.pending_file = "./pending_updates.json"
        self._load_pending()
    
    def _load_pending(self):
        """Load pending updates from disk"""
        try:
            with open(self.pending_file, 'r') as f:
                self.pending_updates = json.load(f)
                if self.pending_updates:
                    logger.warning(f"⚠️  Found {len(self.pending_updates)} pending state updates")
        except FileNotFoundError:
            self.pending_updates = {}
        except Exception as e:
            logger.error(f"Failed to load pending updates: {e}")
            self.pending_updates = {}
    
    def _save_pending(self):
        """Save pending updates to disk"""
        try:
            with open(self.pending_file, 'w') as f:
                json.dump(self.pending_updates, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save pending updates: {e}")
    
    def execute_buy(self, volume: float, price: float, market: str = "BTC-EUR") -> Optional[str]:
        """
        Execute buy order atomically:
        1. Write pending state update to disk
        2. Place order
        3. Return order ID (state update happens later after fill confirmation)
        """
        # Prepare pending update
        pending_id = f"buy_{int(time.time() * 1000)}"
        
        self.pending_updates[pending_id] = {
            'type': 'buy',
            'price': price,
            'volume': volume,
            'timestamp': time.time(),
            'order_id': None,  # Will be filled in after order placement
        }
        self._save_pending()
        
        # Place order
        order_id = self.order_manager.place_limit_order(
            volume=volume,
            side='buy',
            price=price,
            market=market
        )
        
        if order_id:
            # Link order ID to pending update
            self.pending_updates[pending_id]['order_id'] = order_id
            self._save_pending()
            
            logger.info(f"✅ Buy order placed: {order_id} (pending state update: {pending_id})")
            return order_id
        else:
            # Order placement failed - remove pending update
            del self.pending_updates[pending_id]
            self._save_pending()
            logger.error(f"❌ Buy order failed - rolled back pending update")
            return None
    
    def execute_sell(self, volume: float, price: float, market: str = "BTC-EUR") -> Optional[str]:
        """
        Execute sell order atomically (similar to buy)
        """
        pending_id = f"sell_{int(time.time() * 1000)}"
        
        self.pending_updates[pending_id] = {
            'type': 'sell',
            'price': price,
            'volume': volume,
            'timestamp': time.time(),
            'order_id': None,
        }
        self._save_pending()
        
        order_id = self.order_manager.place_limit_order(
            volume=volume,
            side='sell',
            price=price,
            market=market
        )
        
        if order_id:
            self.pending_updates[pending_id]['order_id'] = order_id
            self._save_pending()
            logger.info(f"✅ Sell order placed: {order_id} (pending state update: {pending_id})")
            return order_id
        else:
            del self.pending_updates[pending_id]
            self._save_pending()
            logger.error(f"❌ Sell order failed - rolled back pending update")
            return None
    
    def process_filled_orders(self) -> Dict:
        """
        Check for filled orders and apply their corresponding state updates
        Call this periodically from main loop
        """
        results = {
            'processed': 0,
            'failed': 0,
            'still_pending': 0
        }
        
        # Check all orders
        order_results = self.order_manager.check_and_update_orders()
        
        # Process fills
        for order_id in order_results['filled']:
            # Find corresponding pending update
            pending_id = None
            for pid, update in self.pending_updates.items():
                if update.get('order_id') == order_id:
                    pending_id = pid
                    break
            
            if not pending_id:
                logger.warning(f"⚠️  Filled order {order_id} has no pending update (maybe old order)")
                continue
            
            # Get actual fill data from order manager
            order_data = self.order_manager.filled_orders.get(order_id)
            if not order_data:
                logger.error(f"❌ Cannot find order data for {order_id}")
                results['failed'] += 1
                continue
            
            # Extract actual fill details
            filled_amount = float(order_data.get('filledAmount', 0))
            actual_price = float(order_data.get('price', 0))
            side = order_data.get('side')
            
            # Apply state update with ACTUAL fill data (not expected)
            try:
                if side == 'buy':
                    self.state_manager.update_buy(actual_price, filled_amount)
                    logger.info(f"✅ Applied BUY state update: {filled_amount:.8f} BTC @ €{actual_price:,.2f}")
                elif side == 'sell':
                    self.state_manager.update_sell(actual_price, filled_amount)
                    logger.info(f"✅ Applied SELL state update: {filled_amount:.8f} BTC @ €{actual_price:,.2f}")
                
                # Remove from pending
                del self.pending_updates[pending_id]
                self._save_pending()
                results['processed'] += 1
                
            except Exception as e:
                logger.error(f"❌ Failed to apply state update for {order_id}: {e}")
                results['failed'] += 1
        
        # Clean up old pending updates (>24 hours old, likely cancelled or failed)
        current_time = time.time()
        old_updates = [
            pid for pid, update in self.pending_updates.items()
            if current_time - update.get('timestamp', 0) > 86400  # 24 hours
        ]
        
        for pid in old_updates:
            logger.warning(f"⚠️  Removing stale pending update: {pid}")
            del self.pending_updates[pid]
        
        if old_updates:
            self._save_pending()
        
        results['still_pending'] = len(self.pending_updates)
        
        if results['processed'] > 0 or results['failed'] > 0 or results['still_pending'] > 0:
            logger.info(f"📊 Order processing: {results['processed']} processed, "
                       f"{results['failed']} failed, {results['still_pending']} still pending")
        
        return results
    
    def get_pending_summary(self) -> Dict:
        """Get summary of pending updates"""
        buy_pending = sum(1 for u in self.pending_updates.values() if u['type'] == 'buy')
        sell_pending = sum(1 for u in self.pending_updates.values() if u['type'] == 'sell')
        
        return {
            'total_pending': len(self.pending_updates),
            'buy_pending': buy_pending,
            'sell_pending': sell_pending,
            'pending_updates': self.pending_updates
        }