"""
Simple Synchronous Bitvavo API using CCXT
No async, no complexity - just works like your working project
"""

import ccxt
from typing import Dict, List, Optional, Any
from logger_config import logger
from config import BITVAVO_API_KEY, BITVAVO_API_SECRET
import time


class BitvavoAPI:
    """
    Simple CCXT-based Bitvavo API (synchronous)
    Exactly like your working project - proven to work
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        """Initialize CCXT Bitvavo exchange"""
        self.api_key = api_key or BITVAVO_API_KEY
        self.api_secret = api_secret or BITVAVO_API_SECRET
        
        self.exchange = ccxt.bitvavo({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'adjustForTimeDifference': True,
            }
        })
        
        logger.info("✅ Bitvavo API (CCXT sync) initialized")
   

    def get_order_history(self, market: str = "BTC-EUR", limit: int = 1000, 
                        since: int = None) -> list:
        """
        Get closed orders (filled or cancelled)
        
        Args:
            market: Trading pair (e.g., "BTC-EUR")
            limit: Max orders to fetch (default 1000)
            since: Timestamp in milliseconds (None = recent orders)
        
        Returns:
            List of order dictionaries
        """
        try:
            symbol = market.replace('-', '/')
            
            orders = self.exchange.fetch_closed_orders(
                symbol=symbol,
                since=since,
                limit=limit
            )
            
            logger.info(f"📋 Fetched {len(orders)} closed orders")
            return orders
            
        except Exception as e:
            logger.error(f"Failed to fetch order history: {e}")
            return []


    def get_order_from_history(self, order_id: str, market: str = "BTC-EUR", 
                            lookback_days: int = 7) -> dict:
        """
        Find specific order in recent history
        CRITICAL for handling instant fills!
        
        Args:
            order_id: Order ID to find
            market: Trading pair
            lookback_days: How many days to search back (default 7)
        
        Returns:
            Order data if found, None otherwise
        """
        try:
            import time
            symbol = market.replace('-', '/')
            
            # Calculate lookback timestamp (in milliseconds)
            since = int((time.time() - (lookback_days * 86400)) * 1000)
            
            # Fetch recent orders
            orders = self.exchange.fetch_closed_orders(
                symbol=symbol,
                since=since,
                limit=1000
            )
            
            # Find our order
            for order in orders:
                if order['id'] == order_id:
                    logger.info(f"✅ Found order {order_id[:8]}... in history: {order['status']}")
                    
                    # Convert to format matching get_order_async
                    result = {
                        'orderId': order['id'],
                        'status': order.get('status', 'unknown'),
                        'market': market,
                        'side': order.get('side'),
                        'orderType': order.get('type'),
                        'amount': order.get('amount'),
                        'price': order.get('price'),
                        'filledAmount': order.get('filled', 0),
                        'timestamp': order.get('timestamp'),
                        'datetime': order.get('datetime'),
                    }
                    
                    return result
            
            logger.debug(f"Order {order_id[:8]}... not found in last {lookback_days} days")
            return None
            
        except Exception as e:
            logger.error(f"Failed to search order history for {order_id}: {e}")
            return None


    def get_trades_for_order(self, order_id: str, market: str = "BTC-EUR") -> list:
        """
        Get all trades (fills) for a specific order
        Useful for orders that filled in multiple parts
        
        Args:
            order_id: Order ID
            market: Trading pair
        
        Returns:
            List of trade dictionaries
        """
        try:
            symbol = market.replace('-', '/')
            
            # Fetch recent trades (last 1000)
            trades = self.exchange.fetch_my_trades(symbol=symbol, limit=1000)
            
            # Filter to our order
            order_trades = [t for t in trades if t.get('order') == order_id]
            
            if order_trades:
                logger.info(f"📊 Found {len(order_trades)} trade(s) for order {order_id[:8]}...")
                
                # Calculate totals
                total_amount = sum(t['amount'] for t in order_trades)
                total_cost = sum(t['cost'] for t in order_trades)
                total_fee = sum(t['fee']['cost'] for t in order_trades if t.get('fee'))
                
                logger.info(f"   Total: {total_amount:.8f} BTC for €{total_cost:.2f} (fee: €{total_fee:.4f})")
            
            return order_trades
            
        except Exception as e:
            logger.error(f"Failed to fetch trades for order {order_id}: {e}")
            return []


    def get_recent_filled_orders(self, limit: int = 100, market: str = "BTC-EUR") -> list:
        """
        Get recently filled orders (optimized for instant fill checking)
        
        Args:
            limit: Number of recent orders to fetch
            market: Trading pair
        
        Returns:
            List of filled order dictionaries
        """
        try:
            import time
            symbol = market.replace('-', '/')
            
            # Fetch last 24 hours
            since = int((time.time() - 86400) * 1000)
            
            orders = self.exchange.fetch_closed_orders(
                symbol=symbol,
                since=since,
                limit=limit
            )
            
            # Filter to only filled orders (not cancelled)
            filled = [o for o in orders if o['status'] in ['closed', 'filled']]
            
            logger.debug(f"📋 Found {len(filled)} filled orders in last 24h")
            return filled
            
        except Exception as e:
            logger.error(f"Failed to fetch recent filled orders: {e}")
            return []


    def get_order_statistics(self, days: int = 30, market: str = "BTC-EUR") -> dict:
        """
        Get order statistics for recent period
        
        Args:
            days: Number of days to analyze
            market: Trading pair
        
        Returns:
            Dictionary with statistics
        """
        try:
            import time
            from datetime import datetime
            
            symbol = market.replace('-', '/')
            since = int((time.time() - (days * 86400)) * 1000)
            
            # Fetch orders
            orders = self.exchange.fetch_closed_orders(
                symbol=symbol,
                since=since,
                limit=1000
            )
            
            # Calculate stats
            total_orders = len(orders)
            filled_orders = [o for o in orders if o['status'] in ['closed', 'filled']]
            cancelled_orders = [o for o in orders if o['status'] in ['canceled', 'cancelled']]
            
            buy_orders = [o for o in filled_orders if o['side'] == 'buy']
            sell_orders = [o for o in filled_orders if o['side'] == 'sell']
            
            total_btc_bought = sum(o['filled'] for o in buy_orders)
            total_btc_sold = sum(o['filled'] for o in sell_orders)
            
            total_eur_spent = sum(o['cost'] for o in buy_orders)
            total_eur_received = sum(o['cost'] for o in sell_orders)
            
            stats = {
                'period_days': days,
                'total_orders': total_orders,
                'filled_orders': len(filled_orders),
                'cancelled_orders': len(cancelled_orders),
                'fill_rate': len(filled_orders) / total_orders if total_orders > 0 else 0,
                'buy_orders': len(buy_orders),
                'sell_orders': len(sell_orders),
                'btc_bought': total_btc_bought,
                'btc_sold': total_btc_sold,
                'net_btc': total_btc_bought - total_btc_sold,
                'eur_spent': total_eur_spent,
                'eur_received': total_eur_received,
                'net_eur': total_eur_received - total_eur_spent,
            }
            
            logger.info(f"📊 Order Stats (last {days} days):")
            logger.info(f"   Orders: {stats['filled_orders']}/{stats['total_orders']} filled ({stats['fill_rate']:.1%})")
            logger.info(f"   BTC: {stats['btc_bought']:.8f} bought, {stats['btc_sold']:.8f} sold (net: {stats['net_btc']:+.8f})")
            logger.info(f"   EUR: €{stats['eur_spent']:.2f} spent, €{stats['eur_received']:.2f} received (net: €{stats['net_eur']:+.2f})")
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to calculate order statistics: {e}")
            return {}
    
    
    def get_ticker_async(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get ticker data (keeping method name for compatibility)"""
        try:
            ticker = self.exchange.fetch_ticker(symbol.replace('-', '/'))
            return {"price": float(ticker.get('last', 0))}
        except Exception as e:
            logger.error(f"Failed to get ticker: {e}")
            return None
    
    def get_order_history_async(self, order_id: str, market: str = "BTC-EUR") -> Optional[Dict]:
        """Get order from history (for filled/cancelled orders)"""
        try:
            symbol = market.replace('-', '/')
            
            # CCXT method for order history
            orders = self.exchange.fetch_closed_orders(symbol, limit=50)
            
            # Find our order
            for order in orders:
                if order['id'] == order_id:
                    result = {
                        'orderId': order['id'],
                        'status': order.get('status', 'unknown'),
                        'market': order.get('symbol', symbol).replace('/', '-'),
                        'side': order.get('side'),
                        'orderType': order.get('type'),
                        'amount': order.get('amount'),
                        'price': order.get('price'),
                        'filledAmount': order.get('filled', 0),
                    }
                    return result
            
            return None
        except Exception as e:
            logger.debug(f"Order {order_id} not in history: {e}")
            return None
    
    
    def get_order_book_async(self, symbol: str, depth: int = 10) -> Optional[Dict]:
        """Get order book (keeping method name for compatibility)"""
        try:
            order_book = self.exchange.fetch_order_book(symbol.replace('-', '/'))
            logger.debug(f"Order book: {len(order_book.get('bids', []))} bids, {len(order_book.get('asks', []))} asks")
            return order_book
        except Exception as e:
            logger.error(f"Failed to get order book: {e}")
            return None
    
    def get_balance_async(self) -> Optional[Dict[str, Dict[str, float]]]:
        """Get account balance (keeping method name for compatibility)"""
        try:
            balance = self.exchange.fetch_balance()
            
            # Convert CCXT format to match expected format
            result = {}
            for currency, data in balance.items():
                if currency not in ['info', 'free', 'used', 'total', 'timestamp', 'datetime']:
                    result[currency] = {
                        'available': float(data.get('free', 0)),
                        'inOrder': float(data.get('used', 0)),
                        'amount': float(data.get('total', 0)),
                    }
            
            return result
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return None
    
    def place_order_async(self, order_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Place order (keeping method name for compatibility)
        
        This is synchronous despite the name - no async needed!
        """
        try:
            symbol = order_data.get('market', 'BTC-EUR').replace('-', '/')
            side = order_data.get('side', 'buy')
            amount = float(order_data.get('amount', 0))
            price = float(order_data.get('price', 0))
            
            # Generate operatorId
            operator_id = order_data.get('operatorId')
            if not operator_id:
                operator_id = int(time.time() * 1000)
            
            logger.info(f"Placing {side.upper()} order: {amount:.8f} BTC @ €{price:.2f}")
            
            # Place order using CCXT (synchronous)
            order = self.exchange.create_limit_order(
                symbol=symbol,
                side=side,
                amount=amount,
                price=price,
                params={'operatorId': operator_id}
            )
            
            if order and order.get('id'):
                result = {
                    'orderId': order['id'],
                    'status': order.get('status', 'new'),
                    'market': order.get('symbol', symbol).replace('/', '-'),
                    'side': order.get('side', side),
                    'orderType': order.get('type', 'limit'),
                    'amount': order.get('amount', amount),
                    'price': order.get('price', price),
                    'filledAmount': order.get('filled', 0),
                }
                
                logger.info(f"✅ Order placed: {result['orderId']}")
                return result
            else:
                logger.error(f"Order placement failed: {order}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None
    
    def get_order_async(self, order_id: str, market: str = "BTC-EUR") -> Optional[Dict[str, Any]]:
        """
        Get order status (keeping method name for compatibility)
        
        This is synchronous - CCXT handles everything correctly!
        """
        try:
            symbol = market.replace('-', '/')
            order = self.exchange.fetch_order(order_id, symbol)
            
            if order:
                result = {
                    'orderId': order['id'],
                    'status': order.get('status', 'unknown'),
                    'market': order.get('symbol', symbol).replace('/', '-'),
                    'side': order.get('side'),
                    'orderType': order.get('type'),
                    'amount': order.get('amount'),
                    'price': order.get('price'),
                    'filledAmount': order.get('filled', 0),
                }
                
                logger.debug(f"Order status: {result['orderId']} - {result['status']}")
                return result
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None
    
    def cancel_order_async(self, market: str, order_id: str) -> Optional[Dict[str, Any]]:
        """Cancel order (keeping method name for compatibility)"""
        try:
            symbol = market.replace('-', '/')
            operator_id = int(time.time() * 1000)
            
            result = self.exchange.cancel_order(
                order_id,
                symbol,
                params={'operatorId': operator_id}
            )
            
            logger.info(f"✅ Order cancelled: {order_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return None
    
    def close(self):
        """Close connections (CCXT handles this)"""
        logger.info("Closing Bitvavo API")


# Aliases for backwards compatibility
class EnhancedBitvavoAPI(BitvavoAPI):
    """Alias - same class, different name"""
    pass


def load_api_credentials():
    """Load API credentials from config"""
    return BITVAVO_API_KEY, BITVAVO_API_SECRET