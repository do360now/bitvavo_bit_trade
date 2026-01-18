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
    
    def get_ticker_async(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get ticker data (keeping method name for compatibility)"""
        try:
            ticker = self.exchange.fetch_ticker(symbol.replace('-', '/'))
            return {"price": float(ticker.get('last', 0))}
        except Exception as e:
            logger.error(f"Failed to get ticker: {e}")
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