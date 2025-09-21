# main_optimized.py
import asyncio
import signal
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import uvloop  # For better async performance

from utils.secure_config import TradingConfig
from bitvavo_api_async import APIConnectionPool
from trading.optimized_strategies import AdaptiveStrategyEngine, MarketData
from monitoring.performance_monitor import PerformanceMonitor
from core.indicators import calculate_rsi, calculate_macd, calculate_vwap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bitcoin_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptimizedBitcoinBot:
    """Optimized Bitcoin trading bot with async operations and enhanced performance."""
    
    def __init__(self):
        self.config = TradingConfig()
        self.running = False
        self.api_pool: Optional[APIConnectionPool] = None
        self.strategy_engine = AdaptiveStrategyEngine()
        self.performance_monitor = PerformanceMonitor()
        
        # Trading state
        self.btc_balance = 0.0
        self.eur_balance = 0.0
        self.total_trades = 0
        self.successful_trades = 0
        self.daily_trades = 0
        self.session_start = datetime.now()
        
        # Market data cache
        self.last_market_data: Optional[MarketData] = None
        self.price_history = []
        self.volume_history = []
    
    async def initialize(self):
        """Initialize bot components."""
        try:
            logger.info("Initializing optimized Bitcoin bot...")
            
            # Initialize API connection pool
            api_key, api_secret = self.config.api_credentials
            self.api_pool = APIConnectionPool(api_key, api_secret, pool_size=3)
            await self.api_pool.initialize()
            
            # Test connectivity
            async with self.api_pool.get_connection() as api:
                balance = await api.get_balance()
                logger.info("✅ API connectivity confirmed")
            
            # Initialize balances
            await self._update_balances()
            
            logger.info(f"✅ Bot initialized - BTC: {self.btc_balance:.6f}, EUR: {self.eur_balance:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to initialize bot: {e}")
            raise
    
    async def run(self):
        """Main trading loop with optimized execution."""
        self.running = True
        
        try:
            # Setup signal handlers
            self._setup_signal_handlers()
            
            logger.info("🚀 Starting optimized trading loop...")
            
            while self.running:
                cycle_start = datetime.now()
                
                try:
                    # Execute trading cycle
                    await self._execute_trading_cycle()
                    
                    # Calculate cycle time and sleep
                    cycle_time = (datetime.now() - cycle_start).total_seconds()
                    sleep_time = max(0, 15 - cycle_time)  # 15-second cycles
                    
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
                    
                except Exception as e:
                    logger.error(f"Trading cycle error: {e}")
                    await asyncio.sleep(60)  # Wait 1 minute on error
            
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            await self.shutdown()
    
    async def _execute_trading_cycle(self):
        """Execute a single trading cycle."""
        
        # 1. Fetch market data
        market_data = await self._fetch_market_data()
        if not market_data:
            logger.warning("Failed to fetch market data, skipping cycle")
            return
        
        # 2. Generate trading signal
        signal = await self.strategy_engine.generate_signal(market_data)
        
        # 3. Execute trade if needed
        if signal.action.value != "hold":
            success = await self._execute_trade(signal)
            
            # Update strategy performance
            result = {"success": success, "btc_gained": 0.0}  # Would calculate actual BTC gained
            self.strategy_engine.update_strategy_performance("accumulator", signal, result)
        
        # 4. Update performance metrics
        await self._update_performance_metrics()
        
        # 5. Log cycle summary
        logger.info(f"Cycle: {signal.action.value.upper()} "
                   f"(confidence: {signal.confidence:.1%}) - "
                   f"BTC: {self.btc_balance:.6f}")
    
    async def _fetch_market_data(self) -> Optional[MarketData]:
        """Fetch comprehensive market data."""
        try:
            async with self.api_pool.get_connection() as api:
                # Fetch ticker and order book concurrently
                ticker_task = api.get_ticker("BTC-EUR")
                orderbook_task = api.get_order_book("BTC-EUR", depth=10)
                ohlcv_task = api.get_ohlcv("BTC-EUR", "1m", limit=20)
                
                ticker, orderbook, ohlcv = await asyncio.gather(
                    ticker_task, orderbook_task, ohlcv_task
                )
            
            # Extract basic market data
            price = float(ticker["price"])
            
            # Calculate bid/ask from order book
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
            
            if not bids or not asks:
                logger.warning("Empty order book")
                return None
            
            bid = float(bids[0][0])
            ask = float(asks[0][0])
            spread = ask - bid
            
            # Get volume from recent candles
            if ohlcv:
                recent_volume = sum(float(candle[5]) for candle in ohlcv[-5:]) / 5
                
                # Update price history for technical indicators
                prices = [float(candle[4]) for candle in ohlcv]
                volumes = [float(candle[5]) for candle in ohlcv]
                
                self.price_history.extend(prices)
                self.volume_history.extend(volumes)
                
                # Keep only recent history
                if len(self.price_history) > 200:
                    self.price_history = self.price_history[-100:]
                    self.volume_history = self.volume_history[-100:]
            else:
                recent_volume = 1000.0  # Default
            
            # Calculate technical indicators
            rsi = None
            macd = None
            signal_line = None
            volatility = None
            
            if len(self.price_history) >= 20:
                try:
                    rsi = calculate_rsi(self.price_history)
                    macd, signal_line = calculate_macd(self.price_history)
                    
                    # Calculate volatility
                    import numpy as np
                    returns = np.diff(np.log(self.price_history[-20:]))
                    volatility = np.std(returns) * np.sqrt(24 * 365)  # Annualized
                    
                except Exception as e:
                    logger.warning(f"Technical indicator calculation failed: {e}")
            
            # Calculate order book imbalance
            total_bid_volume = sum(float(bid[1]) for bid in bids[:5])
            total_ask_volume = sum(float(ask[1]) for ask in asks[:5])
            total_volume = total_bid_volume + total_ask_volume
            
            order_book_imbalance = (total_bid_volume - total_ask_volume) / total_volume if total_volume > 0 else 0
            
            # Calculate VWAP
            vwap = None
            if len(self.price_history) >= 10 and len(self.volume_history) >= 10:
                try:
                    vwap = calculate_vwap(self.price_history[-20:], self.volume_history[-20:])
                except:
                    pass
            
            market_data = MarketData(
                timestamp=datetime.now(),
                price=price,
                volume=recent_volume,
                bid=bid,
                ask=ask,
                spread=spread,
                rsi=rsi,
                macd=macd,
                signal=signal_line,
                volatility=volatility,
                order_book_imbalance=order_book_imbalance,
                volume_weighted_price=vwap
            )
            
            self.last_market_data = market_data
            return market_data
            
        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}")
            return None
    
    async def _execute_trade(self, signal) -> bool:
        """Execute trade with optimal parameters."""
        try:
            async with self.api_pool.get_connection() as api:
                
                # Determine order type based on urgency
                if signal.urgency > 0.8:
                    # High urgency - use market order
                    order = await api.create_order(
                        symbol="BTC-EUR",
                        side=signal.action.value,
                        order_type="market",
                        amount=signal.size
                    )
                else:
                    # Normal urgency - use limit order
                    # Adjust price based on market conditions
                    if signal.action.value == "buy":
                        limit_price = signal.price * (1 + signal.max_slippage)
                    else:
                        limit_price = signal.price * (1 - signal.max_slippage)
                    
                    order = await api.create_order(
                        symbol="BTC-EUR",
                        side=signal.action.value,
                        order_type="limit",
                        amount=signal.size,
                        price=limit_price
                    )
                
                if order and order.get("id"):
                    logger.info(f"✅ {signal.action.value.upper()} order placed: "
                               f"{signal.size:.6f} BTC @ €{signal.price:.2f}")
                    
                    self.total_trades += 1
                    self.daily_trades += 1
                    
                    # Update balances
                    await self._update_balances()
                    
                    return True
                else:
                    logger.error(f"Failed to place order: {order}")
                    return False
                    
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return False
    
    async def _update_balances(self):
        """Update current balances."""
        try:
            async with self.api_pool.get_connection() as api:
                balance = await api.get_balance()
                
                for asset in balance:
                    if asset["symbol"] == "BTC":
                        self.btc_balance = float(asset["available"])
                    elif asset["symbol"] == "EUR":
                        self.eur_balance = float(asset["available"])
                        
        except Exception as e:
            logger.error(f"Failed to update balances: {e}")
    
    async def _update_performance_metrics(self):
        """Update performance monitoring."""
        try:
            current_price = self.last_market_data.price if self.last_market_data else 0
            
            await self.performance_monitor.update_metrics(
                btc_balance=self.btc_balance,
                eur_balance=self.eur_balance,
                btc_price=current_price,
                trades_today=self.daily_trades,
                total_trades=self.total_trades,
                successful_trades=self.successful_trades,
                volatility=self.last_market_data.volatility if self.last_market_data else 0
            )
            
        except Exception as e:
            logger.error(f"Failed to update performance metrics: {e}")
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers."""
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, initiating shutdown...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down bot...")
        
        self.running = False
        
        if self.api_pool:
            await self.api_pool.close_all()
        
        # Final performance summary
        summary = self.performance_monitor.get_performance_summary(days=1)
        logger.info(f"Session summary: {summary}")
        
        logger.info("✅ Shutdown complete")

async def main():
    """Main entry point with uvloop for better performance."""
    
    # Use uvloop for better async performance on Linux
    if sys.platform != "win32":
        try:
            uvloop.install()
            logger.info("Using uvloop for enhanced performance")
        except ImportError:
            logger.info("uvloop not available, using default event loop")
    
    bot = OptimizedBitcoinBot()
    
    try:
        await bot.initialize()
        await bot.run()
    except Exception as e:
        logger.error(f"Bot execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())