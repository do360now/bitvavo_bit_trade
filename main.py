#!/usr/bin/env python3
"""
Bitvavo Bitcoin Trading Bot - Main entry point
Uses async pattern for high-performance trading
"""

import asyncio
import signal
import sys
import json
import os

# Import components
from logger_config import logger
from bitvavo_api import EnhancedBitvavoAPI, load_api_credentials
from order_manager import OrderManager
from trade_executor import TradeExecutor
from data_manager import DataManager
from performance_tracker import PerformanceTracker
from indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_moving_average,
    calculate_vwap,
    fetch_enhanced_news,
    calculate_enhanced_sentiment,
    calculate_sentiment,
)


class BitvavoTradingBot:
    """Main trading bot for Bitvavo BTC/EUR market"""

    def __init__(self):
        self.running = True
        self.market = "BTC-EUR"
        self.lookback_period = 100
        self.price_history = []
        self.price_history_file = "price_history.json"

        # Initialize components
        try:
            api_key, api_secret = load_api_credentials()
            self.bitvavo_api = EnhancedBitvavoAPI(api_key, api_secret)

            self.order_manager = OrderManager(self.bitvavo_api)
            self.trade_executor = TradeExecutor(self.bitvavo_api)
            self.data_manager = DataManager(
                "price_history.json", "trading_bot.log"
            )
            self.performance_tracker = PerformanceTracker()

            # Load saved price history
            self._load_price_history()

            logger.info("✅ Bitcoin trading bot initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize bot: {e}")
            raise

    def _load_price_history(self):
        """Load price history from file if it exists"""
        if os.path.exists(self.price_history_file):
            try:
                with open(self.price_history_file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        self.price_history = [float(p) for p in data]
                        logger.info(f"Loaded {len(self.price_history)} historical prices")
            except Exception as e:
                logger.warning(f"Failed to load price history: {e}")

    def _save_price_history(self):
        """Save price history to file"""
        try:
            with open(self.price_history_file, "w") as f:
                json.dump(self.price_history, f)
        except Exception as e:
            logger.error(f"Failed to save price history: {e}")

    async def run(self):
        """Main bot loop"""
        logger.info("Starting Bitvavo Bitcoin trading bot...")
        save_counter = 0

        try:
            while self.running:
                try:
                    await self.execute_strategy()
                except Exception as e:
                    logger.error(f"Strategy execution failed: {e}", exc_info=True)

                # Save price history every 12 cycles (approximately hourly)
                save_counter += 1
                if save_counter >= 12:
                    self._save_price_history()
                    save_counter = 0

                # Sleep until next cycle (typically every 5-10 minutes)
                await asyncio.sleep(300)

        except KeyboardInterrupt:
            logger.info("Bot interrupted by user")
        except Exception as e:
            logger.error(f"Fatal error in bot loop: {e}", exc_info=True)
        finally:
            await self.cleanup()

    async def execute_strategy(self):
        """Execute main trading strategy"""
        try:
            # Fetch market data
            current_price = await self.trade_executor.get_current_price()
            if not current_price:
                logger.warning("Failed to fetch current price")
                return

            self.price_history.append(current_price)
            self.price_history = self.price_history[-self.lookback_period * 4 :]

            # Get order book
            order_book = await self.trade_executor.get_order_book()
            if not order_book:
                logger.warning("Failed to fetch order book")
                return

            # Get OHLCV data (optional - can trade without it)
            ohlcv = await self.trade_executor.get_ohlcv()
            if not ohlcv:
                logger.debug("OHLCV data unavailable, will use price history only")
                ohlcv = []

            # Calculate technical indicators
            prices = [float(p) for p in self.price_history[-100:]]
            
            # Allow trading with as few as 5 prices for faster feedback
            if len(prices) < 5:
                logger.debug("Accumulating price history... ({}/{})".format(
                    len(prices), 5
                ))
                return

            rsi = calculate_rsi(prices) or 50
            macd_result = calculate_macd(prices)
            macd, signal = (macd_result if macd_result[0] is not None else (0, 0))
            upper_band, ma_short, lower_band = calculate_bollinger_bands(prices) or (
                0,
                0,
                0,
            )
            ma_long = calculate_moving_average(prices, 50) or 0
            vwap = calculate_vwap(prices, [1] * len(prices)) or current_price

            # Fetch news and sentiment
            try:
                articles = fetch_enhanced_news(top_n=20)
                news_analysis = calculate_enhanced_sentiment(articles)
                sentiment = calculate_sentiment(articles)
            except Exception as e:
                logger.warning(f"News analysis failed: {e}")
                news_analysis = {"sentiment": 0.0, "risk_off_probability": 0.0}
                sentiment = 0.0

            # Get balances
            btc_balance = await self.trade_executor.get_total_btc_balance() or 0
            eur_balance = await self.trade_executor.get_available_balance("EUR") or 0

            logger.info(
                f"📊 Market Data - Price: €{current_price:.2f}, BTC: {btc_balance:.8f}, EUR: €{eur_balance:.2f}"
            )
            logger.info(
                f"📈 Indicators - RSI: {rsi:.1f}, MACD: {macd:.4f}, Sentiment: {sentiment:.3f}"
            )

            # Simple trading logic
            buy_signal = rsi < 30 and sentiment > -0.1 and eur_balance > 10
            sell_signal = rsi > 70 and btc_balance > 0.0001

            # Execute trades
            if buy_signal:
                logger.info("🟢 BUY SIGNAL detected")
                position_size = self.trade_executor.calculate_position_size(
                    eur_balance, current_price
                )
                if position_size > 0:
                    optimal_price = await self.trade_executor.get_optimal_price(
                        order_book, "buy"
                    )
                    if optimal_price:
                        order_id = await self.order_manager.place_limit_order(
                            volume=position_size,
                            side="buy",
                            price=optimal_price,
                            market=self.market,
                        )
                        if order_id:
                            logger.info(
                                f"✅ Buy order placed: {position_size:.8f} BTC at €{optimal_price:.2f}"
                            )

            elif sell_signal:
                logger.info("🔴 SELL SIGNAL detected")
                optimal_price = await self.trade_executor.get_optimal_price(
                    order_book, "sell"
                )
                if optimal_price:
                    order_id = await self.order_manager.place_limit_order(
                        volume=btc_balance * 0.9,  # Sell 90% of holdings
                        side="sell",
                        price=optimal_price,
                        market=self.market,
                    )
                    if order_id:
                        logger.info(
                            f"✅ Sell order placed: {btc_balance * 0.9:.8f} BTC at €{optimal_price:.2f}"
                        )

            else:
                logger.info("⏸️  HOLD - No trading signals")

            # Update equity and check pending orders
            self.performance_tracker.update_equity(
                btc_balance, eur_balance, current_price
            )

            results = await self.order_manager.check_and_update_orders()
            if results["filled"]:
                logger.info(f"✅ Filled orders: {results['filled']}")
            if results["cancelled"]:
                logger.info(f"❌ Cancelled orders: {results['cancelled']}")

        except Exception as e:
            logger.error(f"Strategy execution error: {e}", exc_info=True)

    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up...")
        try:
            # Save price history before shutdown
            self._save_price_history()
            await self.bitvavo_api.close()
            logger.info("Bot shutdown complete")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    def stop(self):
        """Stop the bot"""
        self.running = False
        logger.info("Stop signal received")
        exit(0)


async def main():
    """Main entry point"""
    bot = BitvavoTradingBot()

    # Handle shutdown signals
    def signal_handler(signum, frame):
        logger.info(f"Signal {signum} received")
        bot.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await bot.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
