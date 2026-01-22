#!/usr/bin/env python3
"""
Bitcoin Accumulation Strategy - SELL AT PEAKS Edition
Goal: Sell at the TOP of moves to maximize EUR, then buy MORE Bitcoin on dips

Key Insight: We want to sell LATE in the rally (near the peak), 
not EARLY when there's still momentum!
"""

import time
import signal
import sys
import json
import os

from logger_config import logger
from bitvavo_api import EnhancedBitvavoAPI, load_api_credentials
from order_manager import OrderManager
from trade_executor import TradeExecutor
from data_manager import DataManager
from performance_tracker import PerformanceTracker
from bot_state_manager import BotStateManager
from atomic_trade_manager import AtomicTradeManager
from indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_moving_average,
    fetch_enhanced_news,
    calculate_sentiment,
)


class BitcoinAccumulationBot:
    """
    Accumulation-focused: Sell at peaks to generate maximum EUR for rebuys
    """

    def __init__(self):
        self.running = True
        self.market = "BTC-EUR"
        self.lookback_period = 100
        self.price_history = []
        self.price_history_file = "price_history.json"
        
        # Accumulation parameters
        self.min_btc_reserve = 0.01
        self.buy_rsi_threshold = 30  # Buy dips (lowered from 35 to catch more opportunities)
        
        # PEAK DETECTION for maximum EUR generation
        self.peak_detection_lookback = 20  # Track recent price peaks
        self.min_profit_to_consider_sell = 9.0  # At least 9% profit
        self.aggressive_profit_target = 14.0  # Sell at 14%+ regardless
        
        # Track price peaks and momentum shifts
        self.recent_prices = []
        self.recent_macd_values = []
        self.recent_signal_values = []  # Track signal line for crossover detection
        self.peak_price = 0.0
        self.peak_timestamp = 0
        
        # Average buy tracking - NOW PERSISTENT!
        self.state_manager = BotStateManager("./bot_state.json")
        self.avg_buy_price = self.state_manager.get_avg_buy_price()
        self.total_btc_bought = self.state_manager.get_total_btc_bought()
        self.peak_price = self.state_manager.get_peak_price()

        # Initialize components
        try:
            api_key, api_secret = load_api_credentials()
            self.bitvavo_api = EnhancedBitvavoAPI(api_key, api_secret)
            self.trade_executor = TradeExecutor(self.bitvavo_api)
            self.order_manager = OrderManager(self.bitvavo_api, self.trade_executor)
            self.atomic_trader = AtomicTradeManager(
                self.order_manager,
                self.state_manager,
                self.trade_executor
            )
            self.data_manager = DataManager("price_history.json", "bot_logs.csv")
            self.performance_tracker = PerformanceTracker()
            self._load_price_history()
            

            logger.info("✅ Bitcoin accumulation bot initialized (PEAK SELLER)")
            logger.info("🎯 Goal: Sell at peaks → Generate EUR → Buy dips → Accumulate MORE BTC")

            pending = self.atomic_trader.get_pending_summary()
            if pending['total_pending'] > 0:
                logger.warning(f"⚠️  Found {pending['total_pending']} pending state updates from previous run")
            
            # Show current state
            if self.avg_buy_price > 0:
                logger.info(f"📊 Loaded state: Avg Buy €{self.avg_buy_price:,.2f}, "
                           f"Total Bought {self.total_btc_bought:.8f} BTC")
            else:
                logger.info("📊 Starting fresh - no previous state found")

        except Exception as e:
            logger.error(f"Failed to initialize bot: {e}")
            raise

    def _load_price_history(self):
        """Load price history from file"""
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
        """Save price history"""
        try:
            with open(self.price_history_file, "w") as f:
                json.dump(self.price_history, f)
        except Exception as e:
            logger.error(f"Failed to save price history: {e}")

    def _detect_peak_reversal(self, current_price: float, macd: float, signal: float, 
                              rsi: float, prices: list) -> tuple[bool, str]:
        """
        Detect if we're at or past a peak - CRITICAL for accumulation strategy
        
        Returns: (should_sell, reason)
        """
        # Track recent prices for peak detection
        self.recent_prices.append(current_price)
        self.recent_prices = self.recent_prices[-self.peak_detection_lookback:]
        
        # Note: MACD tracking is now done in execute_strategy to avoid duplication
        
        # Update peak tracking
        if current_price > self.peak_price:
            self.peak_price = current_price
            self.peak_timestamp = time.time()
            self.state_manager.update_peak(current_price)
        
        if len(self.recent_prices) < 10 or len(self.recent_macd_values) < 5:
            return False, "Insufficient data"
        
        # Calculate profit %
        if self.avg_buy_price > 0:
            profit_percent = ((current_price - self.avg_buy_price) / self.avg_buy_price) * 100
        else:
            profit_percent = 0
        
        # Don't sell below minimum profit threshold
        if profit_percent < self.min_profit_to_consider_sell:
            return False, f"Profit {profit_percent:.1f}% below minimum {self.min_profit_to_consider_sell}%"
        
        # === PEAK DETECTION SIGNALS ===
        
        # Signal 1: Price declining from peak
        price_off_peak_pct = ((self.peak_price - current_price) / self.peak_price) * 100
        price_declining = price_off_peak_pct > 0.3  # 0.3% below peak
        
        # Signal 2: MACD bearish crossover (momentum shift)
        macd_bearish_cross = macd < signal
        
        # Signal 3: MACD declining (losing steam)
        if len(self.recent_macd_values) >= 3:
            macd_declining = (
                self.recent_macd_values[-1] < self.recent_macd_values[-2] and
                self.recent_macd_values[-2] < self.recent_macd_values[-3]
            )
        else:
            macd_declining = False
        
        # Signal 4: Extreme RSI (exhaustion)
        extreme_rsi = rsi > 85
        
        # Signal 5: Making lower highs (trend break)
        if len(self.recent_prices) >= 15:
            recent_high = max(self.recent_prices[-7:])
            previous_high = max(self.recent_prices[-15:-7])
            lower_highs = recent_high < previous_high * 0.999  # Lower by 0.1%
        else:
            lower_highs = False
        
        # Signal 6: MACD negative (momentum loss)
        macd_negative = macd < 0
        
        # === SELL DECISION LOGIC ===
        
        # SCENARIO 1: Aggressive profit taking (protect big gains)
        if profit_percent >= self.aggressive_profit_target:
            if extreme_rsi or price_declining or macd_declining:
                return True, f"💰 Aggressive profit taking: {profit_percent:.1f}% profit"
        
        # SCENARIO 2: Clear peak reversal (ideal sell point)
        reversal_signals = sum([
            price_declining,
            macd_bearish_cross,
            macd_declining,
            lower_highs,
        ])
        
        if reversal_signals >= 3 and profit_percent >= self.min_profit_to_consider_sell:
            return True, f"🔴 Peak reversal detected: {reversal_signals}/4 signals, {profit_percent:.1f}% profit"
        
        # SCENARIO 3: Strong reversal with good profit
        if reversal_signals >= 2 and profit_percent >= 5.0:
            if extreme_rsi or macd_negative:
                return True, f"🟠 Strong reversal: {reversal_signals}/4 signals + exhaustion, {profit_percent:.1f}% profit"
        
        # SCENARIO 4: Momentum completely lost
        if macd_negative and macd_declining and profit_percent >= 4.0:
            return True, f"⚫ Momentum lost: MACD negative & declining, {profit_percent:.1f}% profit"
        
        # Don't sell - trend still strong
        reason = f"✋ HOLD: Profit {profit_percent:.1f}%, Reversal signals {reversal_signals}/4"
        if price_declining:
            reason += f", {price_off_peak_pct:.1f}% off peak"
        else:
            reason += ", still near peak"
        
        return False, reason

    def run(self):
        """Main bot loop"""
        try:
            logger.info("🤖 Starting Bitcoin accumulation bot...")
            save_counter = 0

            while self.running:
                # Execute strategy
                self.execute_strategy()
                
                # NEW: Process filled orders and apply state updates
                self.atomic_trader.process_filled_orders()
                
                # Periodic saves
                save_counter += 1
                if save_counter >= 3:
                    self._save_price_history()
                    save_counter = 0

                time.sleep(300)  # 5 minutes

        except KeyboardInterrupt:
            logger.info("Bot interrupted by user")
        except Exception as e:
            logger.error(f"Fatal error in bot loop: {e}", exc_info=True)
        finally:
            self.cleanup()

    def execute_strategy(self):
        """Execute accumulation-focused trading strategy"""
        try:
            # Fetch market data
            current_price = self.trade_executor.get_current_price()
            if not current_price:
                logger.warning("Failed to fetch current price")
                return

            self.price_history.append(current_price)
            self.price_history = self.price_history[-self.lookback_period * 4:]

            order_book = self.trade_executor.get_order_book()
            if not order_book:
                logger.warning("Failed to fetch order book")
                return

            # Calculate indicators
            prices = [float(p) for p in self.price_history[-100:]]
            
            if len(prices) < 14:
                logger.debug(f"Accumulating price history... ({len(prices)}/14)")
                return

            rsi = calculate_rsi(prices) or 50
            macd_result = calculate_macd(prices)
            macd, signal = (macd_result if macd_result[0] is not None else (0, 0))
            upper_band, ma_short, lower_band = calculate_bollinger_bands(prices) or (0, 0, 0)

            # Sentiment
            try:
                articles = fetch_enhanced_news(top_n=20)
                sentiment = calculate_sentiment(articles)
            except Exception as e:
                logger.debug(f"News analysis unavailable: {e}")
                sentiment = 0.0

            # Balances
            btc_balance = self.trade_executor.get_total_btc_balance() or 0
            eur_balance = self.trade_executor.get_available_balance("EUR") or 0

            # Calculate profit
            if self.avg_buy_price > 0:
                profit_percent = ((current_price - self.avg_buy_price) / self.avg_buy_price) * 100
                rebuy_target = self.avg_buy_price * 0.97  # Target 3% below avg buy
            else:
                profit_percent = 0
                rebuy_target = current_price * 0.95

            logger.info("=" * 80)
            logger.info(f"📊 MARKET | Price: €{current_price:,.2f} (Peak: €{self.peak_price:,.2f})")
            logger.info(f"💼 PORTFOLIO | BTC: {btc_balance:.8f} | EUR: €{eur_balance:.2f}")
            logger.info(f"📈 INDICATORS | RSI: {rsi:.1f} | MACD: {macd:.2f} (Signal: {signal:.2f})")
            logger.info(f"💰 PROFIT | Avg Buy: €{self.avg_buy_price:,.2f} | Current: €{current_price:,.2f} | Profit: {profit_percent:.2f}%")

            # === BUY LOGIC (Enhanced with multiple signals) ===
            min_eur_for_order = self.trade_executor.min_order_size * current_price * 1.05
            
            # Track MACD and signal for crossover detection
            self.recent_macd_values.append(macd)
            self.recent_signal_values.append(signal)
            self.recent_macd_values = self.recent_macd_values[-10:]
            self.recent_signal_values = self.recent_signal_values[-10:]
            
            # === SIGNAL 1: RSI Oversold ===
            rsi_oversold = rsi < self.buy_rsi_threshold
            
            # === SIGNAL 2: MACD Bullish Crossover ===
            macd_bullish_cross = False
            if len(self.recent_macd_values) >= 2 and len(self.recent_signal_values) >= 2:
                # MACD just crossed above signal (or is above)
                macd_bullish_cross = (
                    macd > signal and  # Currently above
                    macd > -100 and  # Not in extreme downtrend
                    self.recent_macd_values[-2] <= self.recent_signal_values[-2]  # Was below or equal recently
                )
            
            # === SIGNAL 3: MACD Improving ===
            macd_improving = False
            if len(self.recent_macd_values) >= 3:
                macd_improving = (
                    self.recent_macd_values[-1] > self.recent_macd_values[-2] and
                    self.recent_macd_values[-2] > self.recent_macd_values[-3]
                )
            
            # === SAFETY FILTERS ===
            safe_to_buy = True
            safety_blocks = []
            
            # Safety 1: Not in extreme downtrend
            if abs(macd) > 200:
                safe_to_buy = False
                safety_blocks.append(f"Extreme MACD ({macd:.1f})")
            
            # Safety 2: Not making aggressive lower lows (mild check)
            if len(self.price_history) >= 10:
                recent_low = min(self.price_history[-5:])
                previous_low = min(self.price_history[-10:-5])
                if recent_low < previous_low * 0.98:  # More than 2% lower
                    safe_to_buy = False
                    safety_blocks.append("Sharp lower lows")
            
            # Safety 3: Sentiment not extremely negative
            if sentiment < -0.5:
                safe_to_buy = False
                safety_blocks.append(f"Extreme negative sentiment ({sentiment:.2f})")
            
            # === BUY DECISION ===
            # Buy if: (RSI oversold OR MACD crossover OR MACD improving) AND safety checks pass AND have funds
            buy_trigger = rsi_oversold or macd_bullish_cross or (macd_improving and macd > -50)
            has_funds = eur_balance > min_eur_for_order
            
            buy_signal = buy_trigger and safe_to_buy and has_funds
            
            # Build reason string
            if buy_signal:
                signals_active = []
                if rsi_oversold:
                    signals_active.append(f"RSI {rsi:.1f}")
                if macd_bullish_cross:
                    signals_active.append(f"MACD cross {macd:.1f}>{signal:.1f}")
                if macd_improving:
                    signals_active.append("MACD improving")
                buy_reason = f"✅ {', '.join(signals_active)}"
            elif not safe_to_buy:
                buy_reason = f"⚠️  Blocked: {', '.join(safety_blocks)}"
            elif not has_funds:
                buy_reason = f"⚠️  Insufficient funds (need €{min_eur_for_order:.2f})"
            elif not buy_trigger:
                reasons = []
                if not rsi_oversold:
                    reasons.append(f"RSI {rsi:.1f} > {self.buy_rsi_threshold}")
                if not macd_bullish_cross:
                    reasons.append("No MACD crossover")
                if not macd_improving:
                    reasons.append("MACD not improving")
                buy_reason = f"⏸️  Waiting: {', '.join(reasons)}"
            else:
                buy_reason = "⏸️  Waiting for signal"
            
            # === SELL LOGIC (peak detection for max EUR generation) ===
            sell_signal, sell_reason = self._detect_peak_reversal(
                current_price, macd, signal, rsi, prices
            )

            # Execute trades
            # === BUY LOGIC ===
            if buy_signal:
                logger.info(f"🟢 BUY SIGNAL - {buy_reason}")
                position_size = self.trade_executor.calculate_position_size(
                    eur_balance, current_price, risk_percent=25.0
                )
                
                if position_size > 0:
                    optimal_price = self.trade_executor.get_optimal_price(order_book, "buy")
                    if optimal_price:
                        # NEW: Use atomic trader
                        order_id = self.atomic_trader.execute_buy(
                            volume=position_size,
                            price=optimal_price,
                            market=self.market
                        )
                        
                        if order_id:
                            logger.info(f"✅ BUY ORDER PLACED | {position_size:.8f} BTC @ €{optimal_price:,.2f}")
                            logger.info(f"   ⏳ State will update after order fills...")

            # === SELL LOGIC ===
            elif sell_signal:
                sellable_btc = btc_balance - self.min_btc_reserve
                sell_amount = min(sellable_btc * 0.5, sellable_btc)
                
                logger.info(f"🔴 SELL SIGNAL - {sell_reason}")
                
                optimal_price = self.trade_executor.get_optimal_price(order_book, "sell")
                if optimal_price and sell_amount > self.trade_executor.min_order_size:
                    # NEW: Use atomic trader
                    order_id = self.atomic_trader.execute_sell(
                        volume=sell_amount,
                        price=optimal_price,
                        market=self.market
                    )
                    
                    if order_id:
                        logger.info(f"✅ SELL ORDER PLACED | {sell_amount:.8f} BTC @ €{optimal_price:,.2f}")
                        logger.info(f"   ⏳ State will update after order fills...")


            else:
                # Not buying or selling - show appropriate message
                logger.info(buy_reason)

            # Update performance
            self.performance_tracker.update_equity(btc_balance, eur_balance, current_price)

            # Check orders
            results = self.order_manager.check_and_update_orders()
            if results["filled"]:
                logger.info(f"✅ Filled orders: {len(results['filled'])}")

            logger.info("=" * 80 + "\n")

        except Exception as e:
            logger.error(f"Strategy execution error: {e}", exc_info=True)

    def cleanup(self):
        """Clean up resources"""
        logger.info("🧹 Cleaning up...")
        try:
            self._save_price_history()
            self.state_manager.save_state()
            self.state_manager.print_statistics()
            self.bitvavo_api.close()
            logger.info("✅ Bot shutdown complete")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    def stop(self):
        """Stop the bot"""
        self.running = False
        logger.info("🛑 Stop signal received")
        exit(0)


def main():
    """Main entry point"""
    bot = BitcoinAccumulationBot()
    
    def signal_handler(signum, frame):
        logger.info(f"Signal {signum} received")
        bot.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        bot.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()