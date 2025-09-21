import time
import json
import os
import logging
import signal
import sys
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

from trading.strategies import StrategyFactory, StrategyType, StrategyConfig, AdvancedStrategyEngine


logger = logging.getLogger(__name__)

# Configuration and Data Classes
@dataclass
class BotConfiguration:
    enable_ml: bool = True
    enable_peak_detection: bool = True
    enable_onchain_analysis: bool = True
    enable_news_sentiment: bool = True
    max_daily_trades: int = 12
    base_position_size_pct: float = 0.08
    stop_loss_pct: float = 0.025
    take_profit_pct: float = 0.08
    min_confidence_threshold: float = 0.35  # Increased from 0.25

class TradingAction(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class TradingSignal:
    action: TradingAction
    confidence: float
    volume: float
    price: float
    reasoning: List[str]
    risk_level: RiskLevel = RiskLevel.MEDIUM

@dataclass
class MarketIndicators:
    current_price: float
    rsi: float
    macd: float
    signal: float
    bollinger_upper: float
    bollinger_lower: float
    vwap: float
    sentiment: float
    volatility: float
    risk_off_probability: float = 0.0
    current_volume: float = 1000.0
    ml_success_probability: float = 0.5
    peak_probability: float = 0.0
    peak_recommendation: str = "normal_entry"
    market_regime: str = "normal"

# Import existing components with fallbacks
try:
    from core.indicators import (
        calculate_rsi, calculate_macd, calculate_bollinger_bands,
        calculate_moving_average, calculate_vwap, fetch_enhanced_news,
        calculate_enhanced_sentiment, calculate_risk_adjusted_indicators
    )
    INDICATORS_AVAILABLE = True
except ImportError:
    logger.warning("Core indicators not available, using simplified versions")
    INDICATORS_AVAILABLE = False

try:
    from trading.strategies import StrategyFactory, StrategyType, StrategyConfig
    STRATEGIES_AVAILABLE = True
except ImportError:
    logger.warning("Advanced strategies not available")
    STRATEGIES_AVAILABLE = False

try:
    from analysis.ml_engine import MLEngine, MLConfig, ModelType
    ML_AVAILABLE = True
except ImportError:
    logger.warning("ML engine not available")
    ML_AVAILABLE = False

try:
    from performance_tracker import PerformanceTracker
    PERFORMANCE_TRACKER_AVAILABLE = True
except ImportError:
    logger.warning("Performance tracker not available")
    PERFORMANCE_TRACKER_AVAILABLE = False

# Core Trading Components
class DataManager:
    def __init__(self, price_history_file="./price_history.json", bot_logs_file="./bot_logs.csv"):
        self.price_history_file = price_history_file
        self.bot_logs_file = bot_logs_file
        self._ensure_files_exist()

    def _ensure_files_exist(self):
        if not os.path.exists(self.price_history_file):
            with open(self.price_history_file, 'w') as f:
                json.dump([], f)
        
        if not os.path.exists(self.bot_logs_file):
            with open(self.bot_logs_file, 'w') as f:
                f.write("timestamp,price,rsi,sentiment,buy_decision,sell_decision,side,trade_volume\n")

    def load_price_history(self) -> Tuple[List[float], List[float]]:
        try:
            with open(self.price_history_file, 'r') as f:
                data = json.load(f)
            
            if not data:
                return [], []
            
            prices = [float(candle[4]) for candle in data]  # Close prices
            volumes = [float(candle[5]) for candle in data]  # Volumes
            
            return prices, volumes
        except Exception as e:
            logger.error(f"Failed to load price history: {e}")
            return [], []

    def append_ohlc_data(self, ohlc_data: List[List]) -> int:
        try:
            existing_data = []
            if os.path.exists(self.price_history_file):
                with open(self.price_history_file, 'r') as f:
                    existing_data = json.load(f)
            
            existing_timestamps = {int(candle[0]) for candle in existing_data}
            new_candles = [candle for candle in ohlc_data if int(candle[0]) not in existing_timestamps]
            
            if new_candles:
                existing_data.extend(new_candles)
                existing_data.sort(key=lambda x: x[0])
                
                # Keep only last 2000 candles
                existing_data = existing_data[-2000:]
                
                with open(self.price_history_file, 'w') as f:
                    json.dump(existing_data, f)
                
                logger.info(f"Added {len(new_candles)} new candles to price history")
                return len(new_candles)
            
            return 0
        except Exception as e:
            logger.error(f"Failed to append OHLC data: {e}")
            return 0

class TradeExecutor:
    def __init__(self, exchange):
        self.exchange = exchange
        self.pair = "BTC/EUR"  # CCXT format

    def fetch_current_price(self) -> Tuple[Optional[float], Optional[float]]:
        try:
            ticker = self.exchange.fetch_ticker(self.pair)
            return ticker['last'], ticker['quoteVolume']
        except Exception as e:
            logger.error(f"Failed to fetch current price: {e}")
            return None, None

    def get_total_btc_balance(self) -> Optional[float]:
        try:
            balance = self.exchange.fetch_balance()
            return float(balance.get('BTC', {}).get('free', 0.0))
        except Exception as e:
            logger.error(f"Failed to get BTC balance: {e}")
            return 0.0

    def get_available_balance(self, asset: str) -> Optional[float]:
        try:
            balance = self.exchange.fetch_balance()
            return float(balance.get(asset, {}).get('free', 0.0))
        except Exception as e:
            logger.error(f"Failed to get balance for {asset}: {e}")
            return 0.0

    def get_ohlc_data(self, pair="BTC/EUR", interval=15, since=None):
        try:
            timeframe = f"{interval}m"
            ohlcv = self.exchange.fetch_ohlcv(pair, timeframe, since)
            return [[
                float(candle[0]),  # timestamp
                float(candle[1]),  # open
                float(candle[2]),  # high
                float(candle[3]),  # low
                float(candle[4]),  # close
                float(candle[5])   # volume
            ] for candle in ohlcv]
        except Exception as e:
            logger.error(f"Failed to get OHLC data: {e}")
            return []

    def get_btc_order_book(self):
        try:
            return self.exchange.fetch_order_book(self.pair)
        except Exception as e:
            logger.error(f"Failed to get order book: {e}")
            return None

    def execute_trade(self, volume: float, side: str, price: float) -> bool:
        """Execute trade with operatorId"""
        try:
            order = self.exchange.create_order(
                symbol=self.pair,
                type='limit',
                side=side.lower(),
                amount=volume,
                price=price,
                params={
                    'operatorId': int(time.time() * 1000)  # Use timestamp as operatorId
                }
            )
            logger.info(f"Trade executed: {side} {volume} BTC at €{price}")
            return True
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return False

    def get_optimal_price(self, order_book: dict, side: str, buffer: float = 0.05) -> Optional[float]:
        """Calculate optimal price from order book"""
        try:
            if not order_book or 'asks' not in order_book or 'bids' not in order_book:
                return None

            if side == "buy":
                best_ask = float(order_book['asks'][0][0])
                return best_ask + buffer  # Add small buffer for buy orders
            elif side == "sell":
                best_bid = float(order_book['bids'][0][0])
                return best_bid - buffer  # Subtract small buffer for sell orders
            return None
            
        except Exception as e:
            logger.error(f"Failed to calculate optimal price: {e}")
            return None

class OrderManager:
    def __init__(self, exchange):
        self.exchange = exchange
        self.pending_orders = {}

    def get_pending_orders(self):
        try:
            open_orders = self.exchange.fetch_open_orders("BTC/EUR")
            return [order['id'] for order in open_orders]
        except Exception as e:
            logger.error(f"Failed to get pending orders: {e}")
            return []

    def get_order_statistics(self):
        return {"fill_rate": 0.85, "avg_fill_time": 120}

    def cleanup_old_orders(self, days=30):
        pass

    def force_refresh_all_orders(self):
        pass

# Technical Analysis Functions - Simplified versions as fallback
def calculate_rsi_simple(prices: List[float], period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd_simple(prices: List[float], fast=12, slow=26, signal_period=9) -> Tuple[float, float]:
    if len(prices) < slow:
        return 0.0, 0.0
    
    prices_array = np.array(prices)
    
    # Calculate EMAs (simplified)
    ema_fast = prices_array[-1]
    ema_slow = np.mean(prices_array[-slow:])
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line  # Simplified
    
    return macd_line, signal_line

def calculate_bollinger_bands_simple(prices: List[float], period=20, std_dev=2) -> Tuple[float, float, float]:
    if len(prices) < period:
        current_price = prices[-1] if prices else 0
        return current_price * 1.02, current_price, current_price * 0.98
    
    prices_array = np.array(prices[-period:])
    sma = np.mean(prices_array)
    std = np.std(prices_array)
    
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    
    return upper_band, sma, lower_band

def calculate_vwap_simple(prices: List[float], volumes: List[float]) -> float:
    if not prices or not volumes or len(prices) != len(volumes):
        return prices[-1] if prices else 0
    
    prices_array = np.array(prices[-50:])
    volumes_array = np.array(volumes[-50:])
    
    return np.sum(prices_array * volumes_array) / np.sum(volumes_array)

# Portfolio Analysis Functions
class PortfolioAnalyzer:
    def __init__(self):
        self.trade_history = []
    
    def analyze_position(self, btc_balance: float, eur_balance: float, current_price: float) -> Dict[str, Any]:
        """Analyze current portfolio position and suggest rebalancing"""
        total_value = eur_balance + (btc_balance * current_price)
        btc_value = btc_balance * current_price
        
        btc_percentage = (btc_value / total_value) if total_value > 0 else 0
        eur_percentage = (eur_balance / total_value) if total_value > 0 else 0
        
        # Determine if we're overweight in BTC
        target_btc_percentage = 0.7  # Target 70% BTC allocation
        
        analysis = {
            "total_value": total_value,
            "btc_percentage": btc_percentage,
            "eur_percentage": eur_percentage,
            "btc_value": btc_value,
            "overweight_btc": btc_percentage > target_btc_percentage + 0.1,
            "underweight_btc": btc_percentage < target_btc_percentage - 0.1,
            "should_rebalance": abs(btc_percentage - target_btc_percentage) > 0.1
        }
        
        return analysis
    
    def calculate_unrealized_pnl(self, btc_balance: float, current_price: float) -> float:
        """Calculate unrealized P&L based on average buy price"""
        if not self.trade_history:
            return 0.0
        
        # Calculate weighted average buy price
        total_bought = 0
        total_spent = 0
        
        for trade in self.trade_history:
            if trade["action"] == "buy":
                total_bought += trade["volume"]
                total_spent += trade["volume"] * trade["price"]
        
        if total_bought == 0:
            return 0.0
        
        avg_buy_price = total_spent / total_bought
        current_value = btc_balance * current_price
        cost_basis = btc_balance * avg_buy_price
        
        return current_value - cost_basis

# Core Bot Class with Balanced Trading
class TradingBot:
    def __init__(self, data_manager, trade_executor, order_manager, config):
        self.data_manager = data_manager
        self.trade_executor = trade_executor
        self.order_manager = order_manager
        self.config = config
        self.price_history = []
        self.volume_history = []
        self.daily_trades = 0
        self.last_trade_time = 0
        self.last_reset_date = datetime.now().date()
        self.portfolio_analyzer = PortfolioAnalyzer()

    def analyze_market(self) -> MarketIndicators:
        # Get current price and volume
        current_price, current_volume = self.trade_executor.fetch_current_price()
        if not current_price:
            raise ValueError("Could not fetch current price")

        # Load price history
        prices, volumes = self.data_manager.load_price_history()
        
        if len(prices) < 50:
            # Try to fetch fresh data
            recent_data = self.trade_executor.get_ohlc_data(
                pair="XXBTZEUR",
                interval="15m",
                since=int(time.time()) - (7 * 24 * 3600)
            )
            
            if recent_data:
                self.data_manager.append_ohlc_data(recent_data)
                prices, volumes = self.data_manager.load_price_history()
            
            if len(prices) < 20:
                raise ValueError("Insufficient price history")

        # Use advanced indicators if available, otherwise fallback to simple ones
        if INDICATORS_AVAILABLE:
            try:
                rsi = calculate_rsi(prices)
                macd, signal = calculate_macd(prices)
                upper_band, middle_band, lower_band = calculate_bollinger_bands(prices)
                vwap = calculate_vwap(prices, volumes)
            except:
                # Fallback to simple versions
                rsi = calculate_rsi_simple(prices)
                macd, signal = calculate_macd_simple(prices)
                upper_band, middle_band, lower_band = calculate_bollinger_bands_simple(prices)
                vwap = calculate_vwap_simple(prices, volumes)
        else:
            rsi = calculate_rsi_simple(prices)
            macd, signal = calculate_macd_simple(prices)
            upper_band, middle_band, lower_band = calculate_bollinger_bands_simple(prices)
            vwap = calculate_vwap_simple(prices, volumes)
        
        # Calculate volatility
        if len(prices) >= 20:
            returns = np.diff(np.log(prices[-20:]))
            volatility = np.std(returns) * np.sqrt(96)  # Annualized for 15-min data
        else:
            volatility = 0.02

        # Simple sentiment (could be enhanced with news analysis)
        sentiment = (rsi - 50) / 50  # RSI-based sentiment

        return MarketIndicators(
            current_price=current_price,
            rsi=rsi,
            macd=macd,
            signal=signal,
            bollinger_upper=upper_band,
            bollinger_lower=lower_band,
            vwap=vwap,
            sentiment=sentiment,
            volatility=volatility,
            current_volume=current_volume or 1000.0
        )

    def generate_signal(self, indicators: MarketIndicators) -> TradingSignal:
        """Generate balanced trading signal"""
        # Get portfolio analysis
        btc_balance = self.trade_executor.get_total_btc_balance() or 0
        eur_balance = self.trade_executor.get_available_balance("EUR") or 0
        
        portfolio_analysis = self.portfolio_analyzer.analyze_position(
            btc_balance, eur_balance, indicators.current_price
        )
        
        signals = []
        reasoning = []

        # Technical signals
        buy_signals = 0
        sell_signals = 0
        
        # RSI signals
        if indicators.rsi < 30:
            buy_signals += 2  # Strong oversold
            reasoning.append("RSI oversold")
        elif indicators.rsi < 40:
            buy_signals += 1  # Mild oversold
            reasoning.append("RSI mildly oversold")
        elif indicators.rsi > 70:
            sell_signals += 2  # Strong overbought
            reasoning.append("RSI overbought")
        elif indicators.rsi > 60:
            sell_signals += 1  # Mild overbought
            reasoning.append("RSI mildly overbought")

        # MACD signals
        if indicators.macd > indicators.signal:
            if indicators.macd > 0:
                buy_signals += 1
                reasoning.append("MACD bullish crossover")
            else:
                buy_signals += 0.5  # Weaker signal when MACD is negative
                reasoning.append("MACD improving but negative")
        else:
            if indicators.macd < 0:
                sell_signals += 1
                reasoning.append("MACD bearish crossover")
            else:
                sell_signals += 0.5
                reasoning.append("MACD weakening but positive")

        # Bollinger Band signals
        bb_position = (indicators.current_price - indicators.bollinger_lower) / (indicators.bollinger_upper - indicators.bollinger_lower)
        
        if bb_position < 0.2:  # Near lower band
            buy_signals += 1
            reasoning.append("Near lower Bollinger band")
        elif bb_position > 0.8:  # Near upper band
            sell_signals += 1
            reasoning.append("Near upper Bollinger band")

        # Price vs VWAP
        vwap_deviation = (indicators.current_price - indicators.vwap) / indicators.vwap
        if vwap_deviation < -0.01:  # 1% below VWAP
            buy_signals += 0.5
            reasoning.append("Below VWAP")
        elif vwap_deviation > 0.01:  # 1% above VWAP
            sell_signals += 0.5
            reasoning.append("Above VWAP")

        # Portfolio-based signals (CRITICAL FOR BALANCE)
        if portfolio_analysis["overweight_btc"]:
            sell_signals += 2  # Strong signal to rebalance
            reasoning.append(f"Portfolio rebalancing: {portfolio_analysis['btc_percentage']:.1%} BTC allocation")
        elif portfolio_analysis["underweight_btc"]:
            buy_signals += 1
            reasoning.append(f"Portfolio rebalancing: {portfolio_analysis['btc_percentage']:.1%} BTC allocation")

        # Profit-taking logic
        unrealized_pnl = self.portfolio_analyzer.calculate_unrealized_pnl(btc_balance, indicators.current_price)
        if unrealized_pnl > 0:
            pnl_percentage = unrealized_pnl / (btc_balance * indicators.current_price)
            if pnl_percentage > 0.05:  # 5% profit
                sell_signals += 1
                reasoning.append(f"Profit taking: {pnl_percentage:.1%} unrealized gain")

        # Risk-based adjustments
        if indicators.volatility > 0.08:  # High volatility
            buy_signals *= 0.7  # Reduce buying in high volatility
            sell_signals *= 1.2  # Increase selling tendency
            reasoning.append("High volatility adjustment")

        # Determine action based on signal strength
        total_buy = buy_signals
        total_sell = sell_signals
        
        logger.info(f"Signal analysis: BUY={total_buy:.2f}, SELL={total_sell:.2f}")
        
        # More conservative thresholds
        if total_buy > total_sell + 1.5 and total_buy > 2.0:  # Need strong buy consensus
            action = TradingAction.BUY
            confidence = min(0.9, (total_buy - total_sell) / 5.0 + 0.5)
        elif total_sell > total_buy + 1.0 and total_sell > 1.5:  # Easier to sell for rebalancing
            action = TradingAction.SELL
            confidence = min(0.9, (total_sell - total_buy) / 4.0 + 0.5)
        else:
            action = TradingAction.HOLD
            confidence = 0.5
            reasoning = [f"Insufficient signal strength: BUY={total_buy:.2f}, SELL={total_sell:.2f}"]

        # Calculate position size
        volume = self._calculate_position_size(action, indicators, confidence, portfolio_analysis)

        return TradingSignal(
            action=action,
            confidence=confidence,
            volume=volume,
            price=indicators.current_price,
            reasoning=reasoning[:3],
            risk_level=self._assess_risk_level(indicators)
        )

    def _calculate_position_size(self, action: TradingAction, indicators: MarketIndicators, confidence: float, portfolio_analysis: Dict) -> float:
        if action == TradingAction.HOLD:
            return 0.0

        base_size = self.config.base_position_size_pct
        
        # Adjust for confidence
        confidence_multiplier = 0.3 + (confidence * 0.7)  # 0.3 to 1.0 range
        
        # Adjust for volatility
        vol_multiplier = max(0.3, 1.0 - indicators.volatility * 8)
        
        # Portfolio-based adjustments
        portfolio_multiplier = 1.0
        if action == TradingAction.SELL and portfolio_analysis["overweight_btc"]:
            # Larger sells when rebalancing
            portfolio_multiplier = 1.5
        elif action == TradingAction.BUY and portfolio_analysis["overweight_btc"]:
            # Smaller buys when already overweight
            portfolio_multiplier = 0.3
        
        final_size_pct = base_size * confidence_multiplier * vol_multiplier * portfolio_multiplier
        
        if action == TradingAction.BUY:
            eur_balance = self.trade_executor.get_available_balance("EUR") or 0
            position_value = eur_balance * final_size_pct
            volume = position_value / indicators.current_price
        else:
            btc_balance = self.trade_executor.get_total_btc_balance() or 0
            volume = btc_balance * final_size_pct

        return max(0.0001, min(volume, 0.01))  # Min 0.0001, max 0.01 BTC

    def _assess_risk_level(self, indicators: MarketIndicators) -> RiskLevel:
        risk_score = 0
        
        if indicators.volatility > 0.08:
            risk_score += 3
        elif indicators.volatility > 0.05:
            risk_score += 2
        elif indicators.volatility > 0.03:
            risk_score += 1

        if indicators.sentiment < -0.3:
            risk_score += 2
        elif indicators.sentiment < -0.1:
            risk_score += 1

        if risk_score >= 4:
            return RiskLevel.HIGH
        elif risk_score >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _can_trade(self) -> bool:
        # Reset daily counter if new day
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_trades = 0
            self.last_reset_date = current_date

        # Check daily limit
        if self.daily_trades >= self.config.max_daily_trades:
            logger.info(f"Daily trade limit reached: {self.daily_trades}")
            return False

        # Check time since last trade (prevent overtrading)
        if time.time() - self.last_trade_time < 900:  # 15 minutes
            logger.info("Too soon since last trade")
            return False

        return True

    def _execute_trade(self, signal: TradingSignal) -> bool:
        try:
            # Get order book for optimal pricing
            order_book = self.trade_executor.get_btc_order_book()
            if not order_book:
                logger.error("Could not get order book")
                return False

            optimal_price = self.trade_executor.get_optimal_price(order_book, signal.action.value)
            if not optimal_price:
                logger.error("Could not determine optimal price")
                return False

            # Execute trade
            success = self.trade_executor.execute_trade(
                volume=signal.volume,
                side=signal.action.value,
                price=optimal_price
            )

            if success:
                self.daily_trades += 1
                self.last_trade_time = time.time()
                self._log_trade(signal, optimal_price)
                
                # Add to portfolio analyzer history
                self.portfolio_analyzer.trade_history.append({
                    "timestamp": datetime.now(),
                    "action": signal.action.value,
                    "volume": signal.volume,
                    "price": optimal_price
                })

            return success

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return False

    def _log_trade(self, signal: TradingSignal, price: float):
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "price": price,
                "rsi": 50,  # Would get from current indicators
                "sentiment": 0,
                "buy_decision": signal.action == TradingAction.BUY,
                "sell_decision": signal.action == TradingAction.SELL,
                "side": signal.action.value,
                "trade_volume": signal.volume
            }

            # Append to CSV (simplified)
            with open(self.data_manager.bot_logs_file, 'a') as f:
                f.write(f"{log_entry['timestamp']},{price},{log_entry['rsi']},{log_entry['sentiment']},{log_entry['buy_decision']},{log_entry['sell_decision']},{log_entry['side']},{signal.volume}\n")

        except Exception as e:
            logger.error(f"Failed to log trade: {e}")

    def _update_pending_orders(self):
        # Check and update pending orders
        try:
            pending = self.order_manager.get_pending_orders()
            logger.debug(f"Pending orders: {len(pending)}")
        except Exception as e:
            logger.warning(f"Failed to update pending orders: {e}")

    def get_status(self) -> Dict[str, Any]:
        try:
            current_price, _ = self.trade_executor.fetch_current_price()
            btc_balance = self.trade_executor.get_total_btc_balance() or 0
            eur_balance = self.trade_executor.get_available_balance("EUR") or 0
            
            return {
                "balances": {
                    "btc": btc_balance,
                    "eur": eur_balance,
                    "total_value_eur": eur_balance + (btc_balance * current_price) if current_price else eur_balance
                },
                "market": {
                    "current_price": current_price,
                    "daily_trades": self.daily_trades,
                    "max_daily_trades": self.config.max_daily_trades
                },
                "trading": {
                    "can_trade": self._can_trade(),
                    "last_trade_time": self.last_trade_time,
                    "time_since_last_trade": time.time() - self.last_trade_time
                }
            }
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return {"error": str(e)}

    def shutdown(self):
        logger.info("Core bot shutting down...")


# Enhanced Bot with Additional Features
class EnhancedTradingBot:
    def __init__(self, bitvavo_api, config: Optional[BotConfiguration] = None):
        self.bitvavo_api = bitvavo_api
        self.config = config or BotConfiguration()
        
        # Initialize core components
        self.data_manager = DataManager()
        self.trade_executor = TradeExecutor(bitvavo_api)
        self.order_manager = OrderManager(bitvavo_api)
        self.core_bot = TradingBot(
            self.data_manager,
            self.trade_executor,
            self.order_manager,
            self.config
        )
        
        # Enhanced features
        self.available_features = {
            'risk_management': True,
            'enhanced_analysis': True,
            'performance_tracking': PERFORMANCE_TRACKER_AVAILABLE,
            'adaptive_sizing': True,
            'portfolio_rebalancing': True
        }
        
        # Initialize performance tracker if available
        if PERFORMANCE_TRACKER_AVAILABLE:
            try:
                btc_balance = self.trade_executor.get_total_btc_balance() or 0
                eur_balance = self.trade_executor.get_available_balance("EUR") or 0
                self.performance_tracker = PerformanceTracker(btc_balance, eur_balance)
                logger.info("Performance tracker initialized")
            except Exception as e:
                logger.error(f"Failed to initialize performance tracker: {e}")
                self.performance_tracker = None
        else:
            self.performance_tracker = None
        
        # Initialize ML engine if available
        if ML_AVAILABLE:
            try:
                ml_config = MLConfig(
                    model_type=ModelType.ENSEMBLE,
                    prediction_task="trade_success",
                    use_technical_features=True,
                    use_market_features=True,
                    use_sentiment_features=True,
                    online_learning_enabled=True,
                )
                self.ml_engine = MLEngine(ml_config)
                
                # Try to load existing model
                if os.path.exists("./ml_model.pkl"):
                    self.ml_engine.load_model("./ml_model.pkl")
                    logger.info("Loaded existing ML model")
                
                logger.info("ML engine initialized")
            except Exception as e:
                logger.error(f"Failed to initialize ML engine: {e}")
                self.ml_engine = None
        else:
            self.ml_engine = None
        
        # Initialize strategies if available
        if STRATEGIES_AVAILABLE:
            try:
                self.advanced_strategy_engine = AdvancedStrategyEngine()
                self.strategies = {}
                
                # Initialize strategy configs
                strategy_configs = {
                    "dca": StrategyConfig(
                        name="DCA Strategy",
                        enabled=True,
                        params={"buy_interval_hours": 48, "buy_amount_percentage": 0.03}  # Reduced frequency and amount
                    ),
                    "momentum": StrategyConfig(
                        name="Momentum Strategy", 
                        enabled=True,
                        params={"momentum_period": 20, "momentum_threshold": 0.03}  # Higher threshold
                    ),
                    "mean_reversion": StrategyConfig(
                        name="Mean Reversion Strategy",
                        enabled=True,
                        params={"lookback_period": 50, "deviation_threshold": 2.5}  # Higher threshold
                    )
                }
                
                for name, config in strategy_configs.items():
                    try:
                        strategy_type = getattr(StrategyType, name.upper())
                        self.strategies[name] = StrategyFactory.create_strategy(strategy_type, config)
                        logger.info(f"Initialized {name} strategy")
                    except Exception as e:
                        logger.warning(f"Failed to initialize {name} strategy: {e}")
                
                logger.info(f"Initialized {len(self.strategies)} advanced strategies")
            except Exception as e:
                logger.error(f"Failed to initialize strategies: {e}")
                self.strategies = {}
                self.advanced_strategy_engine = None
        else:
            self.strategies = {}
            self.advanced_strategy_engine = None
        
        # Simple risk manager
        self.simple_risk_manager = SimpleRiskManager(
            max_daily_trades=self.config.max_daily_trades,
            max_position_pct=0.12  # Reduced from 0.15
        )
        
        # Performance tracking
        self.session_start = datetime.now()
        self.total_trades = 0
        self.enhanced_decisions = 0
        self.trade_history = []
        
        # More conservative thresholds for balanced trading
        self.consensus_thresholds = {
            'buy_threshold': 0.45,   # Increased from 0.24
            'sell_threshold': 0.35,  # Increased from 0.24 but lower than buy
            'confidence_boost': 0.25  # Reduced from 0.36
        }
        
        logger.info("Enhanced Trading Bot initialized with balanced parameters")

    def enhanced_market_analysis(self) -> MarketIndicators:
        """Enhanced market analysis with additional features"""
        # Get base analysis
        indicators = self.core_bot.analyze_market()
        
        # Add enhanced features
        try:
            # Enhanced volatility analysis
            prices, volumes = self.data_manager.load_price_history()
            if len(prices) > 20:
                recent_prices = prices[-20:]
                returns = np.diff(np.log(recent_prices))
                enhanced_volatility = np.std(returns) * np.sqrt(96)
                
                if enhanced_volatility > 0.06:  # Adjusted thresholds
                    indicators.volatility_regime = "high"
                elif enhanced_volatility < 0.025:
                    indicators.volatility_regime = "low"
                else:
                    indicators.volatility_regime = "normal"
            
            # Enhanced RSI analysis with more conservative signals
            if indicators.rsi < 25:
                indicators.rsi_signal = "strong_oversold"
                indicators.rsi_strength = min(1.0, (30 - indicators.rsi) / 5)
            elif indicators.rsi < 35:
                indicators.rsi_signal = "oversold"
                indicators.rsi_strength = min(0.7, (35 - indicators.rsi) / 10)
            elif indicators.rsi > 75:
                indicators.rsi_signal = "strong_overbought"
                indicators.rsi_strength = min(1.0, (indicators.rsi - 70) / 5)
            elif indicators.rsi > 65:
                indicators.rsi_signal = "overbought"
                indicators.rsi_strength = min(0.7, (indicators.rsi - 65) / 10)
            else:
                indicators.rsi_signal = "neutral"
                indicators.rsi_strength = 0.0
            
            # Volume analysis
            if len(volumes) > 10:
                recent_volume = volumes[-10:]
                avg_volume = np.mean(recent_volume)
                current_volume = indicators.current_volume
                indicators.volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                
                if indicators.volume_ratio > 2.5:  # Higher threshold
                    indicators.volume_signal = "high"
                elif indicators.volume_ratio < 0.4:  # Lower threshold
                    indicators.volume_signal = "low"
                else:
                    indicators.volume_signal = "normal"
            
            # ML predictions if available
            if self.ml_engine and self.ml_engine.is_trained:
                try:
                    market_data = {
                        "rsi": indicators.rsi,
                        "macd": indicators.macd,
                        "signal": indicators.signal,
                        "current_price": indicators.current_price,
                        "upper_band": indicators.bollinger_upper,
                        "lower_band": indicators.bollinger_lower,
                        "vwap": indicators.vwap,
                        "sentiment": indicators.sentiment,
                        "volatility": indicators.volatility,
                        "timestamp": datetime.now(),
                    }
                    
                    prediction, confidence = self.ml_engine.predict(market_data)
                    indicators.ml_success_probability = confidence
                    
                    logger.debug(f"ML prediction: {confidence:.3f} confidence")
                    
                except Exception as e:
                    logger.warning(f"ML prediction failed: {e}")
            
        except Exception as e:
            logger.warning(f"Enhanced analysis error: {e}")
        
        return indicators

    def generate_enhanced_signal(self, indicators: MarketIndicators) -> TradingSignal:
        """Generate enhanced trading signal with more conservative approach"""
        try:
            # Get base signal from core bot (which now includes portfolio analysis)
            base_signal = self.core_bot.generate_signal(indicators)
            
            # Start with base signal
            enhanced_confidence = base_signal.confidence
            enhanced_volume = base_signal.volume
            enhanced_reasoning = list(base_signal.reasoning) if base_signal.reasoning else []
            
            # Only apply modest enhancements to avoid over-confidence
            enhancement_applied = False
            
            # RSI enhancements (more conservative)
            rsi_signal = getattr(indicators, 'rsi_signal', 'neutral')
            rsi_strength = getattr(indicators, 'rsi_strength', 0.0)
            
            if base_signal.action == TradingAction.BUY and rsi_signal == 'strong_oversold':
                boost = min(0.15, 0.05 + (rsi_strength * 0.1))  # Max 0.15 boost
                enhanced_confidence = min(0.9, enhanced_confidence + boost)
                enhanced_reasoning.append(f"Strong RSI oversold: +{boost:.2f}")
                enhancement_applied = True
                
            elif base_signal.action == TradingAction.SELL and rsi_signal == 'strong_overbought':
                boost = min(0.15, 0.05 + (rsi_strength * 0.1))
                enhanced_confidence = min(0.9, enhanced_confidence + boost)
                enhanced_reasoning.append(f"Strong RSI overbought: +{boost:.2f}")
                enhancement_applied = True
            
            # Volume enhancements (only for very high volume)
            volume_signal = getattr(indicators, 'volume_signal', 'normal')
            volume_ratio = getattr(indicators, 'volume_ratio', 1.0)
            
            if volume_signal == "high" and volume_ratio > 3.0 and base_signal.action != TradingAction.HOLD:
                boost = min(0.1, 0.05 + (volume_ratio - 3.0) / 20)  # Small boost for extreme volume
                enhanced_confidence = min(0.9, enhanced_confidence + boost)
                enhanced_reasoning.append(f"Extreme volume confirmation ({volume_ratio:.1f}x)")
                enhancement_applied = True
            
            # ML enhancement (if available and confident)
            ml_prediction = getattr(indicators, 'ml_prediction', 0)
            ml_prob = getattr(indicators, 'ml_success_probability', 0.5)
            
            if self.ml_engine and self.ml_engine.is_trained and ml_prob > 0.65:
                if base_signal.action != TradingAction.HOLD:
                    # ML predicts success for the proposed trade
                    if ml_prediction == 1:  # Success predicted
                        boost = min(0.15, (ml_prob - 0.65) / 0.35 * 0.15)  # Scale boost
                        enhanced_confidence = min(0.9, enhanced_confidence + boost)
                        enhanced_reasoning.append(f"ML predicts success: {ml_prob:.2f}")
                        enhancement_applied = True
                    elif ml_prediction == 0 and ml_prob > 0.8:  # Strong failure prediction
                        # Reduce confidence for likely failed trades
                        enhanced_confidence *= 0.7
                        enhanced_reasoning.append(f"ML warns of failure: {ml_prob:.2f}")
                elif ml_prediction == 1 and ml_prob > 0.8:
                    # ML strongly suggests we should be trading but base signal says hold
                    # This could indicate an opportunity, but be conservative
                    enhanced_reasoning.append(f"ML suggests opportunity: {ml_prob:.2f}")
                    # Don't override HOLD, but note the ML suggestion
            
            # Strategy consensus (if available)
            if self.strategies:
                try:
                    strategy_signals = []
                    for name, strategy in self.strategies.items():
                        if hasattr(strategy, 'should_trade') and hasattr(strategy, 'generate_signal'):
                            should_trade, _ = strategy.should_trade(indicators)
                            if should_trade:
                                analysis = strategy.analyze(indicators) if hasattr(strategy, 'analyze') else {}
                                strategy_signal = strategy.generate_signal(indicators, analysis)
                                if strategy_signal.action == base_signal.action:
                                    strategy_signals.append(name)
                    
                    if len(strategy_signals) >= 2:  # Multiple strategies agree
                        boost = min(0.1, len(strategy_signals) * 0.03)
                        enhanced_confidence = min(0.9, enhanced_confidence + boost)
                        enhanced_reasoning.append(f"Strategy consensus: {', '.join(strategy_signals)}")
                        enhancement_applied = True
                        
                except Exception as e:
                    logger.warning(f"Strategy consensus failed: {e}")
            
            # Volatility regime adjustment (reduce position size in high vol)
            volatility_regime = getattr(indicators, 'volatility_regime', 'normal')
            if volatility_regime == "high":
                enhanced_volume *= 0.6  # Significant reduction
                enhanced_reasoning.append("Position reduced for high volatility")
            elif volatility_regime == "low":
                enhanced_volume *= 1.1  # Small increase
                enhanced_reasoning.append("Position slightly increased for low volatility")
            
            # Track enhancements
            if enhancement_applied and enhanced_confidence > base_signal.confidence + 0.03:
                self.enhanced_decisions += 1
                logger.info(f"Signal enhanced: {base_signal.confidence:.2f} -> {enhanced_confidence:.2f}")
            
            return TradingSignal(
                action=base_signal.action,
                confidence=enhanced_confidence,
                volume=enhanced_volume,
                price=base_signal.price,
                reasoning=enhanced_reasoning,
                risk_level=base_signal.risk_level
            )
            
        except Exception as e:
            logger.error(f"Enhanced signal generation failed: {e}")
            return base_signal

    def execute_enhanced_strategy(self):
        """Execute enhanced strategy with risk management"""
        try:
            logger.info("Executing enhanced strategy...")
            
            # Standard checks
            if not self.core_bot._can_trade():
                logger.info("Trading conditions not met")
                return
            
            # Get portfolio info
            btc_balance = self.trade_executor.get_total_btc_balance() or 0
            eur_balance = self.trade_executor.get_available_balance("EUR") or 0
            current_price, _ = self.trade_executor.fetch_current_price()
            portfolio_value = eur_balance + (btc_balance * current_price) if current_price else eur_balance
            
            # Enhanced market analysis
            indicators = self.enhanced_market_analysis()
            
            # Generate enhanced signal
            signal = self.generate_enhanced_signal(indicators)
            
            # Log signal details for debugging
            logger.info(f"Generated signal: {signal.action.value.upper()} (confidence: {signal.confidence:.2f})")
            logger.info(f"Signal reasoning: {', '.join(signal.reasoning[:2])}")
            
            # Risk assessment
            if signal.action != TradingAction.HOLD:
                risk_assessment = self.simple_risk_manager.assess_trade_risk(
                    signal.volume, signal.action.value, current_price, portfolio_value, signal.confidence
                )
                
                if not risk_assessment["approved"]:
                    logger.warning(f"Trade rejected by risk manager: {risk_assessment['warnings']}")
                    return
                
                # Adjust size based on risk assessment
                if risk_assessment["max_recommended_size"] < signal.volume:
                    logger.info(f"Position size adjusted: {signal.volume:.6f} -> {risk_assessment['max_recommended_size']:.6f}")
                    signal.volume = risk_assessment["max_recommended_size"]
                
                # Final volume check
                if signal.volume < 0.0001:
                    logger.info(f"Trade size too small ({signal.volume:.8f}), skipping")
                    return
                
                # Execute trade
                success = self.core_bot._execute_trade(signal)
                
                if success:
                    # Record trade
                    self.simple_risk_manager.record_trade_outcome(signal.volume, 0.0, signal.confidence)
                    self.total_trades += 1
                    
                    # Add to trade history
                    trade_record = {
                        "timestamp": datetime.now(),
                        "action": signal.action.value,
                        "price": current_price,
                        "volume": signal.volume,
                        "confidence": signal.confidence
                    }
                    self.trade_history.append(trade_record)
                    
                    # Update performance tracker
                    if self.performance_tracker:
                        try:
                            order_id = f"trade_{int(time.time())}"
                            fee = signal.price * signal.volume * 0.0025
                            
                            self.performance_tracker.record_trade(
                                order_id=order_id,
                                side=signal.action.value,
                                volume=signal.volume,
                                price=signal.price,
                                fee=fee,
                                timestamp=time.time()
                            )
                        except Exception as e:
                            logger.warning(f"Failed to record trade in performance tracker: {e}")
                    
                    logger.info(f"Enhanced trade executed: {signal.action.value.upper()}")
                else:
                    logger.warning("Trade execution failed")
            else:
                logger.info(f"Holding: {signal.reasoning[0] if signal.reasoning else 'Low confidence'}")
            
        except Exception as e:
            logger.error(f"Enhanced strategy execution error: {e}", exc_info=True)

    def get_enhanced_status(self) -> Dict[str, Any]:
        """Get enhanced status"""
        status = self.core_bot.get_status()
        
        # Add portfolio analysis
        if "balances" in status:
            btc_balance = status["balances"]["btc"]
            eur_balance = status["balances"]["eur"]
            current_price = status["market"]["current_price"]
            
            portfolio_analysis = self.core_bot.portfolio_analyzer.analyze_position(
                btc_balance, eur_balance, current_price
            )
            
            status["portfolio"] = {
                "btc_percentage": portfolio_analysis["btc_percentage"],
                "eur_percentage": portfolio_analysis["eur_percentage"],
                "overweight_btc": portfolio_analysis["overweight_btc"],
                "should_rebalance": portfolio_analysis["should_rebalance"],
                "unrealized_pnl": self.core_bot.portfolio_analyzer.calculate_unrealized_pnl(
                    btc_balance, current_price
                )
            }
        
        # Add enhanced metrics
        enhanced_status = {
            "enhanced_bot": {
                "session_start": self.session_start.isoformat(),
                "total_trades": self.total_trades,
                "enhanced_decisions": self.enhanced_decisions,
                "available_features": self.available_features,
                "trade_history_length": len(self.trade_history)
            },
            "risk_management": {
                "daily_trades": self.simple_risk_manager.daily_trades,
                "max_daily_trades": self.simple_risk_manager.max_daily_trades,
                "emergency_stop": self.simple_risk_manager.emergency_stop
            }
        }
        
        status.update({"enhanced": enhanced_status})
        return status

    def print_enhanced_status(self):
        """Print enhanced status"""
        try:
            status = self.get_enhanced_status()
            
            print(f"\n{'='*80}")
            print("Enhanced Bitcoin Trading Bot Status")
            print(f"{'='*80}")
            
            # Session info
            enhanced = status["enhanced"]["enhanced_bot"]
            uptime = (datetime.now() - datetime.fromisoformat(enhanced["session_start"])).total_seconds() / 3600
            print(f"Uptime: {uptime:.1f} hours")
            print(f"Total Trades: {enhanced['total_trades']}")
            print(f"Enhanced Decisions: {enhanced['enhanced_decisions']}")
            
            # Balances
            if "balances" in status:
                balances = status["balances"]
                print(f"BTC Balance: {balances['btc']:.8f}")
                print(f"EUR Balance: €{balances['eur']:.2f}")
                print(f"Total Value: €{balances['total_value_eur']:.2f}")
            
            # Market info
            if "market" in status:
                market = status["market"]
                print(f"Current BTC Price: €{market['current_price']:.2f}")
                print(f"Daily Trades: {market['daily_trades']}/{market['max_daily_trades']}")
            
            # Portfolio analysis
            if "portfolio" in status:
                portfolio = status["portfolio"]
                print(f"\nPortfolio Analysis:")
                print(f"   BTC Allocation: {portfolio['btc_percentage']:.1%}")
                print(f"   EUR Allocation: {portfolio['eur_percentage']:.1%}")
                print(f"   Rebalancing Needed: {'Yes' if portfolio['should_rebalance'] else 'No'}")
                print(f"   Unrealized P&L: €{portfolio['unrealized_pnl']:.2f}")
                if portfolio['overweight_btc']:
                    print(f"   Status: Overweight BTC - may sell to rebalance")
            
            # Enhanced features
            features = enhanced["available_features"]
            print(f"\nEnhanced Features:")
            for feature, available in features.items():
                print(f"   {feature}: {'✅' if available else '❌'}")
            
            # Risk management
            if "risk_management" in status["enhanced"]:
                risk = status["enhanced"]["risk_management"]
                print(f"\nRisk Management:")
                print(f"   Emergency Stop: {'🚨 ACTIVE' if risk['emergency_stop'] else '✅ Normal'}")
                print(f"   Daily Trade Count: {risk['daily_trades']}/{risk['max_daily_trades']}")
            
            print(f"{'='*80}\n")
            
        except Exception as e:
            logger.error(f"Status printing error: {e}")

    def save_state(self):
        """Save bot state"""
        try:
            state = {
                "session_start": self.session_start.isoformat(),
                "total_trades": self.total_trades,
                "enhanced_decisions": self.enhanced_decisions,
                "trade_history": [
                    {
                        "timestamp": trade["timestamp"].isoformat(),
                        "action": trade["action"],
                        "price": trade["price"],
                        "volume": trade["volume"],
                        "confidence": trade["confidence"]
                    }
                    for trade in self.trade_history[-100:]  # Keep last 100 trades
                ]
            }
            
            with open("./enhanced_bot_state.json", "w") as f:
                json.dump(state, f, indent=2)
            
            # Save ML model if available
            if self.ml_engine:
                self.ml_engine.save_model("./ml_model.pkl")
            
            logger.info("Enhanced bot state saved")
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def load_state(self):
        """Load bot state"""
        try:
            if os.path.exists("./enhanced_bot_state.json"):
                with open("./enhanced_bot_state.json", "r") as f:
                    state = json.load(f)
                
                self.session_start = datetime.fromisoformat(
                    state.get("session_start", datetime.now().isoformat())
                )
                self.total_trades = state.get("total_trades", 0)
                self.enhanced_decisions = state.get("enhanced_decisions", 0)
                
                # Load trade history
                trade_history_data = state.get("trade_history", [])
                self.trade_history = []
                
                for trade_data in trade_history_data:
                    try:
                        trade_data["timestamp"] = datetime.fromisoformat(trade_data["timestamp"])
                        self.trade_history.append(trade_data)
                        
                        # Add to portfolio analyzer history
                        self.core_bot.portfolio_analyzer.trade_history.append(trade_data)
                    except Exception as e:
                        logger.warning(f"Failed to load trade record: {e}")
                
                logger.info("Enhanced bot state loaded")
                
        except Exception as e:
            logger.error(f"Failed to load state: {e}")

    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down enhanced trading bot...")
        self.save_state()
        self.core_bot.shutdown()


class SimpleRiskManager:
    """Simple risk management system"""
    
    def __init__(self, max_daily_trades=10, max_position_pct=0.15):
        self.max_daily_trades = max_daily_trades
        self.max_position_pct = max_position_pct
        self.daily_trades = 0
        self.last_reset = datetime.now().date()
        self.emergency_stop = False
    
    def assess_trade_risk(self, trade_size, trade_direction, current_price, portfolio_value, confidence):
        """Simple risk assessment"""
        # Reset daily counter
        if datetime.now().date() > self.last_reset:
            self.daily_trades = 0
            self.last_reset = datetime.now().date()
        
        # Check daily limit
        if self.daily_trades >= self.max_daily_trades:
            return {
                "approved": False,
                "warnings": ["Daily trade limit exceeded"],
                "max_recommended_size": 0.0
            }
        
        # Check emergency stop
        if self.emergency_stop:
            return {
                "approved": False,
                "warnings": ["Emergency stop active"],
                "max_recommended_size": 0.0
            }
        
        # Check position size
        position_value = trade_size * current_price
        position_pct = position_value / portfolio_value if portfolio_value > 0 else 0
        
        max_size = trade_size
        if position_pct > self.max_position_pct:
            max_size = (self.max_position_pct * portfolio_value) / current_price
        
        return {
            "approved": True,
            "warnings": [],
            "max_recommended_size": max_size,
            "adjustments": [f"Size adjusted to {max_size:.6f}"] if max_size < trade_size else []
        }
    
    def record_trade_outcome(self, trade_size, pnl, confidence):
        """Record trade outcome"""
        self.daily_trades += 1