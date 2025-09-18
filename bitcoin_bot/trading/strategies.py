# trading/strategies.py
"""
Unified Trading Strategy Implementations
Combines institutional-grade strategies with modular base strategies
Integrates with enhanced indicators and market analysis
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque
import json
from scipy import stats
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Import shared components
try:
    from core.bot import TradingAction, RiskLevel, MarketIndicators, TradingSignal
except ImportError:
    # Fallback if imports fail
    from enum import Enum
    
    class TradingAction(Enum):
        BUY = "buy"
        SELL = "sell"
        HOLD = "hold"
    
    class RiskLevel(Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

# Import indicators
try:
    from core.indicators import (
        calculate_rsi, calculate_macd, calculate_bollinger_bands,
        calculate_moving_average, calculate_vwap, fetch_enhanced_news,
        calculate_enhanced_sentiment, calculate_risk_adjusted_indicators
    )
    INDICATORS_AVAILABLE = True
    logger.info("Core indicators module loaded successfully")
except ImportError:
    logger.warning("Core indicators not available, using simplified versions")
    INDICATORS_AVAILABLE = False


class StrategyType(Enum):
    """Available strategy types"""
    DCA = "dca"
    GRID = "grid"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    HYBRID = "hybrid"
    ML_ENHANCED = "ml_enhanced"
    STATISTICAL_ARBITRAGE = "stat_arb"
    VOLATILITY_BREAKOUT = "vol_breakout"
    ADAPTIVE_DCA = "adaptive_dca"
    MTF_MOMENTUM = "mtf_momentum"


class TimeFrame(Enum):
    """Trading timeframes"""
    TICK = "tick"
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


class SignalStrength(Enum):
    """Signal strength levels"""
    VERY_WEAK = 0.1
    WEAK = 0.3
    MODERATE = 0.5
    STRONG = 0.7
    VERY_STRONG = 0.9


@dataclass
class StrategyConfig:
    """Base strategy configuration"""
    name: str
    enabled: bool = True
    
    # Risk parameters
    max_position_size: float = 0.25
    stop_loss_enabled: bool = True
    stop_loss_percentage: float = 0.03
    take_profit_enabled: bool = True
    take_profit_percentage: float = 0.10
    
    # Position sizing
    base_order_size: float = 0.10
    position_scaling: bool = True
    
    # Timing
    min_time_between_orders: int = 300  # seconds
    
    # Strategy-specific parameters
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdvancedSignal:
    """Enhanced signal with multi-timeframe analysis"""
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    timeframe: TimeFrame
    strategy_name: str
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: float = 0.0
    reasoning: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyState:
    """Strategy execution state"""
    last_action: Optional[TradingAction] = None
    last_action_time: Optional[datetime] = None
    last_action_price: Optional[float] = None
    
    open_positions: List[Dict] = field(default_factory=list)
    position_count: int = 0
    total_invested: float = 0.0
    average_entry_price: float = 0.0
    
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    trade_history: List[Dict] = field(default_factory=list)
    
    # Strategy-specific state
    custom_state: Dict[str, Any] = field(default_factory=dict)


class MarketRegimeClassifier:
    """Advanced market regime classification"""
    
    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.price_history = deque(maxlen=lookback)
        self.volume_history = deque(maxlen=lookback)
        
    def update(self, price: float, volume: float):
        """Update with new market data"""
        self.price_history.append(price)
        self.volume_history.append(volume)
    
    def classify_regime(self) -> Dict[str, float]:
        """Classify current market regime with probabilities"""
        if len(self.price_history) < 50:
            return {"trending": 0.33, "mean_reverting": 0.33, "volatile": 0.34}
        
        prices = np.array(self.price_history)
        returns = np.diff(np.log(prices))
        
        # Hurst Exponent for trend persistence
        hurst = self._calculate_hurst_exponent(returns)
        
        # Volatility clustering
        vol_clustering = self._measure_volatility_clustering(returns)
        
        # Autocorrelation strength
        autocorr = self._calculate_autocorrelation(returns)
        
        # Regime probabilities
        trending_prob = max(0, min(1, (hurst - 0.5) * 2))
        mean_reverting_prob = max(0, min(1, (0.5 - hurst) * 2))
        volatile_prob = vol_clustering
        
        # Normalize
        total = trending_prob + mean_reverting_prob + volatile_prob
        if total > 0:
            return {
                "trending": trending_prob / total,
                "mean_reverting": mean_reverting_prob / total,
                "volatile": volatile_prob / total
            }
        
        return {"trending": 0.33, "mean_reverting": 0.33, "volatile": 0.34}
    
    def _calculate_hurst_exponent(self, returns: np.ndarray, max_lag: int = 20) -> float:
        """Calculate Hurst exponent for trend persistence"""
        try:
            lags = range(2, max_lag)
            rs_values = []
            
            for lag in lags:
                if len(returns) < lag * 2:
                    continue
                    
                chunks = len(returns) // lag
                rs_chunk = []
                
                for i in range(chunks):
                    chunk = returns[i*lag:(i+1)*lag]
                    if len(chunk) < lag:
                        continue
                    
                    mean_chunk = np.mean(chunk)
                    cumsum = np.cumsum(chunk - mean_chunk)
                    
                    r = np.max(cumsum) - np.min(cumsum)
                    s = np.std(chunk)
                    
                    if s > 0:
                        rs_chunk.append(r / s)
                
                if rs_chunk:
                    rs_values.append(np.mean(rs_chunk))
            
            if len(rs_values) > 5:
                log_rs = np.log(rs_values)
                log_lags = np.log(lags[:len(rs_values)])
                hurst = np.polyfit(log_lags, log_rs, 1)[0]
                return np.clip(hurst, 0, 1)
            
            return 0.5
            
        except Exception as e:
            logger.warning(f"Hurst calculation error: {e}")
            return 0.5
    
    def _measure_volatility_clustering(self, returns: np.ndarray) -> float:
        """Measure volatility clustering intensity"""
        try:
            if len(returns) < 20:
                return 0.5
            
            window = 10
            vol_series = []
            
            for i in range(window, len(returns)):
                chunk = returns[i-window:i]
                vol_series.append(np.std(chunk))
            
            if len(vol_series) < 10:
                return 0.5
            
            vol_array = np.array(vol_series)
            vol_autocorr = np.corrcoef(vol_array[:-1], vol_array[1:])[0, 1]
            
            clustering_score = (vol_autocorr + 1) / 2
            return np.clip(clustering_score, 0, 1)
            
        except Exception as e:
            logger.warning(f"Volatility clustering error: {e}")
            return 0.5
    
    def _calculate_autocorrelation(self, returns: np.ndarray) -> float:
        """Calculate first-order autocorrelation"""
        try:
            if len(returns) < 10:
                return 0
            
            corr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            return corr if not np.isnan(corr) else 0
            
        except Exception as e:
            logger.warning(f"Autocorrelation error: {e}")
            return 0


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.state = StrategyState()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0

        self.logger.info(f"Initialized {self.config.name} strategy")

    @abstractmethod
    def analyze(self, indicators: MarketIndicators) -> Dict[str, Any]:
        """Analyze market conditions for this strategy"""
        pass

    @abstractmethod
    def generate_signal(self, indicators: MarketIndicators, analysis: Dict[str, Any]) -> TradingSignal:
        """Generate trading signal based on strategy logic"""
        pass

    def should_trade(self, indicators: MarketIndicators) -> Tuple[bool, str]:
        """Check if strategy should trade based on current conditions"""
        if not self.config.enabled:
            return False, "Strategy disabled"

        if self.state.last_action_time:
            time_since_last = (datetime.now() - self.state.last_action_time).total_seconds()
            if time_since_last < self.config.min_time_between_orders:
                return False, f"Cooldown active ({time_since_last:.0f}s)"

        return True, "OK"

    def calculate_position_size(self, action: TradingAction, indicators: MarketIndicators, 
                              confidence: float, available_balance: float) -> float:
        """Calculate position size for the trade"""
        if action == TradingAction.HOLD:
            return 0.0

        base_size = self.config.base_order_size * available_balance

        if self.config.position_scaling:
            base_size *= 0.5 + confidence * 0.5

        # Volatility adjustment
        volatility_mult = 1.0
        if hasattr(indicators, 'volatility'):
            if indicators.volatility > 0.05:
                volatility_mult = 0.7
            elif indicators.volatility < 0.02:
                volatility_mult = 1.2

        position_size = base_size * volatility_mult
        max_size = self.config.max_position_size * available_balance
        position_size = min(position_size, max_size)

        return position_size

    def update_state(self, signal: TradingSignal, executed: bool):
        """Update strategy state after signal generation"""
        self.state.last_action = signal.action
        self.state.last_action_time = datetime.now()
        self.state.last_action_price = signal.price

        if executed and signal.action != TradingAction.HOLD:
            trade_record = {
                "timestamp": datetime.now().isoformat(),
                "action": signal.action.value,
                "price": signal.price,
                "volume": signal.volume,
                "confidence": signal.confidence,
            }

            self.state.trade_history.append(trade_record)
            self.total_trades += 1

            if signal.action == TradingAction.BUY:
                self.state.position_count += 1
                self.state.total_invested += signal.volume * signal.price

                if self.state.position_count > 0:
                    self.state.average_entry_price = self.state.total_invested / self.state.position_count

            elif signal.action == TradingAction.SELL:
                if self.state.average_entry_price > 0:
                    pnl = (signal.price - self.state.average_entry_price) * signal.volume
                    self.total_pnl += pnl

                    if pnl > 0:
                        self.winning_trades += 1

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics"""
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0

        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": win_rate,
            "total_pnl": self.total_pnl,
            "average_entry_price": self.state.average_entry_price,
            "position_count": self.state.position_count,
        }

    def reset(self):
        """Reset strategy state"""
        self.state = StrategyState()
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.logger.info("Strategy state reset")


class DCAStrategy(BaseStrategy):
    """Enhanced Dollar Cost Averaging Strategy with market regime adaptation"""

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

        # DCA specific parameters
        self.buy_interval_hours = config.params.get("buy_interval_hours", 48)  # More conservative
        self.buy_amount_percentage = config.params.get("buy_amount_percentage", 0.03)  # Smaller amounts
        self.price_threshold = config.params.get("price_threshold", None)
        self.accumulation_target = config.params.get("accumulation_target", None)

        # Market regime awareness
        self.market_regime_classifier = MarketRegimeClassifier()

        # Initialize state
        self.state.custom_state["last_dca_buy"] = None
        self.state.custom_state["total_accumulated"] = 0.0

    def analyze(self, indicators: MarketIndicators) -> Dict[str, Any]:
        """Enhanced DCA analysis with market regime consideration"""
        # Update market regime classifier
        current_volume = getattr(indicators, 'current_volume', 1000.0)
        self.market_regime_classifier.update(indicators.current_price, current_volume)
        regime_probs = self.market_regime_classifier.classify_regime()

        analysis = {
            "time_since_last_buy": float("inf"),
            "should_buy": False,
            "price_below_threshold": True,
            "accumulation_complete": False,
            "regime_probs": regime_probs,
            "rsi_oversold": False,
        }

        # Check time since last DCA buy
        last_buy = self.state.custom_state.get("last_dca_buy")
        if last_buy:
            time_since = (datetime.now() - last_buy).total_seconds() / 3600
            analysis["time_since_last_buy"] = time_since
            
            # Adaptive interval based on market regime
            interval_multiplier = self._calculate_interval_multiplier(regime_probs, indicators)
            adaptive_interval = self.buy_interval_hours * interval_multiplier
            
            analysis["should_buy"] = time_since >= adaptive_interval
        else:
            analysis["should_buy"] = True

        # Check price threshold if set
        if self.price_threshold:
            analysis["price_below_threshold"] = indicators.current_price <= self.price_threshold

        # Check accumulation target
        if self.accumulation_target:
            total = self.state.custom_state.get("total_accumulated", 0)
            analysis["accumulation_complete"] = total >= self.accumulation_target

        # RSI check for better entry timing
        if hasattr(indicators, 'rsi'):
            analysis["rsi_oversold"] = indicators.rsi < 40

        return analysis

    def _calculate_interval_multiplier(self, regime_probs: Dict[str, float], indicators: MarketIndicators) -> float:
        """Calculate how to adjust the DCA interval based on market conditions"""
        base_multiplier = 1.0
        
        # Buy more frequently in oversold conditions
        if hasattr(indicators, 'rsi'):
            if indicators.rsi < 30:
                base_multiplier *= 0.5  # Buy twice as often when heavily oversold
            elif indicators.rsi < 40:
                base_multiplier *= 0.7
            elif indicators.rsi > 70:
                base_multiplier *= 1.5  # Less frequent when overbought

        # Adjust for market regime
        if regime_probs["trending"] > 0.6:
            base_multiplier *= 1.2  # Less frequent in trending markets
        elif regime_probs["volatile"] > 0.6:
            base_multiplier *= 0.8  # More frequent in volatile markets

        return np.clip(base_multiplier, 0.25, 3.0)

    def generate_signal(self, indicators: MarketIndicators, analysis: Dict[str, Any]) -> TradingSignal:
        """Generate enhanced DCA trading signal"""
        action = TradingAction.HOLD
        confidence = 0.5
        reasoning = []

        # Check if we should buy
        if (analysis["should_buy"] and 
            analysis["price_below_threshold"] and 
            not analysis["accumulation_complete"]):

            action = TradingAction.BUY
            
            # Base confidence
            confidence = 0.6
            
            # Boost confidence in favorable conditions
            if analysis["rsi_oversold"]:
                confidence += 0.15
                reasoning.append("RSI oversold - favorable entry")
            
            regime = analysis["regime_probs"]
            if regime["mean_reverting"] > 0.5:
                confidence += 0.1
                reasoning.append("Mean reverting market detected")
            
            confidence = min(0.95, confidence)
            reasoning.append(f"Enhanced DCA buy (interval: {self.buy_interval_hours}h)")

        return TradingSignal(
            action=action,
            confidence=confidence,
            volume=0.0,
            price=indicators.current_price,
            reasoning=reasoning,
            risk_level=RiskLevel.LOW,
        )

    def update_state(self, signal: TradingSignal, executed: bool):
        """Update DCA specific state"""
        super().update_state(signal, executed)

        if executed and signal.action == TradingAction.BUY:
            self.state.custom_state["last_dca_buy"] = datetime.now()
            self.state.custom_state["total_accumulated"] = (
                self.state.custom_state.get("total_accumulated", 0) + signal.volume
            )


class MomentumStrategy(BaseStrategy):
    """Enhanced Momentum Trading Strategy with multi-timeframe analysis"""

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

        # Momentum parameters
        self.momentum_period = config.params.get("momentum_period", 20)
        self.momentum_threshold = config.params.get("momentum_threshold", 0.03)  # More conservative
        self.volume_confirmation = config.params.get("volume_confirmation", True)
        self.trend_filter = config.params.get("trend_filter", True)

        # Multi-timeframe tracking
        self.timeframes = {
            TimeFrame.M5: deque(maxlen=100),
            TimeFrame.M15: deque(maxlen=200),
            TimeFrame.H1: deque(maxlen=300)
        }

        # Price history for momentum calculation
        self.state.custom_state["price_history"] = deque(maxlen=self.momentum_period * 2)
        self.state.custom_state["volume_history"] = deque(maxlen=self.momentum_period * 2)

    def analyze(self, indicators: MarketIndicators) -> Dict[str, Any]:
        """Enhanced momentum analysis with multi-timeframe consideration"""
        # Update price history
        price_history = self.state.custom_state.get("price_history", deque(maxlen=self.momentum_period * 2))
        volume_history = self.state.custom_state.get("volume_history", deque(maxlen=self.momentum_period * 2))

        price_history.append(indicators.current_price)
        current_volume = getattr(indicators, 'current_volume', 1000.0)
        volume_history.append(current_volume)

        # Update timeframe data
        self.timeframes[TimeFrame.M15].append(indicators.current_price)

        analysis = {
            "momentum": 0.0,
            "momentum_direction": "neutral",
            "volume_surge": False,
            "trend_aligned": True,
            "strength": 0.0,
            "mtf_momentum": 0.0,
        }

        if len(price_history) >= self.momentum_period:
            # Calculate momentum
            recent_prices = list(price_history)[-self.momentum_period:]
            older_prices = list(price_history)[-self.momentum_period * 2:-self.momentum_period]

            if older_prices:
                recent_avg = np.mean(recent_prices)
                older_avg = np.mean(older_prices)
                momentum = (recent_avg - older_avg) / older_avg

                analysis["momentum"] = momentum

                if momentum > self.momentum_threshold:
                    analysis["momentum_direction"] = "bullish"
                elif momentum < -self.momentum_threshold:
                    analysis["momentum_direction"] = "bearish"

                # Calculate momentum strength
                analysis["strength"] = min(1.0, abs(momentum) / (self.momentum_threshold * 3))

            # Multi-timeframe momentum
            analysis["mtf_momentum"] = self._calculate_mtf_momentum()

            # Check volume surge
            if len(volume_history) >= self.momentum_period:
                recent_volume = np.mean(list(volume_history)[-5:])
                avg_volume = np.mean(list(volume_history))
                analysis["volume_surge"] = recent_volume > avg_volume * 2.0  # Higher threshold

            # Trend alignment check
            if hasattr(indicators, 'market_regime'):
                analysis["trend_aligned"] = (
                    (analysis["momentum_direction"] == "bullish" and indicators.market_regime == "uptrend") or
                    (analysis["momentum_direction"] == "bearish" and indicators.market_regime == "downtrend")
                )

        return analysis

    def _calculate_mtf_momentum(self) -> float:
        """Calculate multi-timeframe momentum score"""
        momentum_scores = {}
        
        for tf, prices in self.timeframes.items():
            if len(prices) >= 20:
                momentum_scores[tf] = self._calculate_momentum_score(list(prices))

        if not momentum_scores:
            return 0.0

        # Weighted average (higher weight for longer timeframes)
        weights = {TimeFrame.M5: 0.2, TimeFrame.M15: 0.3, TimeFrame.H1: 0.5}
        
        total_momentum = 0
        total_weight = 0
        
        for tf, score in momentum_scores.items():
            weight = weights.get(tf, 0.3)
            total_momentum += score * weight
            total_weight += weight

        return total_momentum / total_weight if total_weight > 0 else 0

    def _calculate_momentum_score(self, prices: List[float]) -> float:
        """Calculate normalized momentum score (-1 to 1)"""
        try:
            if len(prices) < 10:
                return 0.0

            prices_array = np.array(prices)

            # Price Rate of Change
            short_roc = (prices_array[-1] - prices_array[-5]) / prices_array[-5] if len(prices) >= 5 else 0
            long_roc = (prices_array[-1] - prices_array[-20]) / prices_array[-20] if len(prices) >= 20 else 0

            # Trend strength using linear regression slope
            x = np.arange(len(prices_array))
            slope, _, r_value, _, _ = stats.linregress(x, prices_array)
            trend_strength = slope * r_value

            # Combine momentum measures
            momentum_score = (
                0.4 * np.tanh(short_roc * 100) +
                0.4 * np.tanh(long_roc * 50) +
                0.2 * np.tanh(trend_strength * 1000)
            )

            return np.clip(momentum_score, -1, 1)

        except Exception as e:
            logger.warning(f"Momentum calculation error: {e}")
            return 0.0

    def generate_signal(self, indicators: MarketIndicators, analysis: Dict[str, Any]) -> TradingSignal:
        """Generate enhanced momentum trading signal"""
        action = TradingAction.HOLD
        confidence = 0.5
        reasoning = []

        # Check for bullish momentum
        if analysis["momentum_direction"] == "bullish":
            conditions_met = True

            if self.volume_confirmation and not analysis["volume_surge"]:
                conditions_met = False
                reasoning.append("Waiting for volume confirmation")

            if self.trend_filter and not analysis["trend_aligned"]:
                conditions_met = False
                reasoning.append("Momentum not aligned with trend")

            if conditions_met:
                action = TradingAction.BUY
                confidence = 0.5 + analysis["strength"] * 0.4
                
                # Multi-timeframe boost
                if analysis["mtf_momentum"] > 0.3:
                    confidence = min(0.9, confidence + 0.1)
                    reasoning.append("Multi-timeframe momentum confirmed")
                
                reasoning.append(f"Bullish momentum: {analysis['momentum']:.1%}")
                if analysis["volume_surge"]:
                    reasoning.append("Volume surge confirmed")

        # Check for bearish momentum
        elif analysis["momentum_direction"] == "bearish":
            conditions_met = True

            if self.trend_filter and not analysis["trend_aligned"]:
                conditions_met = False

            if conditions_met:
                action = TradingAction.SELL
                confidence = 0.5 + analysis["strength"] * 0.4
                
                if analysis["mtf_momentum"] < -0.3:
                    confidence = min(0.9, confidence + 0.1)
                    reasoning.append("Multi-timeframe bearish momentum")
                
                reasoning.append(f"Bearish momentum: {analysis['momentum']:.1%}")

        # RSI filters
        if hasattr(indicators, 'rsi'):
            if action == TradingAction.BUY and indicators.rsi > 75:
                action = TradingAction.HOLD
                reasoning.append("RSI extremely overbought - avoiding chase")
            elif action == TradingAction.SELL and indicators.rsi < 25:
                action = TradingAction.HOLD
                reasoning.append("RSI extremely oversold - avoiding panic sell")

        return TradingSignal(
            action=action,
            confidence=confidence,
            volume=0.0,
            price=indicators.current_price,
            reasoning=reasoning,
            risk_level=RiskLevel.MEDIUM if confidence > 0.7 else RiskLevel.HIGH,
        )


class MeanReversionStrategy(BaseStrategy):
    """Enhanced Mean Reversion Strategy with statistical arbitrage elements"""

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

        # Mean reversion parameters
        self.lookback_period = config.params.get("lookback_period", 50)
        self.deviation_threshold = config.params.get("deviation_threshold", 2.5)  # Higher threshold
        self.use_bollinger = config.params.get("use_bollinger", True)
        self.use_rsi = config.params.get("use_rsi", True)
        self.rsi_oversold = config.params.get("rsi_oversold", 25)  # More extreme
        self.rsi_overbought = config.params.get("rsi_overbought", 75)  # More extreme

        # Statistical arbitrage components
        self.price_history = deque(maxlen=200)
        self.scaler = StandardScaler()

    def analyze(self, indicators: MarketIndicators) -> Dict[str, Any]:
        """Enhanced mean reversion analysis with statistical elements"""
        # Update price history
        self.price_history.append(indicators.current_price)

        analysis = {
            "deviation_from_mean": 0.0,
            "z_score": 0.0,
            "bollinger_position": 0.5,
            "rsi_signal": "neutral",
            "reversion_expected": False,
            "reversion_direction": None,
            "half_life": 10.0,
        }

        # Calculate deviation from moving average
        if hasattr(indicators, 'ma_long') and indicators.ma_long > 0:
            deviation = (indicators.current_price - indicators.ma_long) / indicators.ma_long
            analysis["deviation_from_mean"] = deviation

            # Calculate z-score if we have Bollinger Bands
            if (self.use_bollinger and 
                hasattr(indicators, 'bollinger_upper') and 
                hasattr(indicators, 'bollinger_lower')):
                
                band_width = indicators.bollinger_upper - indicators.bollinger_lower
                if band_width > 0:
                    bollinger_middle = (indicators.bollinger_upper + indicators.bollinger_lower) / 2
                    z_score = (indicators.current_price - bollinger_middle) / (band_width / 4)
                    analysis["z_score"] = z_score

                    # Bollinger position
                    analysis["bollinger_position"] = (
                        (indicators.current_price - indicators.bollinger_lower) / band_width
                        if band_width > 0 else 0.5
                    )

        # Calculate half-life for mean reversion
        if len(self.price_history) >= 20:
            analysis["half_life"] = self._calculate_half_life(np.array(self.price_history))

        # RSI signals
        if self.use_rsi and hasattr(indicators, 'rsi'):
            if indicators.rsi < self.rsi_oversold:
                analysis["rsi_signal"] = "oversold"
            elif indicators.rsi > self.rsi_overbought:
                analysis["rsi_signal"] = "overbought"

        # Determine if reversion is expected
        if abs(analysis["z_score"]) > self.deviation_threshold:
            analysis["reversion_expected"] = True
            analysis["reversion_direction"] = "up" if analysis["z_score"] < 0 else "down"

        return analysis

    def _calculate_half_life(self, prices: np.ndarray) -> float:
        """Calculate mean reversion half-life using Ornstein-Uhlenbeck process"""
        try:
            if len(prices) < 20:
                return 10.0

            price_diff = np.diff(prices)
            price_lag = prices[:-1]
            price_mean = np.mean(prices)

            # Linear regression: Δx = α + β(x - μ) + ε
            X = price_lag - price_mean
            y = price_diff

            if len(X) > 0 and np.var(X) > 0:
                beta = np.cov(X, y)[0, 1] / np.var(X)
                theta = -beta

                if theta > 0:
                    half_life = np.log(2) / theta
                    return max(1, min(100, half_life))

            return 10.0

        except Exception as e:
            logger.warning(f"Half-life calculation error: {e}")
            return 10.0

    def generate_signal(self, indicators: MarketIndicators, analysis: Dict[str, Any]) -> TradingSignal:
        """Generate enhanced mean reversion signal"""
        action = TradingAction.HOLD
        confidence = 0.5
        reasoning = []

        if analysis["reversion_expected"]:
            # Calculate reversion strength based on half-life
            reversion_strength = 1 / max(1, analysis["half_life"] / 10)
            
            # Buy signal - price below mean and expected to revert up
            if analysis["reversion_direction"] == "up":
                buy_conditions = [
                    analysis["z_score"] < -self.deviation_threshold,
                    analysis["bollinger_position"] < 0.2 if self.use_bollinger else True,
                    analysis["rsi_signal"] == "oversold" if self.use_rsi else True,
                ]

                if all(buy_conditions):
                    action = TradingAction.BUY
                    confidence = min(0.9, 0.5 + abs(analysis["z_score"]) * 0.1 * reversion_strength)
                    reasoning.append(f"Price {analysis['deviation_from_mean']:.1%} below mean")
                    reasoning.append(f"Z-score: {analysis['z_score']:.2f}, Half-life: {analysis['half_life']:.1f}")
                    if analysis["rsi_signal"] == "oversold":
                        reasoning.append(f"RSI oversold: {indicators.rsi:.1f}")

            # Sell signal - price above mean and expected to revert down
            elif analysis["reversion_direction"] == "down":
                sell_conditions = [
                    analysis["z_score"] > self.deviation_threshold,
                    analysis["bollinger_position"] > 0.8 if self.use_bollinger else True,
                    analysis["rsi_signal"] == "overbought" if self.use_rsi else True,
                ]

                if all(sell_conditions):
                    action = TradingAction.SELL
                    confidence = min(0.9, 0.5 + abs(analysis["z_score"]) * 0.1 * reversion_strength)
                    reasoning.append(f"Price {analysis['deviation_from_mean']:.1%} above mean")
                    reasoning.append(f"Z-score: {analysis['z_score']:.2f}, Half-life: {analysis['half_life']:.1f}")
                    if analysis["rsi_signal"] == "overbought":
                        reasoning.append(f"RSI overbought: {indicators.rsi:.1f}")

        return TradingSignal(
            action=action,
            confidence=confidence,
            volume=0.0,
            price=indicators.current_price,
            reasoning=reasoning,
            risk_level=RiskLevel.MEDIUM,
        )


class VolatilityBreakoutStrategy(BaseStrategy):
    """Advanced volatility breakout strategy with dynamic thresholds"""

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

        self.lookback = config.params.get("lookback_period", 50)
        self.volume_threshold = config.params.get("volume_threshold", 2.0)
        
        self.price_history = deque(maxlen=self.lookback)
        self.volume_history = deque(maxlen=self.lookback)

    def analyze(self, indicators: MarketIndicators) -> Dict[str, Any]:
        """Analyze for volatility breakout opportunities"""
        current_volume = getattr(indicators, 'current_volume', 1000.0)
        
        self.price_history.append(indicators.current_price)
        self.volume_history.append(current_volume)

        analysis = {
            "breakout_type": None,
            "breakout_strength": 0.0,
            "volume_confirmed": False,
            "volatility_expansion": False,
            "dynamic_threshold": 0.02,
        }

        if len(self.price_history) >= 20:
            prices = np.array(self.price_history)
            volumes = np.array(self.volume_history)

            # Calculate dynamic volatility threshold
            returns = np.diff(np.log(prices))
            current_vol = np.std(returns[-10:]) if len(returns) >= 10 else np.std(returns)
            historical_vol = np.std(returns)
            
            # VWAP calculation
            vwap = np.sum(prices * volumes) / np.sum(volumes) if np.sum(volumes) > 0 else prices[-1]

            # Dynamic Bollinger-like bands
            vol_multiplier = max(1.5, current_vol / historical_vol * 2) if historical_vol > 0 else 2.0
            upper_threshold = vwap * (1 + vol_multiplier * historical_vol)
            lower_threshold = vwap * (1 - vol_multiplier * historical_vol)

            # Volume confirmation
            avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            analysis["volume_confirmed"] = volume_ratio > self.volume_threshold

            # Breakout detection
            current_price = indicators.current_price
            if current_price > upper_threshold:
                analysis["breakout_type"] = "upward"
                analysis["breakout_strength"] = (current_price - upper_threshold) / upper_threshold
            elif current_price < lower_threshold:
                analysis["breakout_type"] = "downward" 
                analysis["breakout_strength"] = (lower_threshold - current_price) / lower_threshold

            # Volatility expansion detection
            analysis["volatility_expansion"] = current_vol > historical_vol * 1.5

        return analysis

    def generate_signal(self, indicators: MarketIndicators, analysis: Dict[str, Any]) -> TradingSignal:
        """Generate volatility breakout signal"""
        action = TradingAction.HOLD
        confidence = 0.5
        reasoning = []

        if analysis["breakout_type"] and analysis["volume_confirmed"]:
            base_confidence = 0.6 + analysis["breakout_strength"] * 2

            if analysis["volatility_expansion"]:
                base_confidence += 0.1
                reasoning.append("Volatility expansion detected")

            if analysis["breakout_type"] == "upward":
                action = TradingAction.BUY
                confidence = min(0.9, base_confidence)
                reasoning.append("Upward volatility breakout")
                reasoning.append(f"Volume surge: {analysis.get('volume_ratio', 1):.1f}x")

            elif analysis["breakout_type"] == "downward":
                action = TradingAction.SELL
                confidence = min(0.9, base_confidence)
                reasoning.append("Downward volatility breakout")
                reasoning.append(f"Volume surge: {analysis.get('volume_ratio', 1):.1f}x")

        return TradingSignal(
            action=action,
            confidence=confidence,
            volume=0.0,
            price=indicators.current_price,
            reasoning=reasoning,
            risk_level=RiskLevel.HIGH if analysis["breakout_type"] else RiskLevel.MEDIUM,
        )


class HybridAdaptiveStrategy(BaseStrategy):
    """Enhanced Hybrid Strategy combining multiple approaches with dynamic weighting"""

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

        # Initialize sub-strategies with more conservative parameters
        self.strategies = {
            "dca": DCAStrategy(self._create_sub_config("dca", {
                "buy_interval_hours": 48,
                "buy_amount_percentage": 0.02
            })),
            "momentum": MomentumStrategy(self._create_sub_config("momentum", {
                "momentum_period": 25,
                "momentum_threshold": 0.035
            })),
            "mean_reversion": MeanReversionStrategy(self._create_sub_config("mean_reversion", {
                "lookback_period": 60,
                "deviation_threshold": 2.5
            })),
            "vol_breakout": VolatilityBreakoutStrategy(self._create_sub_config("vol_breakout", {
                "lookback_period": 50,
                "volume_threshold": 2.5
            }))
        }

        # More conservative strategy weights
        self.state.custom_state["strategy_weights"] = {
            "dca": 0.15,           # Reduced from typical 25%
            "momentum": 0.30,      # Balanced
            "mean_reversion": 0.35, # Slightly higher for stability
            "vol_breakout": 0.20   # Conservative breakout trading
        }

        # Performance tracking for adaptation
        self.state.custom_state["strategy_performance"] = {
            name: {"trades": 0, "wins": 0, "total_pnl": 0.0}
            for name in self.strategies.keys()
        }

        # Market regime awareness
        self.market_regime_classifier = MarketRegimeClassifier()

    def _create_sub_config(self, strategy_name: str, params: Dict[str, Any]) -> StrategyConfig:
        """Create configuration for sub-strategy"""
        sub_config = StrategyConfig(
            name=f"{self.config.name}_{strategy_name}",
            enabled=True,
            max_position_size=self.config.max_position_size * 0.4,  # More conservative
            stop_loss_enabled=self.config.stop_loss_enabled,
            stop_loss_percentage=self.config.stop_loss_percentage,
            take_profit_enabled=self.config.take_profit_enabled,
            take_profit_percentage=self.config.take_profit_percentage,
            params=params
        )
        return sub_config

    def analyze(self, indicators: MarketIndicators) -> Dict[str, Any]:
        """Analyze using all sub-strategies with market regime consideration"""
        # Update market regime classifier
        current_volume = getattr(indicators, 'current_volume', 1000.0)
        self.market_regime_classifier.update(indicators.current_price, current_volume)
        regime_probs = self.market_regime_classifier.classify_regime()

        analysis = {
            "sub_analyses": {},
            "market_regime": regime_probs,
            "dominant_strategy": self._determine_dominant_strategy(regime_probs, indicators),
            "market_condition": self._determine_market_condition(indicators),
        }

        # Run analysis for each sub-strategy
        for name, strategy in self.strategies.items():
            try:
                analysis["sub_analyses"][name] = strategy.analyze(indicators)
            except Exception as e:
                logger.warning(f"Strategy {name} analysis failed: {e}")
                analysis["sub_analyses"][name] = {}

        return analysis

    def _determine_dominant_strategy(self, regime_probs: Dict[str, float], indicators: MarketIndicators) -> str:
        """Determine which strategy should have higher weight based on market conditions"""
        # Regime-based strategy selection
        if regime_probs["trending"] > 0.5:
            return "momentum"
        elif regime_probs["mean_reverting"] > 0.5:
            return "mean_reversion"
        elif regime_probs["volatile"] > 0.5:
            return "vol_breakout"
        
        # RSI-based fallback
        if hasattr(indicators, 'rsi'):
            if indicators.rsi < 35:
                return "dca"  # Favor DCA in oversold conditions
            elif indicators.rsi > 65:
                return "mean_reversion"  # Favor mean reversion when overbought
        
        return "mean_reversion"  # Default to most conservative

    def _determine_market_condition(self, indicators: MarketIndicators) -> str:
        """Determine overall market condition"""
        conditions = []

        # Trend analysis
        if hasattr(indicators, 'market_regime'):
            if indicators.market_regime in ["uptrend", "downtrend"]:
                conditions.append("trending")
            else:
                conditions.append("ranging")
        else:
            conditions.append("ranging")

        # Volatility analysis
        if hasattr(indicators, 'volatility'):
            if indicators.volatility > 0.06:
                conditions.append("high_volatility")
            elif indicators.volatility < 0.025:
                conditions.append("low_volatility")
            else:
                conditions.append("normal_volatility")
        else:
            conditions.append("normal_volatility")

        # Risk environment
        risk_off_prob = getattr(indicators, 'risk_off_probability', 0.0)
        if risk_off_prob > 0.6:
            conditions.append("risk_off")
        else:
            conditions.append("risk_on")

        return "_".join(conditions)

    def generate_signal(self, indicators: MarketIndicators, analysis: Dict[str, Any]) -> TradingSignal:
        """Generate hybrid signal with enhanced weighting logic"""
        weights = dict(self.state.custom_state["strategy_weights"])
        
        # Boost dominant strategy weight
        dominant = analysis["dominant_strategy"]
        if dominant in weights:
            # Redistribute weights to favor dominant strategy
            boost = 0.15
            weights[dominant] = min(0.5, weights[dominant] + boost)
            
            # Reduce others proportionally
            remaining_weight = 1.0 - weights[dominant]
            other_strategies = [s for s in weights.keys() if s != dominant]
            if other_strategies:
                weight_per_other = remaining_weight / len(other_strategies)
                for strategy in other_strategies:
                    weights[strategy] = weight_per_other

        # Collect signals from all strategies
        signals = {}
        for name, strategy in self.strategies.items():
            try:
                sub_analysis = analysis["sub_analyses"].get(name, {})
                signals[name] = strategy.generate_signal(indicators, sub_analysis)
            except Exception as e:
                logger.warning(f"Strategy {name} signal generation failed: {e}")
                continue

        # Weighted consensus with more conservative thresholds
        buy_score = 0.0
        sell_score = 0.0
        combined_reasoning = []

        for name, signal in signals.items():
            weight = weights.get(name, 0.25)

            if signal.action == TradingAction.BUY:
                buy_score += weight * signal.confidence
                combined_reasoning.extend([f"[{name}] {r}" for r in signal.reasoning[:1]])
            elif signal.action == TradingAction.SELL:
                sell_score += weight * signal.confidence
                combined_reasoning.extend([f"[{name}] {r}" for r in signal.reasoning[:1]])

        # More conservative thresholds for final decision
        if buy_score > 0.45 and buy_score > sell_score * 1.3:  # Need strong buy consensus
            action = TradingAction.BUY
            confidence = min(0.9, buy_score + 0.1)
        elif sell_score > 0.35 and sell_score > buy_score * 1.2:  # Easier to sell for rebalancing
            action = TradingAction.SELL
            confidence = min(0.9, sell_score + 0.15)
        else:
            action = TradingAction.HOLD
            confidence = 0.5
            combined_reasoning = [f"Insufficient consensus: BUY={buy_score:.2f}, SELL={sell_score:.2f}"]

        # Adapt weights periodically based on performance
        if self.total_trades > 0 and self.total_trades % 15 == 0:
            self._adapt_weights()

        return TradingSignal(
            action=action,
            confidence=confidence,
            volume=0.0,
            price=indicators.current_price,
            reasoning=combined_reasoning[:3],
            risk_level=RiskLevel.MEDIUM,
        )

    def _adapt_weights(self):
        """Adapt strategy weights based on recent performance"""
        try:
            performances = self.state.custom_state["strategy_performance"]

            # Calculate performance scores with minimum trade requirement
            scores = {}
            for name, perf in performances.items():
                if perf["trades"] > 3:  # Minimum trades for reliability
                    # Combine win rate with total PnL
                    win_rate = perf["wins"] / perf["trades"]
                    pnl_score = max(0, min(1, (perf["total_pnl"] / max(1, perf["trades"])) + 0.5))
                    scores[name] = (win_rate * 0.6) + (pnl_score * 0.4)
                else:
                    scores[name] = 0.4  # Default score for strategies with few trades

            # Apply minimum and maximum weight constraints
            for name in scores:
                scores[name] = max(0.05, min(0.5, scores[name]))  # 5% min, 50% max

            # Normalize to sum to 1
            total_score = sum(scores.values())
            if total_score > 0:
                self.state.custom_state["strategy_weights"] = {
                    name: score / total_score for name, score in scores.items()
                }

            logger.info(f"Adapted strategy weights: {self.state.custom_state['strategy_weights']}")

        except Exception as e:
            logger.error(f"Weight adaptation error: {e}")


class AdvancedStrategyEngine:
    """Main engine coordinating all advanced strategies with enhanced consensus logic"""

    def __init__(self):
        # Initialize strategies with conservative parameters
        self.strategies = {
            'adaptive_dca': DCAStrategy(StrategyConfig(
                name="Adaptive DCA",
                params={"buy_interval_hours": 48, "buy_amount_percentage": 0.025}
            )),
            'momentum': MomentumStrategy(StrategyConfig(
                name="Enhanced Momentum", 
                params={"momentum_period": 25, "momentum_threshold": 0.035}
            )),
            'mean_reversion': MeanReversionStrategy(StrategyConfig(
                name="Statistical Mean Reversion",
                params={"lookback_period": 60, "deviation_threshold": 2.5}
            )),
            'vol_breakout': VolatilityBreakoutStrategy(StrategyConfig(
                name="Volatility Breakout",
                params={"lookback_period": 50, "volume_threshold": 2.5}
            ))
        }

        # Conservative strategy weights
        self.weights = {
            'adaptive_dca': 0.15,
            'momentum': 0.30,
            'mean_reversion': 0.35,
            'vol_breakout': 0.20
        }

        self.performance_tracking = {name: [] for name in self.strategies.keys()}
        self.market_regime_classifier = MarketRegimeClassifier()

    def update_all_strategies(self, price: float, volume: float, indicators: Dict[str, float]):
        """Update all strategies with new market data"""
        try:
            # Update market regime classifier
            self.market_regime_classifier.update(price, volume)
            
            # Update individual strategies that need market data
            for name, strategy in self.strategies.items():
                if hasattr(strategy, 'market_regime_classifier'):
                    strategy.market_regime_classifier.update(price, volume)
                
                # Update price/volume histories for strategies that use them
                if hasattr(strategy, 'price_history'):
                    strategy.price_history.append(price)
                if hasattr(strategy, 'volume_history'):
                    strategy.volume_history.append(volume)

        except Exception as e:
            logger.error(f"Strategy update error: {e}")

    def generate_consensus_signal(self, current_price: float, indicators: Dict[str, float]) -> AdvancedSignal:
        """Generate consensus signal from all strategies with conservative thresholds"""
        try:
            # Create MarketIndicators object from indicators dict
            market_indicators = self._create_market_indicators(current_price, indicators)
            
            signals = {}
            
            # Get signals from all strategies
            for name, strategy in self.strategies.items():
                try:
                    analysis = strategy.analyze(market_indicators)
                    signal = strategy.generate_signal(market_indicators, analysis)
                    
                    # Convert TradingSignal to AdvancedSignal
                    adv_signal = AdvancedSignal(
                        action=signal.action.value.upper(),
                        confidence=signal.confidence,
                        timeframe=TimeFrame.M15,
                        strategy_name=name,
                        entry_price=current_price,
                        reasoning=signal.reasoning,
                        metadata={"original_signal": signal}
                    )
                    signals[name] = adv_signal
                    
                except Exception as e:
                    logger.warning(f"Strategy {name} signal error: {e}")
                    continue

            if not signals:
                return AdvancedSignal("HOLD", 0.0, TimeFrame.M15, "Consensus", current_price)

            # Calculate weighted consensus with conservative thresholds
            buy_score = 0.0
            sell_score = 0.0
            total_weight = 0.0

            all_reasoning = []
            combined_metadata = {}

            for name, signal in signals.items():
                weight = self.weights.get(name, 0.25)

                if signal.action == "BUY":
                    buy_score += weight * signal.confidence
                elif signal.action == "SELL":
                    sell_score += weight * signal.confidence

                total_weight += weight

                # Collect reasoning and metadata
                if signal.action != "HOLD":
                    all_reasoning.extend([f"[{name}] {r}" for r in signal.reasoning[:1]])
                    combined_metadata[name] = signal.metadata

            # Normalize scores
            if total_weight > 0:
                buy_score /= total_weight
                sell_score /= total_weight

            # Determine consensus action with conservative thresholds
            if buy_score > 0.45 and buy_score > sell_score * 1.3:
                action = "BUY"
                confidence = min(0.9, buy_score + 0.15)
                position_size = 0.001  # Conservative position size
                
            elif sell_score > 0.35 and sell_score > buy_score * 1.2:
                action = "SELL"
                confidence = min(0.9, sell_score + 0.15)
                position_size = 0.001
                
            else:
                action = "HOLD"
                confidence = 0.5
                position_size = 0.0
                all_reasoning = [f"Insufficient consensus: BUY={buy_score:.3f}, SELL={sell_score:.3f}"]

            # Create consensus signal
            consensus_signal = AdvancedSignal(
                action=action,
                confidence=confidence,
                timeframe=TimeFrame.M15,
                strategy_name="Advanced_Consensus",
                entry_price=current_price,
                position_size=position_size,
                reasoning=all_reasoning[:3],
                metadata={
                    "individual_signals": {name: s.action for name, s in signals.items()},
                    "buy_score": buy_score,
                    "sell_score": sell_score,
                    "strategy_metadata": combined_metadata,
                    "regime_probs": self.market_regime_classifier.classify_regime()
                }
            )

            return consensus_signal

        except Exception as e:
            logger.error(f"Consensus signal error: {e}")
            return AdvancedSignal("HOLD", 0.0, TimeFrame.M15, "Consensus", current_price)

    def _create_market_indicators(self, current_price: float, indicators: Dict[str, float]):
        """Create MarketIndicators object from indicators dictionary"""
        # This is a simplified version - you might want to create a proper MarketIndicators class
        class SimpleMarketIndicators:
            def __init__(self, price, indicators_dict):
                self.current_price = price
                self.rsi = indicators_dict.get('rsi', 50)
                self.macd = indicators_dict.get('macd', 0)
                self.signal = indicators_dict.get('signal', 0)
                self.volatility = indicators_dict.get('volatility', 0.02)
                self.sentiment = indicators_dict.get('sentiment', 0)
                self.current_volume = indicators_dict.get('volume', 1000)
                self.market_regime = "normal"
                self.risk_off_probability = 0.0
                
                # Add Bollinger Band attributes if available
                self.bollinger_upper = indicators_dict.get('upper_band', price * 1.02)
                self.bollinger_lower = indicators_dict.get('lower_band', price * 0.98)
                self.ma_long = indicators_dict.get('ma_long', price)

        return SimpleMarketIndicators(current_price, indicators)

    def optimize_weights(self, lookback_periods: int = 50):
        """Dynamically optimize strategy weights based on recent performance"""
        try:
            performance = self.get_strategy_performance()

            # Calculate performance scores with conservative adjustments
            scores = {}
            for name, perf in performance.items():
                if perf['total_signals'] > 5:
                    # Weighted score combining win rate and consistency
                    score = (perf['win_rate'] * 0.7) + (perf['avg_confidence'] * 0.3)
                    scores[name] = max(0.05, min(0.45, score))  # Constrain between 5% and 45%
                else:
                    scores[name] = 0.2  # Default weight for low-activity strategies

            # Normalize to sum to 1
            total_score = sum(scores.values())
            if total_score > 0:
                self.weights = {name: score / total_score for name, score in scores.items()}

            logger.info(f"Optimized strategy weights: {self.weights}")

        except Exception as e:
            logger.error(f"Weight optimization error: {e}")

    def get_strategy_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all strategies"""
        performance = {}

        for name in self.strategies.keys():
            signals = self.performance_tracking.get(name, [])

            if signals:
                total_signals = len(signals)
                profitable_signals = sum(1 for s in signals if s.get('pnl', 0) > 0)

                performance[name] = {
                    'total_signals': total_signals,
                    'win_rate': profitable_signals / total_signals if total_signals > 0 else 0,
                    'avg_confidence': np.mean([s.get('confidence', 0) for s in signals]),
                    'total_pnl': sum(s.get('pnl', 0) for s in signals)
                }
            else:
                performance[name] = {
                    'total_signals': 0,
                    'win_rate': 0,
                    'avg_confidence': 0,
                    'total_pnl': 0
                }

        return performance


class StrategyFactory:
    """Factory class for creating and managing strategies"""

    @staticmethod
    def create_strategy(strategy_type: StrategyType, config: Optional[StrategyConfig] = None) -> BaseStrategy:
        """Create a strategy instance"""
        
        # Default configurations with conservative parameters
        default_configs = {
            StrategyType.DCA: StrategyConfig(
                name="Conservative DCA Strategy",
                params={"buy_interval_hours": 48, "buy_amount_percentage": 0.03},
            ),
            StrategyType.MOMENTUM: StrategyConfig(
                name="Enhanced Momentum Strategy",
                params={"momentum_period": 25, "momentum_threshold": 0.035},
            ),
            StrategyType.MEAN_REVERSION: StrategyConfig(
                name="Statistical Mean Reversion Strategy",
                params={"lookback_period": 60, "deviation_threshold": 2.5},
            ),
            StrategyType.VOLATILITY_BREAKOUT: StrategyConfig(
                name="Volatility Breakout Strategy",
                params={"lookback_period": 50, "volume_threshold": 2.5},
            ),
            StrategyType.HYBRID: StrategyConfig(
                name="Hybrid Adaptive Strategy"
            ),
        }

        # Use provided config or default
        if not config:
            config = default_configs.get(strategy_type)
            if not config:
                raise ValueError(f"Unknown strategy type: {strategy_type}")

        # Create strategy instance
        strategy_classes = {
            StrategyType.DCA: DCAStrategy,
            StrategyType.MOMENTUM: MomentumStrategy,
            StrategyType.MEAN_REVERSION: MeanReversionStrategy,
            StrategyType.VOLATILITY_BREAKOUT: VolatilityBreakoutStrategy,
            StrategyType.HYBRID: HybridAdaptiveStrategy,
        }

        strategy_class = strategy_classes.get(strategy_type)
        if not strategy_class:
            raise ValueError(f"Strategy class not found for type: {strategy_type}")

        return strategy_class(config)

    @staticmethod
    def create_from_config_file(config_file: str) -> BaseStrategy:
        """Create strategy from configuration file"""
        with open(config_file, "r") as f:
            config_data = json.load(f)

        strategy_type = StrategyType(config_data["type"])
        config = StrategyConfig(**config_data["config"])

        return StrategyFactory.create_strategy(strategy_type, config)


# Simplified fallback indicators for when core.indicators is not available
def calculate_rsi_simple(prices: List[float], period: int = 14) -> float:
    """Simple RSI calculation fallback"""
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
    """Simple MACD calculation fallback"""
    if len(prices) < slow:
        return 0.0, 0.0
    
    prices_array = np.array(prices)
    
    # Simplified EMA calculation
    ema_fast = prices_array[-1]  # Simplified
    ema_slow = np.mean(prices_array[-slow:])
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line  # Simplified
    
    return macd_line, signal_line


def calculate_vwap_simple(prices: List[float], volumes: List[float]) -> float:
    """Simple VWAP calculation fallback"""
    if not prices or not volumes or len(prices) != len(volumes):
        return prices[-1] if prices else 0
    
    prices_array = np.array(prices[-50:])  # Last 50 periods
    volumes_array = np.array(volumes[-50:])
    
    if np.sum(volumes_array) == 0:
        return prices[-1]
    
    return np.sum(prices_array * volumes_array) / np.sum(volumes_array)


# Use advanced indicators if available, otherwise use simple fallbacks
if INDICATORS_AVAILABLE:
    logger.info("Using advanced indicators from core.indicators module")
else:
    logger.info("Using simplified fallback indicators")
    # Override with simple versions
    calculate_rsi = calculate_rsi_simple
    calculate_macd = calculate_macd_simple
    calculate_vwap = calculate_vwap_simple


# Example usage and testing
if __name__ == "__main__":
    # Test strategy creation
    print("Testing Unified Strategies Module...")
    
    try:
        # Create different strategies
        dca = StrategyFactory.create_strategy(StrategyType.DCA)
        momentum = StrategyFactory.create_strategy(StrategyType.MOMENTUM)
        hybrid = StrategyFactory.create_strategy(StrategyType.HYBRID)
        
        print(f"✅ Created DCA strategy: {dca.config.name}")
        print(f"✅ Created Momentum strategy: {momentum.config.name}")
        print(f"✅ Created Hybrid strategy: {hybrid.config.name}")
        
        # Test advanced strategy engine
        engine = AdvancedStrategyEngine()
        print(f"✅ Created Advanced Strategy Engine with {len(engine.strategies)} strategies")
        
        # Test market regime classifier
        classifier = MarketRegimeClassifier()
        test_prices = [100 + i + np.random.random() for i in range(50)]
        test_volumes = [1000 + np.random.random() * 500 for _ in range(50)]
        
        for price, volume in zip(test_prices[-10:], test_volumes[-10:]):
            classifier.update(price, volume)
        
        regime = classifier.classify_regime()
        print(f"✅ Market regime classification: {regime}")
        
        print("\n🎉 All unified strategy components loaded successfully!")
        print("\nAvailable Strategy Types:")
        for strategy_type in StrategyType:
            print(f"   - {strategy_type.value}")
            
        print(f"\nIndicators Module: {'✅ Advanced' if INDICATORS_AVAILABLE else '⚠️ Fallback'}")
        
    except Exception as e:
        print(f"❌ Error testing strategies: {e}")
        import traceback
        traceback.print_exc()


# Configuration example for reference
EXAMPLE_STRATEGY_CONFIG = {
    "type": "hybrid",
    "config": {
        "name": "Custom Hybrid Strategy",
        "enabled": True,
        "max_position_size": 0.15,
        "base_order_size": 0.08,
        "stop_loss_enabled": True,
        "stop_loss_percentage": 0.025,
        "take_profit_enabled": True,
        "take_profit_percentage": 0.08,
        "params": {
            "sub_strategies": ["dca", "momentum", "mean_reversion"],
            "adaptation_enabled": True,
            "conservative_mode": True
        }
    }
}