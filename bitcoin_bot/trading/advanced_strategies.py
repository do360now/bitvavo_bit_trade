# trading/advanced_strategies.py
"""
Institutional-Grade Multi-Timeframe Strategy Engine
Implements sophisticated trading strategies used by top-tier funds
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from datetime import datetime, timedelta
import talib
from scipy import stats
from sklearn.preprocessing import StandardScaler
from collections import deque

logger = logging.getLogger(__name__)


class TimeFrame(Enum):
    TICK = "tick"
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


class SignalStrength(Enum):
    VERY_WEAK = 0.1
    WEAK = 0.3
    MODERATE = 0.5
    STRONG = 0.7
    VERY_STRONG = 0.9


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
        
        # Volatility clustering (GARCH-like measure)
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
                    
                # Reshape returns into chunks
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
            
            # Calculate rolling volatility
            window = 10
            vol_series = []
            
            for i in range(window, len(returns)):
                chunk = returns[i-window:i]
                vol_series.append(np.std(chunk))
            
            if len(vol_series) < 10:
                return 0.5
            
            # Measure autocorrelation in volatility
            vol_array = np.array(vol_series)
            vol_autocorr = np.corrcoef(vol_array[:-1], vol_array[1:])[0, 1]
            
            # Convert to 0-1 scale
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


class StatisticalArbitrageStrategy:
    """Statistical arbitrage using mean reversion and cointegration"""
    
    def __init__(self, lookback: int = 200, z_threshold: float = 2.0):
        self.lookback = lookback
        self.z_threshold = z_threshold
        self.price_history = deque(maxlen=lookback)
        self.scaler = StandardScaler()
        
    def update(self, price: float, volume: float, indicators: Dict[str, float]):
        """Update with new market data"""
        self.price_history.append(price)
    
    def generate_signal(self, current_price: float, indicators: Dict[str, float]) -> AdvancedSignal:
        """Generate statistical arbitrage signal"""
        if len(self.price_history) < self.lookback // 2:
            return AdvancedSignal("HOLD", 0.0, TimeFrame.M15, "StatArb", current_price)
        
        try:
            prices = np.array(self.price_history)
            
            # Calculate z-score of current price vs historical mean
            mean_price = np.mean(prices)
            std_price = np.std(prices)
            
            if std_price == 0:
                return AdvancedSignal("HOLD", 0.0, TimeFrame.M15, "StatArb", current_price)
            
            z_score = (current_price - mean_price) / std_price
            
            # Enhanced signal with Ornstein-Uhlenbeck mean reversion
            half_life = self._calculate_half_life(prices)
            reversion_strength = 1 / max(1, half_life / 10)  # Stronger signal for faster reversion
            
            # Generate signal based on z-score and reversion strength
            if z_score > self.z_threshold:
                # Price too high, sell signal
                confidence = min(0.95, abs(z_score) / self.z_threshold * reversion_strength * 0.5)
                return AdvancedSignal(
                    "SELL", confidence, TimeFrame.M15, "StatArb", current_price,
                    stop_loss=current_price * 1.01,
                    take_profit=mean_price,
                    reasoning=[f"Z-score: {z_score:.2f}, Half-life: {half_life:.1f}"]
                )
            
            elif z_score < -self.z_threshold:
                # Price too low, buy signal
                confidence = min(0.95, abs(z_score) / self.z_threshold * reversion_strength * 0.5)
                return AdvancedSignal(
                    "BUY", confidence, TimeFrame.M15, "StatArb", current_price,
                    stop_loss=current_price * 0.99,
                    take_profit=mean_price,
                    reasoning=[f"Z-score: {z_score:.2f}, Half-life: {half_life:.1f}"]
                )
            
            return AdvancedSignal("HOLD", 0.0, TimeFrame.M15, "StatArb", current_price)
            
        except Exception as e:
            logger.error(f"StatArb signal error: {e}")
            return AdvancedSignal("HOLD", 0.0, TimeFrame.M15, "StatArb", current_price)
    
    def _calculate_half_life(self, prices: np.ndarray) -> float:
        """Calculate mean reversion half-life"""
        try:
            if len(prices) < 20:
                return 10.0
            
            # Ornstein-Uhlenbeck process: dx = θ(μ - x)dt + σdW
            # Half-life = ln(2) / θ
            
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


class MultiTimeframeMomentumStrategy:
    """Advanced momentum strategy across multiple timeframes"""
    
    def __init__(self):
        self.timeframes = {
            TimeFrame.M5: deque(maxlen=100),
            TimeFrame.M15: deque(maxlen=200),
            TimeFrame.H1: deque(maxlen=300)
        }
        
    def update(self, price: float, timeframe: TimeFrame):
        """Update price for specific timeframe"""
        if timeframe in self.timeframes:
            self.timeframes[timeframe].append(price)
    
    def generate_signal(self, current_price: float, indicators: Dict[str, float]) -> AdvancedSignal:
        """Generate multi-timeframe momentum signal"""
        try:
            momentum_scores = {}
            
            # Calculate momentum for each timeframe
            for tf, prices in self.timeframes.items():
                if len(prices) >= 20:
                    momentum_scores[tf] = self._calculate_momentum_score(list(prices))
            
            if not momentum_scores:
                return AdvancedSignal("HOLD", 0.0, TimeFrame.M15, "MTF_Momentum", current_price)
            
            # Weighted average of momentum scores (higher weight for longer timeframes)
            weights = {TimeFrame.M5: 0.2, TimeFrame.M15: 0.3, TimeFrame.H1: 0.5}
            
            total_momentum = 0
            total_weight = 0
            
            for tf, score in momentum_scores.items():
                weight = weights.get(tf, 0.3)
                total_momentum += score * weight
                total_weight += weight
            
            avg_momentum = total_momentum / total_weight if total_weight > 0 else 0
            
            # Generate signal
            if avg_momentum > 0.3:
                confidence = min(0.9, avg_momentum)
                return AdvancedSignal(
                    "BUY", confidence, TimeFrame.M15, "MTF_Momentum", current_price,
                    reasoning=[f"Multi-TF momentum: {avg_momentum:.3f}"],
                    metadata={"momentum_scores": momentum_scores}
                )
            
            elif avg_momentum < -0.3:
                confidence = min(0.9, abs(avg_momentum))
                return AdvancedSignal(
                    "SELL", confidence, TimeFrame.M15, "MTF_Momentum", current_price,
                    reasoning=[f"Multi-TF momentum: {avg_momentum:.3f}"],
                    metadata={"momentum_scores": momentum_scores}
                )
            
            return AdvancedSignal("HOLD", 0.0, TimeFrame.M15, "MTF_Momentum", current_price)
            
        except Exception as e:
            logger.error(f"MTF Momentum error: {e}")
            return AdvancedSignal("HOLD", 0.0, TimeFrame.M15, "MTF_Momentum", current_price)
    
    def _calculate_momentum_score(self, prices: List[float]) -> float:
        """Calculate normalized momentum score (-1 to 1)"""
        try:
            if len(prices) < 10:
                return 0.0
            
            prices_array = np.array(prices)
            
            # Multiple momentum measures
            # 1. Price Rate of Change
            short_roc = (prices_array[-1] - prices_array[-5]) / prices_array[-5] if len(prices) >= 5 else 0
            long_roc = (prices_array[-1] - prices_array[-20]) / prices_array[-20] if len(prices) >= 20 else 0
            
            # 2. Trend strength using linear regression slope
            x = np.arange(len(prices_array))
            slope, _, r_value, _, _ = stats.linregress(x, prices_array)
            trend_strength = slope * r_value  # Slope weighted by R-squared
            
            # 3. Momentum oscillator
            if len(prices) >= 14:
                gains = []
                losses = []
                for i in range(1, len(prices)):
                    change = prices[i] - prices[i-1]
                    if change > 0:
                        gains.append(change)
                        losses.append(0)
                    else:
                        gains.append(0)
                        losses.append(-change)
                
                avg_gain = np.mean(gains[-14:])
                avg_loss = np.mean(losses[-14:])
                
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi_momentum = (rs - 1) / (rs + 1)  # Normalized RSI momentum
                else:
                    rsi_momentum = 1.0
            else:
                rsi_momentum = 0.0
            
            # Combine momentum measures
            momentum_score = (
                0.3 * np.tanh(short_roc * 100) +
                0.3 * np.tanh(long_roc * 50) +
                0.2 * np.tanh(trend_strength * 1000) +
                0.2 * rsi_momentum
            )
            
            return np.clip(momentum_score, -1, 1)
            
        except Exception as e:
            logger.warning(f"Momentum calculation error: {e}")
            return 0.0


class VolatilityBreakoutStrategy:
    """Advanced volatility breakout strategy with dynamic thresholds"""
    
    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        self.price_history = deque(maxlen=lookback)
        self.volume_history = deque(maxlen=lookback)
        
    def update(self, price: float, volume: float):
        """Update with new market data"""
        self.price_history.append(price)
        self.volume_history.append(volume)
    
    def generate_signal(self, current_price: float, indicators: Dict[str, float]) -> AdvancedSignal:
        """Generate volatility breakout signal"""
        if len(self.price_history) < 20:
            return AdvancedSignal("HOLD", 0.0, TimeFrame.M15, "VolBreakout", current_price)
        
        try:
            prices = np.array(self.price_history)
            volumes = np.array(self.volume_history)
            
            # Calculate dynamic volatility bands
            returns = np.diff(np.log(prices))
            current_vol = np.std(returns[-10:]) if len(returns) >= 10 else np.std(returns)
            historical_vol = np.std(returns)
            
            # Volume-weighted average price
            vwap = np.sum(prices * volumes) / np.sum(volumes)
            
            # Bollinger-like bands with volatility adjustment
            vol_multiplier = max(1.5, current_vol / historical_vol * 2)
            upper_band = vwap + vol_multiplier * historical_vol * vwap
            lower_band = vwap - vol_multiplier * historical_vol * vwap
            
            # Volume surge detection
            avg_volume = np.mean(volumes[-20:])
            current_volume = volumes[-1] if len(volumes) > 0 else avg_volume
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Breakout detection
            if current_price > upper_band and volume_ratio > 1.5:
                # Upward breakout with volume confirmation
                confidence = min(0.9, 0.5 + (current_price - upper_band) / upper_band * 10 + volume_ratio * 0.1)
                return AdvancedSignal(
                    "BUY", confidence, TimeFrame.M15, "VolBreakout", current_price,
                    stop_loss=lower_band,
                    take_profit=current_price * (1 + 2 * current_vol),
                    reasoning=[f"Upward breakout, Vol ratio: {volume_ratio:.2f}"],
                    metadata={"breakout_level": upper_band, "volume_surge": volume_ratio}
                )
            
            elif current_price < lower_band and volume_ratio > 1.5:
                # Downward breakout with volume confirmation
                confidence = min(0.9, 0.5 + (lower_band - current_price) / lower_band * 10 + volume_ratio * 0.1)
                return AdvancedSignal(
                    "SELL", confidence, TimeFrame.M15, "VolBreakout", current_price,
                    stop_loss=upper_band,
                    take_profit=current_price * (1 - 2 * current_vol),
                    reasoning=[f"Downward breakout, Vol ratio: {volume_ratio:.2f}"],
                    metadata={"breakout_level": lower_band, "volume_surge": volume_ratio}
                )
            
            return AdvancedSignal("HOLD", 0.0, TimeFrame.M15, "VolBreakout", current_price)
            
        except Exception as e:
            logger.error(f"Vol breakout error: {e}")
            return AdvancedSignal("HOLD", 0.0, TimeFrame.M15, "VolBreakout", current_price)


class AdaptiveDCAStrategy:
    """Sophisticated Dollar Cost Averaging with market regime adaptation"""
    
    def __init__(self, base_interval_hours: int = 24, base_amount: float = 0.001):
        self.base_interval_hours = base_interval_hours
        self.base_amount = base_amount
        self.last_purchase = None
        self.purchase_history = []
        self.market_regime_classifier = MarketRegimeClassifier()
        
    def update(self, price: float, volume: float):
        """Update with market data"""
        self.market_regime_classifier.update(price, volume)
    
    def generate_signal(self, current_price: float, indicators: Dict[str, float]) -> AdvancedSignal:
        """Generate adaptive DCA signal"""
        try:
            current_time = datetime.now()
            
            # Get market regime
            regime_probs = self.market_regime_classifier.classify_regime()
            
            # Adaptive timing based on market conditions
            rsi = indicators.get('rsi', 50)
            volatility = indicators.get('volatility', 0.02)
            
            # Calculate adaptive interval and amount
            interval_multiplier = self._calculate_interval_multiplier(regime_probs, rsi, volatility)
            amount_multiplier = self._calculate_amount_multiplier(regime_probs, rsi, current_price)
            
            adaptive_interval = self.base_interval_hours * interval_multiplier
            adaptive_amount = self.base_amount * amount_multiplier
            
            # Check if it's time to buy
            if self.last_purchase is None:
                should_buy = True
            else:
                time_since_last = (current_time - self.last_purchase).total_seconds() / 3600
                should_buy = time_since_last >= adaptive_interval
            
            if should_buy:
                # Enhanced confidence based on market conditions
                base_confidence = 0.7
                
                # Boost confidence in favorable conditions
                if rsi < 30:  # Oversold
                    base_confidence += 0.15
                elif rsi < 40:
                    base_confidence += 0.1
                
                if regime_probs["mean_reverting"] > 0.5:  # Mean reverting market
                    base_confidence += 0.1
                
                if volatility > 0.05:  # High volatility = potential opportunity
                    base_confidence += 0.05
                
                confidence = min(0.95, base_confidence)
                
                # Update purchase tracking
                self.last_purchase = current_time
                self.purchase_history.append({
                    'timestamp': current_time,
                    'price': current_price,
                    'amount': adaptive_amount,
                    'regime': regime_probs
                })
                
                return AdvancedSignal(
                    "BUY", confidence, TimeFrame.D1, "Adaptive_DCA", current_price,
                    position_size=adaptive_amount,
                    reasoning=[
                        f"Adaptive DCA: RSI={rsi:.1f}, Regime={max(regime_probs, key=regime_probs.get)}",
                        f"Interval: {adaptive_interval:.1f}h, Amount: {amount_multiplier:.2f}x"
                    ],
                    metadata={
                        "regime_probs": regime_probs,
                        "interval_multiplier": interval_multiplier,
                        "amount_multiplier": amount_multiplier
                    }
                )
            
            return AdvancedSignal("HOLD", 0.0, TimeFrame.D1, "Adaptive_DCA", current_price)
            
        except Exception as e:
            logger.error(f"Adaptive DCA error: {e}")
            return AdvancedSignal("HOLD", 0.0, TimeFrame.D1, "Adaptive_DCA", current_price)
    
    def _calculate_interval_multiplier(self, regime_probs: Dict[str, float], rsi: float, volatility: float) -> float:
        """Calculate how to adjust the DCA interval"""
        base_multiplier = 1.0
        
        # Buy more frequently in bear markets (lower RSI)
        if rsi < 30:
            base_multiplier *= 0.5  # Halve the interval (buy twice as often)
        elif rsi < 40:
            base_multiplier *= 0.7
        elif rsi > 70:
            base_multiplier *= 1.5  # Increase interval in overbought conditions
        
        # Adjust for market regime
        if regime_probs["trending"] > 0.6:
            base_multiplier *= 1.2  # Less frequent in trending markets
        elif regime_probs["volatile"] > 0.6:
            base_multiplier *= 0.8  # More frequent in volatile markets
        
        return np.clip(base_multiplier, 0.25, 3.0)
    
    def _calculate_amount_multiplier(self, regime_probs: Dict[str, float], rsi: float, current_price: float) -> float:
        """Calculate how to adjust the DCA amount"""
        base_multiplier = 1.0
        
        # Buy more when oversold
        if rsi < 20:
            base_multiplier *= 2.0
        elif rsi < 30:
            base_multiplier *= 1.5
        elif rsi < 40:
            base_multiplier *= 1.2
        elif rsi > 80:
            base_multiplier *= 0.5
        elif rsi > 70:
            base_multiplier *= 0.7
        
        # Price-based adjustment (buy more when price is lower relative to recent history)
        if len(self.purchase_history) > 0:
            recent_prices = [p['price'] for p in self.purchase_history[-10:]]
            avg_recent_price = np.mean(recent_prices)
            
            if current_price < avg_recent_price * 0.95:
                base_multiplier *= 1.3  # Buy more if price dropped
            elif current_price > avg_recent_price * 1.05:
                base_multiplier *= 0.8  # Buy less if price increased
        
        return np.clip(base_multiplier, 0.5, 3.0)


class AdvancedStrategyEngine:
    """Main engine coordinating all advanced strategies"""
    
    def __init__(self):
        # Initialize all strategies
        self.strategies = {
            'stat_arb': StatisticalArbitrageStrategy(),
            'mtf_momentum': MultiTimeframeMomentumStrategy(),
            'vol_breakout': VolatilityBreakoutStrategy(),
            'adaptive_dca': AdaptiveDCAStrategy()
        }
        
        # Strategy weights (can be optimized)
        self.weights = {
            'stat_arb': 0.3,
            'mtf_momentum': 0.3,
            'vol_breakout': 0.25,
            'adaptive_dca': 0.15
        }
        
        self.performance_tracking = {name: [] for name in self.strategies.keys()}
    
    def update_all_strategies(self, price: float, volume: float, indicators: Dict[str, float]):
        """Update all strategies with new market data"""
        try:
            # Update statistical arbitrage
            self.strategies['stat_arb'].update(price, volume, indicators)
            
            # Update multi-timeframe momentum (assuming 15min timeframe)
            self.strategies['mtf_momentum'].update(price, TimeFrame.M15)
            
            # Update volatility breakout
            self.strategies['vol_breakout'].update(price, volume)
            
            # Update adaptive DCA
            self.strategies['adaptive_dca'].update(price, volume)
            
        except Exception as e:
            logger.error(f"Strategy update error: {e}")
    
    def generate_consensus_signal(self, current_price: float, indicators: Dict[str, float]) -> AdvancedSignal:
        """Generate consensus signal from all strategies"""
        try:
            signals = {}
            
            # Get signals from all strategies
            for name, strategy in self.strategies.items():
                try:
                    signal = strategy.generate_signal(current_price, indicators)
                    signals[name] = signal
                except Exception as e:
                    logger.warning(f"Strategy {name} signal error: {e}")
                    continue
            
            if not signals:
                return AdvancedSignal("HOLD", 0.0, TimeFrame.M15, "Consensus", current_price)
            
            # Calculate weighted consensus
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
            
            # Determine consensus action with more aggressive thresholds
            if buy_score > 0.35 and buy_score > sell_score * 1.2:
                action = "BUY"
                confidence = min(0.95, buy_score + 0.2)
                
                # Calculate position size based on strongest signals
                strongest_buy_signal = max(
                    [s for s in signals.values() if s.action == "BUY"],
                    key=lambda x: x.confidence,
                    default=None
                )
                position_size = strongest_buy_signal.position_size if strongest_buy_signal else 0.001
                
            elif sell_score > 0.35 and sell_score > buy_score * 1.2:
                action = "SELL"
                confidence = min(0.95, sell_score + 0.2)
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
                    "strategy_metadata": combined_metadata
                }
            )
            
            return consensus_signal
            
        except Exception as e:
            logger.error(f"Consensus signal error: {e}")
            return AdvancedSignal("HOLD", 0.0, TimeFrame.M15, "Consensus", current_price)
    
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
    
    def optimize_weights(self, lookback_periods: int = 50):
        """Dynamically optimize strategy weights based on recent performance"""
        try:
            performance = self.get_strategy_performance()
            
            # Calculate performance scores
            scores = {}
            for name, perf in performance.items():
                if perf['total_signals'] > 5:  # Minimum signals for reliability
                    # Combine win rate and average confidence
                    score = perf['win_rate'] * 0.7 + perf['avg_confidence'] * 0.3
                    scores[name] = max(0.1, score)  # Minimum weight
                else:
                    scores[name] = 0.25  # Default weight for new/low-activity strategies
            
            # Normalize to sum to 1
            total_score = sum(scores.values())
            if total_score > 0:
                self.weights = {name: score / total_score for name, score in scores.items()}
            
            logger.info(f"Updated strategy weights: {self.weights}")
            
        except Exception as e:
            logger.error(f"Weight optimization error: {e}")