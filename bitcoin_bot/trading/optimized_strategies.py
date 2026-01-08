# trading/optimized_strategies.py
import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Protocol
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class TradingAction(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

@dataclass
class MarketData:
    """Optimized market data structure."""
    timestamp: datetime
    price: float
    volume: float
    bid: float
    ask: float
    spread: float
    
    # Technical indicators (calculated once, cached)
    rsi: Optional[float] = None
    macd: Optional[float] = None
    signal: Optional[float] = None
    volatility: Optional[float] = None
    
    # Market microstructure
    order_book_imbalance: Optional[float] = None
    volume_weighted_price: Optional[float] = None

@dataclass
class TradingSignal:
    """Enhanced trading signal with execution details."""
    action: TradingAction
    confidence: float
    size: float
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: List[str] = field(default_factory=list)
    urgency: float = 0.5  # 0-1, higher means execute faster
    max_slippage: float = 0.002  # Maximum acceptable slippage
    time_limit: Optional[datetime] = None

class StrategyProtocol(Protocol):
    """Protocol for trading strategies."""
    
    async def analyze(self, market_data: MarketData) -> TradingSignal:
        """Analyze market and generate trading signal."""
        ...
    
    def update_performance(self, signal: TradingSignal, result: Dict) -> None:
        """Update strategy performance metrics."""
        ...

class OptimizedBitcoinAccumulator:
    """Optimized strategy specifically for Bitcoin accumulation through volatility trading."""
    
    def __init__(
        self,
        target_accumulation_rate: float = 0.02,  # 2% more BTC per month
        volatility_threshold: float = 0.03,      # 3% volatility to trigger trading
        max_position_size: float = 0.15,         # 15% max position
        rebalance_threshold: float = 0.05        # 5% deviation triggers rebalance
    ):
        self.target_accumulation_rate = target_accumulation_rate
        self.volatility_threshold = volatility_threshold
        self.max_position_size = max_position_size
        self.rebalance_threshold = rebalance_threshold
        
        # State tracking
        self.btc_accumulated = 0.0
        self.trades_count = 0
        self.last_rebalance = datetime.now()
        self.volatility_history = []
        self.price_history = []
        
        # Performance tracking
        self.monthly_accumulation = 0.0
        self.success_rate = 0.0
        self.average_holding_time = timedelta(hours=4)
    
    async def analyze(self, market_data: MarketData) -> TradingSignal:
        """Analyze market for Bitcoin accumulation opportunities."""
        # Update histories
        self.price_history.append(market_data.price)
        if len(self.price_history) > 100:
            self.price_history.pop(0)
        
        # Calculate current volatility
        if len(self.price_history) >= 20:
            returns = np.diff(np.log(self.price_history[-20:]))
            current_volatility = np.std(returns) * np.sqrt(24 * 365)  # Annualized
            self.volatility_history.append(current_volatility)
            
            if len(self.volatility_history) > 48:  # Keep 48 hours of data
                self.volatility_history.pop(0)
        else:
            current_volatility = 0.02  # Default
        
        # Determine trading action
        signal = await self._generate_accumulation_signal(market_data, current_volatility)
        
        return signal
    
    async def _generate_accumulation_signal(
        self, 
        market_data: MarketData, 
        volatility: float
    ) -> TradingSignal:
        """Generate signal optimized for Bitcoin accumulation."""
        
        confidence = 0.5
        reasoning = []
        
        # High volatility creates opportunities
        if volatility > self.volatility_threshold:
            # Look for mean reversion opportunities
            if market_data.rsi and market_data.rsi < 35:
                # Oversold in volatile market - good accumulation opportunity
                action = TradingAction.BUY
                confidence = min(0.9, 0.6 + (35 - market_data.rsi) / 70)
                reasoning.append(f"Oversold in volatile market (RSI: {market_data.rsi:.1f})")
                reasoning.append(f"High volatility: {volatility:.1%}")
                
                # Aggressive sizing for high conviction
                size = self._calculate_position_size(confidence, volatility, "accumulate")
                
            elif market_data.rsi and market_data.rsi > 75:
                # Overbought - consider taking profits to buy back lower
                action = TradingAction.SELL
                confidence = min(0.85, 0.5 + (market_data.rsi - 75) / 50)
                reasoning.append(f"Overbought profit-taking (RSI: {market_data.rsi:.1f})")
                
                # Conservative sizing for profit-taking
                size = self._calculate_position_size(confidence, volatility, "profit_take")
                
            else:
                action = TradingAction.HOLD
                size = 0.0
                reasoning.append("Waiting for better entry in volatile market")
        
        else:
            # Low volatility - DCA or hold
            time_since_last = datetime.now() - self.last_rebalance
            
            if time_since_last > timedelta(hours=12):  # DCA every 12 hours in low vol
                action = TradingAction.BUY
                confidence = 0.4  # Lower confidence for DCA
                size = self._calculate_position_size(confidence, volatility, "dca")
                reasoning.append("DCA in low volatility period")
                self.last_rebalance = datetime.now()
            else:
                action = TradingAction.HOLD
                size = 0.0
                reasoning.append("DCA cooldown active")
        
        # Enhanced signal with accumulation-specific parameters
        signal = TradingSignal(
            action=action,
            confidence=confidence,
            size=size,
            price=market_data.price,
            reasoning=reasoning,
            urgency=min(0.9, volatility * 10),  # Higher urgency in volatile markets
            max_slippage=min(0.005, volatility / 2),  # Allow more slippage in volatile markets
            time_limit=datetime.now() + timedelta(minutes=max(5, int(30 / max(volatility * 100, 1))))
        )
        
        # Add stop loss and take profit for non-DCA trades
        if action == TradingAction.BUY and confidence > 0.6:
            signal.stop_loss = market_data.price * (1 - min(0.04, volatility * 2))
            signal.take_profit = market_data.price * (1 + min(0.08, volatility * 3))
        
        return signal
    
    def _calculate_position_size(
        self, 
        confidence: float, 
        volatility: float, 
        trade_type: str
    ) -> float:
        """Calculate optimal position size for Bitcoin accumulation."""
        
        base_size = self.max_position_size
        
        # Adjust for trade type
        if trade_type == "accumulate":
            # Aggressive sizing for high-conviction accumulation
            size_multiplier = 0.8 + (confidence * 0.4)  # 0.8x to 1.2x
            
        elif trade_type == "profit_take":
            # Conservative sizing for profit-taking
            size_multiplier = 0.3 + (confidence * 0.3)  # 0.3x to 0.6x
            
        elif trade_type == "dca":
            # Fixed small size for DCA
            size_multiplier = 0.2  # 20% of max position
            
        else:
            size_multiplier = 0.5
        
        # Volatility adjustment
        vol_adjustment = max(0.5, 1.0 - (volatility - 0.02) * 5)  # Reduce size in high volatility
        
        # Confidence scaling
        confidence_scaling = 0.5 + (confidence * 0.5)
        
        final_size = base_size * size_multiplier * vol_adjustment * confidence_scaling
        
        return max(0.01, min(self.max_position_size, final_size))
    
    def update_performance(self, signal: TradingSignal, result: Dict) -> None:
        """Update performance metrics for optimization."""
        self.trades_count += 1
        
        if result.get("success", False):
            btc_gained = result.get("btc_gained", 0)
            self.btc_accumulated += btc_gained
            
            # Update success rate (exponential moving average)
            alpha = 0.1
            self.success_rate = alpha * 1.0 + (1 - alpha) * self.success_rate
        else:
            # Failed trade
            alpha = 0.1
            self.success_rate = alpha * 0.0 + (1 - alpha) * self.success_rate
        
        # Log performance metrics
        if self.trades_count % 10 == 0:
            logger.info(f"Accumulation Performance: {self.btc_accumulated:.6f} BTC gained over {self.trades_count} trades")
            logger.info(f"Success rate: {self.success_rate:.1%}")

class AdaptiveStrategyEngine:
    """Engine that adapts strategy selection based on market conditions and performance."""
    
    def __init__(self):
        self.strategies = {
            "accumulator": OptimizedBitcoinAccumulator(),
            # Could add other strategies here
        }
        
        self.strategy_weights = {"accumulator": 1.0}
        self.performance_history = []
        
        # Market regime detection
        self.current_regime = "normal"
        self.regime_confidence = 0.5
    
    async def generate_signal(self, market_data: MarketData) -> TradingSignal:
        """Generate signal using the best strategy for current conditions."""
        
        # Update market regime
        await self._update_market_regime(market_data)
        
        # Get signals from active strategies
        signals = []
        for name, strategy in self.strategies.items():
            if self.strategy_weights.get(name, 0) > 0:
                try:
                    signal = await strategy.analyze(market_data)
                    signal.confidence *= self.strategy_weights[name]
                    signals.append((name, signal))
                except Exception as e:
                    logger.error(f"Strategy {name} failed: {e}")
        
        if not signals:
            # Fallback signal
            return TradingSignal(
                action=TradingAction.HOLD,
                confidence=0.0,
                size=0.0,
                price=market_data.price,
                reasoning=["No active strategies available"]
            )
        
        # For now, just use the accumulator strategy
        # In a more complex system, you might combine signals
        return signals[0][1]
    
    async def _update_market_regime(self, market_data: MarketData):
        """Update market regime classification."""
        # Simplified regime detection based on volatility
        if market_data.volatility:
            if market_data.volatility > 0.05:
                self.current_regime = "high_volatility"
                self.regime_confidence = min(0.9, market_data.volatility * 10)
            elif market_data.volatility < 0.02:
                self.current_regime = "low_volatility"
                self.regime_confidence = min(0.9, (0.02 - market_data.volatility) * 20)
            else:
                self.current_regime = "normal"
                self.regime_confidence = 0.6
    
    def update_strategy_performance(
        self, 
        strategy_name: str, 
        signal: TradingSignal, 
        result: Dict
    ):
        """Update performance and potentially adjust strategy weights."""
        
        if strategy_name in self.strategies:
            self.strategies[strategy_name].update_performance(signal, result)
        
        # Store performance for adaptation
        self.performance_history.append({
            "timestamp": datetime.now(),
            "strategy": strategy_name,
            "signal": signal,
            "result": result,
            "regime": self.current_regime
        })
        
        # Adaptive weight adjustment (simplified)
        if len(self.performance_history) >= 20:
            self._adapt_strategy_weights()
    
    def _adapt_strategy_weights(self):
        """Adapt strategy weights based on recent performance."""
        # This is a simplified version - could be much more sophisticated
        recent_performance = self.performance_history[-20:]
        
        strategy_scores = {}
        for perf in recent_performance:
            strategy = perf["strategy"]
            success = perf["result"].get("success", False)
            
            if strategy not in strategy_scores:
                strategy_scores[strategy] = {"successes": 0, "total": 0}
            
            strategy_scores[strategy]["total"] += 1
            if success:
                strategy_scores[strategy]["successes"] += 1
        
        # Update weights based on success rates
        for strategy, score in strategy_scores.items():
            if score["total"] > 0:
                success_rate = score["successes"] / score["total"]
                # Exponential moving average of weights
                alpha = 0.1
                current_weight = self.strategy_weights.get(strategy, 0.5)
                new_weight = alpha * success_rate + (1 - alpha) * current_weight
                self.strategy_weights[strategy] = max(0.1, min(1.0, new_weight))
        
        logger.info(f"Updated strategy weights: {self.strategy_weights}")