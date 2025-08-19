# bitcoin_bot/trading/strategies.py
"""
Trading Strategy Implementations
Modular strategies that can be plugged into the main trading bot
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

logger = logging.getLogger(__name__)

# Import shared components
from core.bot import TradingAction, RiskLevel, MarketIndicators, TradingSignal

class StrategyType(Enum):
    """Available strategy types"""
    DCA = "dca"  # Dollar Cost Averaging
    GRID = "grid"  # Grid Trading
    MOMENTUM = "momentum"  # Momentum Following
    MEAN_REVERSION = "mean_reversion"  # Mean Reversion
    BREAKOUT = "breakout"  # Breakout Trading
    SCALPING = "scalping"  # High-frequency Scalping
    HYBRID = "hybrid"  # Hybrid Adaptive
    ML_ENHANCED = "ml_enhanced"  # Machine Learning Enhanced

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

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies
    """
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize base strategy
        
        Args:
            config: Strategy configuration
        """
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
        """
        Analyze market conditions for this strategy
        
        Args:
            indicators: Current market indicators
            
        Returns:
            Strategy-specific analysis results
        """
        pass
    
    @abstractmethod
    def generate_signal(
        self, 
        indicators: MarketIndicators,
        analysis: Dict[str, Any]
    ) -> TradingSignal:
        """
        Generate trading signal based on strategy logic
        
        Args:
            indicators: Current market indicators
            analysis: Results from analyze()
            
        Returns:
            Trading signal
        """
        pass
    
    def should_trade(self, indicators: MarketIndicators) -> Tuple[bool, str]:
        """
        Check if strategy should trade based on current conditions
        
        Args:
            indicators: Current market indicators
            
        Returns:
            Tuple of (should_trade, reason)
        """
        # Check if strategy is enabled
        if not self.config.enabled:
            return False, "Strategy disabled"
        
        # Check minimum time between trades
        if self.state.last_action_time:
            time_since_last = (datetime.now() - self.state.last_action_time).total_seconds()
            if time_since_last < self.config.min_time_between_orders:
                return False, f"Cooldown active ({time_since_last:.0f}s)"
        
        return True, "OK"
    
    def calculate_position_size(
        self,
        action: TradingAction,
        indicators: MarketIndicators,
        confidence: float,
        available_balance: float
    ) -> float:
        """
        Calculate position size for the trade
        
        Args:
            action: Trading action
            indicators: Market indicators
            confidence: Signal confidence
            available_balance: Available balance for trading
            
        Returns:
            Position size
        """
        if action == TradingAction.HOLD:
            return 0.0
        
        # Base position size
        base_size = self.config.base_order_size * available_balance
        
        # Apply confidence scaling if enabled
        if self.config.position_scaling:
            base_size *= (0.5 + confidence * 0.5)  # Scale from 50% to 100%
        
        # Apply volatility adjustment
        volatility_mult = 1.0
        if indicators.volatility > 0.05:  # High volatility
            volatility_mult = 0.7
        elif indicators.volatility < 0.02:  # Low volatility
            volatility_mult = 1.2
        
        position_size = base_size * volatility_mult
        
        # Apply maximum position size limit
        max_size = self.config.max_position_size * available_balance
        position_size = min(position_size, max_size)
        
        return position_size
    
    def update_state(self, signal: TradingSignal, executed: bool):
        """
        Update strategy state after signal generation
        
        Args:
            signal: Generated trading signal
            executed: Whether the trade was executed
        """
        self.state.last_action = signal.action
        self.state.last_action_time = datetime.now()
        self.state.last_action_price = signal.price
        
        if executed and signal.action != TradingAction.HOLD:
            trade_record = {
                'timestamp': signal.timestamp.isoformat(),
                'action': signal.action.value,
                'price': signal.price,
                'volume': signal.volume,
                'confidence': signal.confidence
            }
            
            self.state.trade_history.append(trade_record)
            self.total_trades += 1
            
            # Update position tracking
            if signal.action == TradingAction.BUY:
                self.state.position_count += 1
                self.state.total_invested += signal.volume * signal.price
                
                # Update average entry price
                if self.state.position_count > 0:
                    self.state.average_entry_price = (
                        self.state.total_invested / self.state.position_count
                    )
            
            elif signal.action == TradingAction.SELL:
                # Calculate PnL if we have entry price
                if self.state.average_entry_price > 0:
                    pnl = (signal.price - self.state.average_entry_price) * signal.volume
                    self.total_pnl += pnl
                    
                    if pnl > 0:
                        self.winning_trades += 1
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics"""
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'average_entry_price': self.state.average_entry_price,
            'position_count': self.state.position_count
        }
    
    def reset(self):
        """Reset strategy state"""
        self.state = StrategyState()
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.logger.info("Strategy state reset")

class DCAStrategy(BaseStrategy):
    """
    Dollar Cost Averaging Strategy
    Accumulates Bitcoin at regular intervals regardless of price
    """
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        
        # DCA specific parameters
        self.buy_interval_hours = config.params.get('buy_interval_hours', 24)
        self.buy_amount_percentage = config.params.get('buy_amount_percentage', 0.10)
        self.price_threshold = config.params.get('price_threshold', None)
        self.accumulation_target = config.params.get('accumulation_target', None)
        
        # Initialize state
        self.state.custom_state['last_dca_buy'] = None
        self.state.custom_state['total_accumulated'] = 0.0
    
    def analyze(self, indicators: MarketIndicators) -> Dict[str, Any]:
        """Analyze market for DCA strategy"""
        analysis = {
            'time_since_last_buy': float('inf'),
            'should_buy': False,
            'price_below_threshold': True,
            'accumulation_complete': False
        }
        
        # Check time since last DCA buy
        last_buy = self.state.custom_state.get('last_dca_buy')
        if last_buy:
            time_since = (datetime.now() - last_buy).total_seconds() / 3600
            analysis['time_since_last_buy'] = time_since
            analysis['should_buy'] = time_since >= self.buy_interval_hours
        else:
            analysis['should_buy'] = True
        
        # Check price threshold if set
        if self.price_threshold:
            analysis['price_below_threshold'] = indicators.current_price <= self.price_threshold
        
        # Check accumulation target
        if self.accumulation_target:
            total = self.state.custom_state.get('total_accumulated', 0)
            analysis['accumulation_complete'] = total >= self.accumulation_target
        
        return analysis
    
    def generate_signal(
        self,
        indicators: MarketIndicators,
        analysis: Dict[str, Any]
    ) -> TradingSignal:
        """Generate DCA trading signal"""
        action = TradingAction.HOLD
        confidence = 0.5
        reasoning = []
        
        # Check if we should buy
        if (analysis['should_buy'] and 
            analysis['price_below_threshold'] and 
            not analysis['accumulation_complete']):
            
            action = TradingAction.BUY
            confidence = 0.8  # High confidence for DCA
            reasoning.append(f"DCA interval reached ({self.buy_interval_hours}h)")
            
            if self.price_threshold:
                reasoning.append(f"Price below threshold (€{self.price_threshold:.0f})")
        
        # Never sell in pure DCA strategy
        # (unless you want to add rebalancing logic)
        
        return TradingSignal(
            action=action,
            confidence=confidence,
            volume=0.0,  # Will be calculated separately
            price=indicators.current_price,
            reasoning=reasoning,
            risk_level=RiskLevel.LOW  # DCA is low risk
        )
    
    def update_state(self, signal: TradingSignal, executed: bool):
        """Update DCA specific state"""
        super().update_state(signal, executed)
        
        if executed and signal.action == TradingAction.BUY:
            self.state.custom_state['last_dca_buy'] = datetime.now()
            self.state.custom_state['total_accumulated'] = (
                self.state.custom_state.get('total_accumulated', 0) + signal.volume
            )

class GridStrategy(BaseStrategy):
    """
    Grid Trading Strategy
    Places buy and sell orders at regular price intervals
    """
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        
        # Grid parameters
        self.grid_levels = config.params.get('grid_levels', 10)
        self.grid_spacing_percentage = config.params.get('grid_spacing_percentage', 0.02)
        self.grid_base_price = config.params.get('grid_base_price', None)
        
        # Initialize grid
        self.state.custom_state['grid_orders'] = []
        self.state.custom_state['active_grids'] = {}
    
    def analyze(self, indicators: MarketIndicators) -> Dict[str, Any]:
        """Analyze market for grid trading"""
        current_price = indicators.current_price
        
        # Set base price if not set
        if not self.grid_base_price:
            self.grid_base_price = current_price
            self.state.custom_state['grid_base_price'] = current_price
        
        # Calculate grid levels
        grid_levels = []
        for i in range(-self.grid_levels // 2, self.grid_levels // 2 + 1):
            level_price = self.grid_base_price * (1 + i * self.grid_spacing_percentage)
            grid_levels.append(level_price)
        
        # Find nearest grid levels
        nearest_below = max([p for p in grid_levels if p < current_price], default=None)
        nearest_above = min([p for p in grid_levels if p > current_price], default=None)
        
        # Check active grids
        active_grids = self.state.custom_state.get('active_grids', {})
        
        return {
            'grid_levels': grid_levels,
            'nearest_buy_level': nearest_below,
            'nearest_sell_level': nearest_above,
            'active_buy_grids': [p for p, t in active_grids.items() if t == 'buy'],
            'active_sell_grids': [p for p, t in active_grids.items() if t == 'sell'],
            'price_in_range': nearest_below is not None and nearest_above is not None
        }
    
    def generate_signal(
        self,
        indicators: MarketIndicators,
        analysis: Dict[str, Any]
    ) -> TradingSignal:
        """Generate grid trading signal"""
        action = TradingAction.HOLD
        confidence = 0.7
        reasoning = []
        target_price = indicators.current_price
        
        # Check if we should place a buy order
        if analysis['nearest_buy_level']:
            buy_level = analysis['nearest_buy_level']
            
            # Check if we already have an order at this level
            if buy_level not in analysis['active_buy_grids']:
                if indicators.current_price <= buy_level * 1.01:  # Within 1% of grid level
                    action = TradingAction.BUY
                    target_price = buy_level
                    reasoning.append(f"Grid buy level at €{buy_level:.0f}")
        
        # Check if we should place a sell order
        if analysis['nearest_sell_level'] and action == TradingAction.HOLD:
            sell_level = analysis['nearest_sell_level']
            
            # Check if we already have an order at this level
            if sell_level not in analysis['active_sell_grids']:
                if indicators.current_price >= sell_level * 0.99:  # Within 1% of grid level
                    action = TradingAction.SELL
                    target_price = sell_level
                    reasoning.append(f"Grid sell level at €{sell_level:.0f}")
        
        return TradingSignal(
            action=action,
            confidence=confidence,
            volume=0.0,
            price=target_price,
            reasoning=reasoning,
            risk_level=RiskLevel.MEDIUM
        )

class MomentumStrategy(BaseStrategy):
    """
    Momentum Trading Strategy
    Follows strong price movements and trends
    """
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        
        # Momentum parameters
        self.momentum_period = config.params.get('momentum_period', 20)
        self.momentum_threshold = config.params.get('momentum_threshold', 0.02)
        self.volume_confirmation = config.params.get('volume_confirmation', True)
        self.trend_filter = config.params.get('trend_filter', True)
        
        # Price history for momentum calculation
        self.state.custom_state['price_history'] = deque(maxlen=self.momentum_period * 2)
        self.state.custom_state['volume_history'] = deque(maxlen=self.momentum_period * 2)
    
    def analyze(self, indicators: MarketIndicators) -> Dict[str, Any]:
        """Analyze momentum indicators"""
        # Update price history
        price_history = self.state.custom_state.get('price_history', deque(maxlen=self.momentum_period * 2))
        volume_history = self.state.custom_state.get('volume_history', deque(maxlen=self.momentum_period * 2))
        
        price_history.append(indicators.current_price)
        volume_history.append(indicators.current_volume)
        
        analysis = {
            'momentum': 0.0,
            'momentum_direction': 'neutral',
            'volume_surge': False,
            'trend_aligned': True,
            'strength': 0.0
        }
        
        if len(price_history) >= self.momentum_period:
            # Calculate momentum
            recent_prices = list(price_history)[-self.momentum_period:]
            older_prices = list(price_history)[-self.momentum_period*2:-self.momentum_period]
            
            if older_prices:
                recent_avg = np.mean(recent_prices)
                older_avg = np.mean(older_prices)
                momentum = (recent_avg - older_avg) / older_avg
                
                analysis['momentum'] = momentum
                
                if momentum > self.momentum_threshold:
                    analysis['momentum_direction'] = 'bullish'
                elif momentum < -self.momentum_threshold:
                    analysis['momentum_direction'] = 'bearish'
                
                # Calculate momentum strength
                analysis['strength'] = min(1.0, abs(momentum) / (self.momentum_threshold * 3))
            
            # Check volume surge
            if len(volume_history) >= self.momentum_period:
                recent_volume = np.mean(list(volume_history)[-5:])
                avg_volume = np.mean(list(volume_history))
                analysis['volume_surge'] = recent_volume > avg_volume * 1.5
            
            # Check trend alignment
            if self.trend_filter:
                analysis['trend_aligned'] = (
                    (analysis['momentum_direction'] == 'bullish' and indicators.market_regime == 'uptrend') or
                    (analysis['momentum_direction'] == 'bearish' and indicators.market_regime == 'downtrend')
                )
        
        return analysis
    
    def generate_signal(
        self,
        indicators: MarketIndicators,
        analysis: Dict[str, Any]
    ) -> TradingSignal:
        """Generate momentum trading signal"""
        action = TradingAction.HOLD
        confidence = 0.5
        reasoning = []
        
        # Check for bullish momentum
        if analysis['momentum_direction'] == 'bullish':
            conditions_met = True
            
            if self.volume_confirmation and not analysis['volume_surge']:
                conditions_met = False
                reasoning.append("Waiting for volume confirmation")
            
            if self.trend_filter and not analysis['trend_aligned']:
                conditions_met = False
                reasoning.append("Momentum not aligned with trend")
            
            if conditions_met:
                action = TradingAction.BUY
                confidence = 0.5 + analysis['strength'] * 0.4
                reasoning.append(f"Bullish momentum: {analysis['momentum']:.1%}")
                if analysis['volume_surge']:
                    reasoning.append("Volume surge confirmed")
        
        # Check for bearish momentum (for selling)
        elif analysis['momentum_direction'] == 'bearish':
            conditions_met = True
            
            if self.trend_filter and not analysis['trend_aligned']:
                conditions_met = False
            
            if conditions_met:
                action = TradingAction.SELL
                confidence = 0.5 + analysis['strength'] * 0.4
                reasoning.append(f"Bearish momentum: {analysis['momentum']:.1%}")
        
        # Additional confirmation from technical indicators
        if action == TradingAction.BUY and indicators.rsi > 70:
            action = TradingAction.HOLD
            reasoning.append("RSI overbought - avoiding chase")
        elif action == TradingAction.SELL and indicators.rsi < 30:
            action = TradingAction.HOLD
            reasoning.append("RSI oversold - avoiding panic sell")
        
        return TradingSignal(
            action=action,
            confidence=confidence,
            volume=0.0,
            price=indicators.current_price,
            reasoning=reasoning,
            risk_level=RiskLevel.MEDIUM if confidence > 0.7 else RiskLevel.HIGH
        )

class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy
    Trades on the assumption that prices will revert to their mean
    """
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        
        # Mean reversion parameters
        self.lookback_period = config.params.get('lookback_period', 50)
        self.deviation_threshold = config.params.get('deviation_threshold', 2.0)  # Standard deviations
        self.use_bollinger = config.params.get('use_bollinger', True)
        self.use_rsi = config.params.get('use_rsi', True)
        self.rsi_oversold = config.params.get('rsi_oversold', 30)
        self.rsi_overbought = config.params.get('rsi_overbought', 70)
    
    def analyze(self, indicators: MarketIndicators) -> Dict[str, Any]:
        """Analyze mean reversion opportunities"""
        analysis = {
            'deviation_from_mean': 0.0,
            'z_score': 0.0,
            'bollinger_position': 0.5,
            'rsi_signal': 'neutral',
            'reversion_expected': False,
            'reversion_direction': None
        }
        
        # Calculate deviation from moving average
        if indicators.ma_long > 0:
            deviation = (indicators.current_price - indicators.ma_long) / indicators.ma_long
            analysis['deviation_from_mean'] = deviation
            
            # Calculate z-score if we have Bollinger Bands
            if self.use_bollinger and indicators.bollinger_middle > 0:
                band_width = indicators.bollinger_upper - indicators.bollinger_lower
                if band_width > 0:
                    z_score = (indicators.current_price - indicators.bollinger_middle) / (band_width / 4)
                    analysis['z_score'] = z_score
                    
                    # Bollinger position (0 = lower band, 1 = upper band)
                    analysis['bollinger_position'] = (
                        (indicators.current_price - indicators.bollinger_lower) / band_width
                        if band_width > 0 else 0.5
                    )
        
        # RSI signals
        if self.use_rsi:
            if indicators.rsi < self.rsi_oversold:
                analysis['rsi_signal'] = 'oversold'
            elif indicators.rsi > self.rsi_overbought:
                analysis['rsi_signal'] = 'overbought'
        
        # Determine if reversion is expected
        if abs(analysis['z_score']) > self.deviation_threshold:
            analysis['reversion_expected'] = True
            analysis['reversion_direction'] = 'up' if analysis['z_score'] < 0 else 'down'
        
        return analysis
    
    def generate_signal(
        self,
        indicators: MarketIndicators,
        analysis: Dict[str, Any]
    ) -> TradingSignal:
        """Generate mean reversion signal"""
        action = TradingAction.HOLD
        confidence = 0.5
        reasoning = []
        
        if analysis['reversion_expected']:
            # Buy signal - price below mean and expected to revert up
            if analysis['reversion_direction'] == 'up':
                buy_conditions = [
                    analysis['z_score'] < -self.deviation_threshold,
                    analysis['bollinger_position'] < 0.2 if self.use_bollinger else True,
                    analysis['rsi_signal'] == 'oversold' if self.use_rsi else True
                ]
                
                if all(buy_conditions):
                    action = TradingAction.BUY
                    confidence = min(0.9, 0.5 + abs(analysis['z_score']) * 0.1)
                    reasoning.append(f"Price {analysis['deviation_from_mean']:.1%} below mean")
                    reasoning.append(f"Z-score: {analysis['z_score']:.2f}")
                    if analysis['rsi_signal'] == 'oversold':
                        reasoning.append(f"RSI oversold: {indicators.rsi:.1f}")
            
            # Sell signal - price above mean and expected to revert down
            elif analysis['reversion_direction'] == 'down':
                sell_conditions = [
                    analysis['z_score'] > self.deviation_threshold,
                    analysis['bollinger_position'] > 0.8 if self.use_bollinger else True,
                    analysis['rsi_signal'] == 'overbought' if self.use_rsi else True
                ]
                
                if all(sell_conditions):
                    action = TradingAction.SELL
                    confidence = min(0.9, 0.5 + abs(analysis['z_score']) * 0.1)
                    reasoning.append(f"Price {analysis['deviation_from_mean']:.1%} above mean")
                    reasoning.append(f"Z-score: {analysis['z_score']:.2f}")
                    if analysis['rsi_signal'] == 'overbought':
                        reasoning.append(f"RSI overbought: {indicators.rsi:.1f}")
        
        return TradingSignal(
            action=action,
            confidence=confidence,
            volume=0.0,
            price=indicators.current_price,
            reasoning=reasoning,
            risk_level=RiskLevel.MEDIUM
        )

class BreakoutStrategy(BaseStrategy):
    """
    Breakout Trading Strategy
    Trades on price breaking through key resistance/support levels
    """
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        
        # Breakout parameters
        self.lookback_period = config.params.get('lookback_period', 100)
        self.breakout_confirmation_bars = config.params.get('confirmation_bars', 3)
        self.volume_increase_threshold = config.params.get('volume_increase', 1.5)
        self.use_atr = config.params.get('use_atr', True)
        
        # Track price levels
        self.state.custom_state['resistance_levels'] = []
        self.state.custom_state['support_levels'] = []
        self.state.custom_state['breakout_attempts'] = deque(maxlen=10)
    
    def _identify_levels(self, prices: List[float]) -> Tuple[List[float], List[float]]:
        """Identify support and resistance levels"""
        if len(prices) < 20:
            return [], []
        
        resistance_levels = []
        support_levels = []
        
        # Simple peak and trough detection
        for i in range(2, len(prices) - 2):
            # Resistance (local high)
            if (prices[i] > prices[i-1] and prices[i] > prices[i-2] and
                prices[i] > prices[i+1] and prices[i] > prices[i+2]):
                resistance_levels.append(prices[i])
            
            # Support (local low)
            if (prices[i] < prices[i-1] and prices[i] < prices[i-2] and
                prices[i] < prices[i+1] and prices[i] < prices[i+2]):
                support_levels.append(prices[i])
        
        # Keep only significant levels (cluster similar prices)
        def cluster_levels(levels: List[float], threshold: float = 0.02) -> List[float]:
            if not levels:
                return []
            
            levels = sorted(levels)
            clustered = [levels[0]]
            
            for level in levels[1:]:
                if (level - clustered[-1]) / clustered[-1] > threshold:
                    clustered.append(level)
            
            return clustered
        
        return cluster_levels(support_levels), cluster_levels(resistance_levels)
    
    def analyze(self, indicators: MarketIndicators) -> Dict[str, Any]:
        """Analyze for breakout opportunities"""
        analysis = {
            'nearest_resistance': None,
            'nearest_support': None,
            'breakout_type': None,
            'breakout_strength': 0.0,
            'volume_confirmed': False,
            'consolidation_detected': False
        }
        
        # Get price history (would need to be passed in or stored)
        price_history = self.state.custom_state.get('price_history', [indicators.current_price])
        volume_history = self.state.custom_state.get('volume_history', [indicators.current_volume])
        
        if len(price_history) >= 20:
            # Identify levels
            support, resistance = self._identify_levels(price_history[-self.lookback_period:])
            
            # Find nearest levels
            current_price = indicators.current_price
            
            if resistance:
                above_current = [r for r in resistance if r > current_price]
                if above_current:
                    analysis['nearest_resistance'] = min(above_current)
            
            if support:
                below_current = [s for s in support if s < current_price]
                if below_current:
                    analysis['nearest_support'] = max(below_current)
            
            # Check for breakout
            if analysis['nearest_resistance']:
                if current_price > analysis['nearest_resistance'] * 1.01:  # 1% above resistance
                    analysis['breakout_type'] = 'resistance'
                    analysis['breakout_strength'] = (
                        (current_price - analysis['nearest_resistance']) / analysis['nearest_resistance']
                    )
            
            if analysis['nearest_support']:
                if current_price < analysis['nearest_support'] * 0.99:  # 1% below support
                    analysis['breakout_type'] = 'support'
                    analysis['breakout_strength'] = (
                        (analysis['nearest_support'] - current_price) / analysis['nearest_support']
                    )
            
            # Volume confirmation
            if len(volume_history) >= 10:
                recent_volume = np.mean(volume_history[-3:])
                avg_volume = np.mean(volume_history[-20:])
                analysis['volume_confirmed'] = recent_volume > avg_volume * self.volume_increase_threshold
            
            # Detect consolidation (low volatility before breakout)
            if len(price_history) >= 20:
                recent_volatility = np.std(price_history[-10:]) / np.mean(price_history[-10:])
                older_volatility = np.std(price_history[-20:-10]) / np.mean(price_history[-20:-10])
                analysis['consolidation_detected'] = recent_volatility < older_volatility * 0.7
        
        return analysis
    
    def generate_signal(
        self,
        indicators: MarketIndicators,
        analysis: Dict[str, Any]
    ) -> TradingSignal:
        """Generate breakout trading signal"""
        action = TradingAction.HOLD
        confidence = 0.5
        reasoning = []
        
        if analysis['breakout_type']:
            # Resistance breakout - buy signal
            if analysis['breakout_type'] == 'resistance':
                if analysis['volume_confirmed']:
                    action = TradingAction.BUY
                    confidence = min(0.9, 0.6 + analysis['breakout_strength'] * 2)
                    reasoning.append(f"Resistance breakout at €{analysis['nearest_resistance']:.0f}")
                    reasoning.append("Volume surge confirmed")
                    
                    if analysis['consolidation_detected']:
                        confidence = min(0.95, confidence + 0.1)
                        reasoning.append("Consolidation before breakout")
                else:
                    reasoning.append("Breakout needs volume confirmation")
            
            # Support breakdown - sell signal
            elif analysis['breakout_type'] == 'support':
                if analysis['volume_confirmed']:
                    action = TradingAction.SELL
                    confidence = min(0.9, 0.6 + analysis['breakout_strength'] * 2)
                    reasoning.append(f"Support breakdown at €{analysis['nearest_support']:.0f}")
                    reasoning.append("Volume increase confirmed")
                else:
                    reasoning.append("Breakdown needs volume confirmation")
        
        # Risk management - avoid false breakouts
        if action != TradingAction.HOLD:
            # Check if we've had recent failed breakouts
            recent_attempts = self.state.custom_state.get('breakout_attempts', deque(maxlen=10))
            failed_attempts = sum(1 for attempt in recent_attempts if not attempt['successful'])
            
            if failed_attempts >= 3:
                confidence *= 0.7
                reasoning.append("Caution: Recent false breakouts")
        
        return TradingSignal(
            action=action,
            confidence=confidence,
            volume=0.0,
            price=indicators.current_price,
            reasoning=reasoning,
            risk_level=RiskLevel.HIGH if analysis['breakout_type'] else RiskLevel.MEDIUM
        )

class HybridAdaptiveStrategy(BaseStrategy):
    """
    Hybrid Adaptive Strategy
    Combines multiple strategies and adapts based on market conditions
    """
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        
        # Initialize sub-strategies
        self.strategies = {
            'momentum': MomentumStrategy(self._create_sub_config('momentum')),
            'mean_reversion': MeanReversionStrategy(self._create_sub_config('mean_reversion')),
            'breakout': BreakoutStrategy(self._create_sub_config('breakout'))
        }
        
        # Strategy weights (can be adaptive)
        self.state.custom_state['strategy_weights'] = {
            'momentum': 0.33,
            'mean_reversion': 0.33,
            'breakout': 0.34
        }
        
        # Performance tracking for adaptation
        self.state.custom_state['strategy_performance'] = {
            name: {'trades': 0, 'wins': 0, 'total_pnl': 0.0}
            for name in self.strategies.keys()
        }
    
    def _create_sub_config(self, strategy_name: str) -> StrategyConfig:
        """Create configuration for sub-strategy"""
        sub_config = StrategyConfig(
            name=f"{self.config.name}_{strategy_name}",
            enabled=True,
            max_position_size=self.config.max_position_size * 0.5,  # Reduce for each sub-strategy
            stop_loss_enabled=self.config.stop_loss_enabled,
            stop_loss_percentage=self.config.stop_loss_percentage,
            take_profit_enabled=self.config.take_profit_enabled,
            take_profit_percentage=self.config.take_profit_percentage
        )
        return sub_config
    
    def _adapt_weights(self):
        """Adapt strategy weights based on performance"""
        performances = self.state.custom_state['strategy_performance']
        
        # Calculate win rates
        win_rates = {}
        for name, perf in performances.items():
            if perf['trades'] > 0:
                win_rates[name] = perf['wins'] / perf['trades']
            else:
                win_rates[name] = 0.5  # Default
        
        # Adjust weights based on win rates
        total_win_rate = sum(win_rates.values())
        if total_win_rate > 0:
            for name in self.strategies.keys():
                self.state.custom_state['strategy_weights'][name] = (
                    win_rates[name] / total_win_rate
                )
    
    def analyze(self, indicators: MarketIndicators) -> Dict[str, Any]:
        """Analyze using all sub-strategies"""
        analysis = {
            'sub_analyses': {},
            'market_condition': self._determine_market_condition(indicators),
            'dominant_strategy': None
        }
        
        # Run analysis for each sub-strategy
        for name, strategy in self.strategies.items():
            analysis['sub_analyses'][name] = strategy.analyze(indicators)
        
        # Determine dominant strategy based on market condition
        if indicators.market_regime == 'uptrend':
            analysis['dominant_strategy'] = 'momentum'
        elif indicators.market_regime == 'ranging':
            analysis['dominant_strategy'] = 'mean_reversion'
        else:
            analysis['dominant_strategy'] = 'breakout'
        
        return analysis
    
    def _determine_market_condition(self, indicators: MarketIndicators) -> str:
        """Determine overall market condition"""
        conditions = []
        
        # Trend
        if indicators.market_regime in ['uptrend', 'downtrend']:
            conditions.append('trending')
        else:
            conditions.append('ranging')
        
        # Volatility
        if indicators.volatility > 0.05:
            conditions.append('high_volatility')
        elif indicators.volatility < 0.02:
            conditions.append('low_volatility')
        else:
            conditions.append('normal_volatility')
        
        # Risk
        if indicators.risk_off_probability > 0.6:
            conditions.append('risk_off')
        else:
            conditions.append('risk_on')
        
        return '_'.join(conditions)
    
    def generate_signal(
        self,
        indicators: MarketIndicators,
        analysis: Dict[str, Any]
    ) -> TradingSignal:
        """Generate hybrid signal combining all strategies"""
        weights = self.state.custom_state['strategy_weights']
        
        # Collect signals from all strategies
        signals = {}
        for name, strategy in self.strategies.items():
            sub_analysis = analysis['sub_analyses'][name]
            signals[name] = strategy.generate_signal(indicators, sub_analysis)
        
        # Weighted voting
        buy_score = 0.0
        sell_score = 0.0
        hold_score = 0.0
        combined_reasoning = []
        
        for name, signal in signals.items():
            weight = weights[name]
            
            # Boost weight for dominant strategy
            if name == analysis['dominant_strategy']:
                weight *= 1.5
            
            if signal.action == TradingAction.BUY:
                buy_score += weight * signal.confidence
                combined_reasoning.extend([f"[{name}] {r}" for r in signal.reasoning[:1]])
            elif signal.action == TradingAction.SELL:
                sell_score += weight * signal.confidence
                combined_reasoning.extend([f"[{name}] {r}" for r in signal.reasoning[:1]])
            else:
                hold_score += weight * 0.5
        
        # Determine final action
        max_score = max(buy_score, sell_score, hold_score)
        
        if max_score == buy_score and buy_score > 0.5:
            action = TradingAction.BUY
            confidence = min(0.95, buy_score)
        elif max_score == sell_score and sell_score > 0.5:
            action = TradingAction.SELL
            confidence = min(0.95, sell_score)
        else:
            action = TradingAction.HOLD
            confidence = 0.5
            combined_reasoning = ["Insufficient consensus among strategies"]
        
        # Adapt weights periodically
        if self.total_trades > 0 and self.total_trades % 10 == 0:
            self._adapt_weights()
        
        return TradingSignal(
            action=action,
            confidence=confidence,
            volume=0.0,
            price=indicators.current_price,
            reasoning=combined_reasoning[:3],
            risk_level=RiskLevel.MEDIUM
        )

class StrategyFactory:
    """
    Factory class for creating and managing strategies
    """
    
    @staticmethod
    def create_strategy(strategy_type: StrategyType, config: Optional[StrategyConfig] = None) -> BaseStrategy:
        """
        Create a strategy instance
        
        Args:
            strategy_type: Type of strategy to create
            config: Strategy configuration (uses defaults if None)
            
        Returns:
            Strategy instance
        """
        # Default configurations for each strategy
        default_configs = {
            StrategyType.DCA: StrategyConfig(
                name="DCA Strategy",
                params={
                    'buy_interval_hours': 24,
                    'buy_amount_percentage': 0.10
                }
            ),
            StrategyType.GRID: StrategyConfig(
                name="Grid Strategy",
                params={
                    'grid_levels': 10,
                    'grid_spacing_percentage': 0.02
                }
            ),
            StrategyType.MOMENTUM: StrategyConfig(
                name="Momentum Strategy",
                params={
                    'momentum_period': 20,
                    'momentum_threshold': 0.02
                }
            ),
            StrategyType.MEAN_REVERSION: StrategyConfig(
                name="Mean Reversion Strategy",
                params={
                    'lookback_period': 50,
                    'deviation_threshold': 2.0
                }
            ),
            StrategyType.BREAKOUT: StrategyConfig(
                name="Breakout Strategy",
                params={
                    'lookback_period': 100,
                    'confirmation_bars': 3
                }
            ),
            StrategyType.HYBRID: StrategyConfig(
                name="Hybrid Adaptive Strategy"
            )
        }
        
        # Use provided config or default
        if not config:
            config = default_configs.get(strategy_type)
            if not config:
                raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        # Create strategy instance
        strategy_classes = {
            StrategyType.DCA: DCAStrategy,
            StrategyType.GRID: GridStrategy,
            StrategyType.MOMENTUM: MomentumStrategy,
            StrategyType.MEAN_REVERSION: MeanReversionStrategy,
            StrategyType.BREAKOUT: BreakoutStrategy,
            StrategyType.HYBRID: HybridAdaptiveStrategy
        }
        
        strategy_class = strategy_classes.get(strategy_type)
        if not strategy_class:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        return strategy_class(config)
    
    @staticmethod
    def create_from_config_file(config_file: str) -> BaseStrategy:
        """
        Create strategy from configuration file
        
        Args:
            config_file: Path to JSON configuration file
            
        Returns:
            Strategy instance
        """
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        strategy_type = StrategyType(config_data['type'])
        config = StrategyConfig(**config_data['config'])
        
        return StrategyFactory.create_strategy(strategy_type, config)

# Example usage
if __name__ == "__main__":
    # Create different strategies
    dca = StrategyFactory.create_strategy(StrategyType.DCA)
    momentum = StrategyFactory.create_strategy(StrategyType.MOMENTUM)
    hybrid = StrategyFactory.create_strategy(StrategyType.HYBRID)
    
    # Example configuration file format
    example_config = {
        "type": "momentum",
        "config": {
            "name": "Custom Momentum",
            "enabled": True,
            "max_position_size": 0.20,
            "params": {
                "momentum_period": 30,
                "momentum_threshold": 0.03,
                "volume_confirmation": True
            }
        }
    }
    
    print("Strategies module loaded successfully")
