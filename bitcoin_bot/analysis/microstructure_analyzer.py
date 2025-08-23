# analysis/microstructure_analyzer.py
"""
Advanced Market Microstructure Analysis for Institutional Trading
Provides order book analysis, market impact modeling, and execution optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import time
from collections import deque
from utils.logger import logger


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"  
    RANGE_BOUND = "range_bound"
    VOLATILE = "volatile"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"


@dataclass
class OrderBookSnapshot:
    """Order book snapshot with depth analysis"""
    timestamp: float
    bids: List[Tuple[float, float]]  # [(price, volume), ...]
    asks: List[Tuple[float, float]]
    mid_price: float
    spread: float
    bid_volume: float
    ask_volume: float
    imbalance: float


@dataclass
class MicrostructureSignals:
    """Microstructure-based trading signals"""
    regime: MarketRegime
    liquidity_score: float  # 0-1, higher = more liquid
    order_flow_imbalance: float  # -1 to 1, positive = buy pressure
    market_impact_score: float  # Expected impact of trading
    optimal_trade_size: float
    execution_urgency: float  # 0-1, higher = execute faster
    smart_routing_recommendation: str
    confidence: float


class MicrostructureAnalyzer:
    """Advanced market microstructure analysis"""
    
    def __init__(self, lookback_periods: int = 100):
        self.lookback_periods = lookback_periods
        self.order_book_history = deque(maxlen=lookback_periods)
        self.trade_history = deque(maxlen=lookback_periods * 10)
        self.volume_profile = {}
        
        # Calibration parameters (to be optimized)
        self.params = {
            'regime_detection_window': 50,
            'liquidity_depth_levels': 10,
            'impact_coefficient': 0.001,
            'imbalance_threshold': 0.3,
            'volatility_lookback': 20
        }
    
    def update_order_book(self, order_book_data: Dict) -> OrderBookSnapshot:
        """Process and store order book update"""
        try:
            # Parse order book data
            bids = [(float(bid[0]), float(bid[1])) for bid in order_book_data.get('bids', [])]
            asks = [(float(ask[0]), float(ask[1])) for ask in order_book_data.get('asks', [])]
            
            if not bids or not asks:
                return None
            
            # Calculate key metrics
            best_bid = max(bids, key=lambda x: x[0])
            best_ask = min(asks, key=lambda x: x[0])
            
            mid_price = (best_bid[0] + best_ask[0]) / 2
            spread = best_ask[0] - best_bid[0]
            
            # Calculate volume imbalance
            bid_volume = sum(vol for _, vol in bids[:self.params['liquidity_depth_levels']])
            ask_volume = sum(vol for _, vol in asks[:self.params['liquidity_depth_levels']])
            
            total_volume = bid_volume + ask_volume
            imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
            
            snapshot = OrderBookSnapshot(
                timestamp=time.time(),
                bids=bids,
                asks=asks,
                mid_price=mid_price,
                spread=spread,
                bid_volume=bid_volume,
                ask_volume=ask_volume,
                imbalance=imbalance
            )
            
            self.order_book_history.append(snapshot)
            return snapshot
            
        except Exception as e:
            logger.error(f"Order book processing error: {e}")
            return None
    
    def detect_market_regime(self) -> MarketRegime:
        """Detect current market regime using multiple indicators"""
        if len(self.order_book_history) < self.params['regime_detection_window']:
            return MarketRegime.RANGE_BOUND
        
        try:
            # Get recent price and volume data
            recent_snapshots = list(self.order_book_history)[-self.params['regime_detection_window']:]
            prices = [snap.mid_price for snap in recent_snapshots]
            volumes = [snap.bid_volume + snap.ask_volume for snap in recent_snapshots]
            imbalances = [snap.imbalance for snap in recent_snapshots]
            
            # Price momentum
            price_change = (prices[-1] - prices[0]) / prices[0]
            price_volatility = np.std(np.diff(prices)) / np.mean(prices)
            
            # Volume analysis
            avg_volume = np.mean(volumes)
            volume_trend = np.corrcoef(range(len(volumes)), volumes)[0, 1]
            
            # Order flow analysis
            avg_imbalance = np.mean(imbalances)
            imbalance_consistency = np.std(imbalances)
            
            # Regime classification logic
            if price_volatility > 0.02:  # High volatility threshold
                return MarketRegime.VOLATILE
            
            if abs(price_change) > 0.005:  # Trending threshold
                if price_change > 0:
                    if avg_imbalance > 0.2 and volume_trend > 0:
                        return MarketRegime.TRENDING_UP
                    else:
                        return MarketRegime.DISTRIBUTION
                else:
                    if avg_imbalance < -0.2 and volume_trend > 0:
                        return MarketRegime.TRENDING_DOWN
                    else:
                        return MarketRegime.ACCUMULATION
            
            return MarketRegime.RANGE_BOUND
            
        except Exception as e:
            logger.error(f"Regime detection error: {e}")
            return MarketRegime.RANGE_BOUND
    
    def calculate_liquidity_score(self) -> float:
        """Calculate current market liquidity score (0-1)"""
        if not self.order_book_history:
            return 0.5
        
        try:
            snapshot = self.order_book_history[-1]
            
            # Factors affecting liquidity score
            spread_score = max(0, 1 - (snapshot.spread / snapshot.mid_price) * 1000)  # Tighter spread = higher score
            depth_score = min(1, (snapshot.bid_volume + snapshot.ask_volume) / 10)  # More depth = higher score
            
            # Historical spread stability
            if len(self.order_book_history) > 10:
                recent_spreads = [snap.spread / snap.mid_price for snap in list(self.order_book_history)[-10:]]
                stability_score = max(0, 1 - np.std(recent_spreads) * 10000)
            else:
                stability_score = 0.5
            
            # Weighted combination
            liquidity_score = (
                0.4 * spread_score +
                0.4 * depth_score +
                0.2 * stability_score
            )
            
            return np.clip(liquidity_score, 0, 1)
            
        except Exception as e:
            logger.error(f"Liquidity calculation error: {e}")
            return 0.5
    
    def calculate_market_impact(self, trade_size: float, side: str) -> float:
        """Estimate market impact of a trade"""
        if not self.order_book_history:
            return 0.01  # Default 1% impact
        
        try:
            snapshot = self.order_book_history[-1]
            
            # Determine which side of book to analyze
            if side.lower() == 'buy':
                relevant_orders = snapshot.asks
            else:
                relevant_orders = snapshot.bids
            
            # Calculate cumulative impact
            cumulative_volume = 0
            weighted_price = 0
            total_value = 0
            
            for price, volume in relevant_orders:
                if cumulative_volume >= trade_size:
                    break
                
                volume_to_take = min(volume, trade_size - cumulative_volume)
                weighted_price += price * volume_to_take
                total_value += volume_to_take
                cumulative_volume += volume_to_take
            
            if total_value == 0:
                return 0.05  # 5% impact if no liquidity
            
            avg_execution_price = weighted_price / total_value
            impact = abs(avg_execution_price - snapshot.mid_price) / snapshot.mid_price
            
            # Add additional impact for large trades
            size_penalty = min(0.02, trade_size / (snapshot.bid_volume + snapshot.ask_volume) * 0.01)
            
            return impact + size_penalty
            
        except Exception as e:
            logger.error(f"Market impact calculation error: {e}")
            return 0.01
    
    def optimize_trade_size(self, desired_size: float, max_impact: float = 0.003) -> float:
        """Optimize trade size based on market impact constraints"""
        if not self.order_book_history:
            return min(desired_size, 0.001)  # Conservative default
        
        try:
            # Binary search for optimal size
            min_size = 0.00001
            max_size = desired_size
            optimal_size = desired_size
            
            for _ in range(10):  # Maximum 10 iterations
                test_size = (min_size + max_size) / 2
                impact = self.calculate_market_impact(test_size, 'buy')  # Use buy as reference
                
                if impact <= max_impact:
                    min_size = test_size
                    optimal_size = test_size
                else:
                    max_size = test_size
                
                if max_size - min_size < 0.000001:
                    break
            
            return optimal_size
            
        except Exception as e:
            logger.error(f"Trade size optimization error: {e}")
            return min(desired_size, 0.001)
    
    def generate_microstructure_signals(self, desired_trade_size: float = 0.001) -> MicrostructureSignals:
        """Generate comprehensive microstructure-based signals"""
        try:
            # Detect market regime
            regime = self.detect_market_regime()
            
            # Calculate liquidity metrics
            liquidity_score = self.calculate_liquidity_score()
            
            # Order flow analysis
            if self.order_book_history:
                recent_imbalances = [snap.imbalance for snap in list(self.order_book_history)[-10:]]
                order_flow_imbalance = np.mean(recent_imbalances)
            else:
                order_flow_imbalance = 0
            
            # Market impact and optimal sizing
            market_impact_score = self.calculate_market_impact(desired_trade_size, 'buy')
            optimal_trade_size = self.optimize_trade_size(desired_trade_size)
            
            # Execution urgency based on regime and liquidity
            if regime in [MarketRegime.VOLATILE, MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
                execution_urgency = 0.8  # High urgency in trending/volatile markets
            elif liquidity_score < 0.3:
                execution_urgency = 0.9  # High urgency in illiquid markets
            else:
                execution_urgency = 0.4  # Low urgency in stable conditions
            
            # Smart routing recommendation
            if liquidity_score > 0.7 and market_impact_score < 0.002:
                smart_routing = "aggressive"  # Market orders
            elif liquidity_score > 0.4:
                smart_routing = "passive"  # Limit orders
            else:
                smart_routing = "iceberg"  # Split large orders
            
            # Overall confidence
            confidence = (liquidity_score + (1 - market_impact_score)) / 2
            
            return MicrostructureSignals(
                regime=regime,
                liquidity_score=liquidity_score,
                order_flow_imbalance=order_flow_imbalance,
                market_impact_score=market_impact_score,
                optimal_trade_size=optimal_trade_size,
                execution_urgency=execution_urgency,
                smart_routing_recommendation=smart_routing,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return MicrostructureSignals(
                regime=MarketRegime.RANGE_BOUND,
                liquidity_score=0.5,
                order_flow_imbalance=0,
                market_impact_score=0.01,
                optimal_trade_size=desired_trade_size * 0.5,
                execution_urgency=0.5,
                smart_routing_recommendation="passive",
                confidence=0.3
            )
    
    def get_volume_profile(self, price_levels: int = 20) -> Dict[float, float]:
        """Generate volume profile for recent trading"""
        volume_profile = {}
        
        if len(self.order_book_history) < 10:
            return volume_profile
        
        try:
            # Get price range
            all_prices = [snap.mid_price for snap in self.order_book_history]
            min_price = min(all_prices)
            max_price = max(all_prices)
            
            # Create price buckets
            price_step = (max_price - min_price) / price_levels
            
            for i in range(price_levels):
                bucket_price = min_price + i * price_step
                bucket_volume = 0
                
                # Aggregate volume for this price level
                for snap in self.order_book_history:
                    if bucket_price <= snap.mid_price < bucket_price + price_step:
                        bucket_volume += snap.bid_volume + snap.ask_volume
                
                if bucket_volume > 0:
                    volume_profile[bucket_price] = bucket_volume
            
            return volume_profile
            
        except Exception as e:
            logger.error(f"Volume profile error: {e}")
            return volume_profile