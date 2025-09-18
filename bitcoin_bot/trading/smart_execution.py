# trading/smart_execution.py
"""
Institutional-Grade Smart Order Execution System
Implements TWAP, VWAP, Iceberg orders, and intelligent routing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import logging
import time
import random
from collections import deque
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    MARKET = "market"           # Immediate execution
    LIMIT = "limit"             # Passive limit orders
    TWAP = "twap"              # Time-Weighted Average Price
    VWAP = "vwap"              # Volume-Weighted Average Price
    ICEBERG = "iceberg"         # Hidden size orders
    ADAPTIVE = "adaptive"       # AI-driven adaptive execution
    LIQUIDITY_SEEKING = "liquidity_seeking"  # Hunt for liquidity


class OrderStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class SmartOrder:
    """Enhanced order with smart execution parameters"""
    symbol: str
    side: str  # 'buy' or 'sell'
    total_size: float
    strategy: ExecutionStrategy
    
    # Optional parameters
    limit_price: Optional[float] = None
    time_limit: Optional[int] = None  # Seconds
    max_participation_rate: float = 0.2  # Max 20% of volume
    min_fill_size: float = 0.0001  # Minimum fill size
    
    # TWAP parameters
    twap_duration: Optional[int] = None  # Seconds
    twap_intervals: int = 10
    
    # VWAP parameters
    vwap_lookback: int = 100  # Historical periods for VWAP calculation
    
    # Iceberg parameters
    iceberg_visible_size: float = 0.001
    iceberg_randomness: float = 0.1  # 10% size randomization
    
    # Adaptive parameters
    urgency: float = 0.5  # 0-1, higher = more aggressive
    
    # State tracking
    order_id: str = ""
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0.0
    remaining_size: float = 0.0
    average_fill_price: float = 0.0
    child_orders: List[str] = field(default_factory=list)
    created_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        self.remaining_size = self.total_size
        if not self.order_id:
            self.order_id = f"smart_{int(time.time() * 1000)}"


@dataclass
class ExecutionMetrics:
    """Execution quality metrics"""
    implementation_shortfall: float  # Cost relative to arrival price
    volume_participation: float      # % of market volume consumed
    fill_rate: float                # % of order filled
    time_to_complete: float         # Seconds to complete
    slippage: float                 # Price movement during execution
    market_impact: float            # Estimated impact on price
    vwap_performance: float         # Performance vs VWAP benchmark


class MarketDataFeed:
    """Mock market data feed for execution algorithms"""
    
    def __init__(self):
        self.price_history = deque(maxlen=1000)
        self.volume_history = deque(maxlen=1000)
        self.order_book = {"bids": [], "asks": []}
        self.last_price = 0.0
        self.last_volume = 0.0
    
    def update(self, price: float, volume: float, order_book: Dict):
        """Update market data"""
        self.last_price = price
        self.last_volume = volume
        self.price_history.append(price)
        self.volume_history.append(volume)
        self.order_book = order_book
    
    def get_vwap(self, periods: int = 100) -> float:
        """Calculate Volume-Weighted Average Price"""
        if len(self.price_history) < periods or len(self.volume_history) < periods:
            return self.last_price
        
        prices = list(self.price_history)[-periods:]
        volumes = list(self.volume_history)[-periods:]
        
        total_value = sum(p * v for p, v in zip(prices, volumes))
        total_volume = sum(volumes)
        
        return total_value / total_volume if total_volume > 0 else self.last_price
    
    def get_participation_rate(self, our_volume: float, lookback: int = 10) -> float:
        """Calculate our participation rate in recent volume"""
        if len(self.volume_history) < lookback:
            return 0.0
        
        recent_volume = sum(list(self.volume_history)[-lookback:])
        return our_volume / recent_volume if recent_volume > 0 else 0.0
    
    def estimate_market_impact(self, size: float, side: str) -> float:
        """Estimate market impact of order"""
        if not self.order_book["bids"] or not self.order_book["asks"]:
            return 0.01  # 1% default impact
        
        if side.lower() == "buy":
            levels = self.order_book["asks"]
        else:
            levels = self.order_book["bids"]
        
        # Calculate weighted average price for size
        remaining_size = size
        total_cost = 0.0
        
        for price, available in levels:
            if remaining_size <= 0:
                break
            
            fill_size = min(remaining_size, available)
            total_cost += fill_size * price
            remaining_size -= fill_size
        
        if remaining_size > 0:
            return 0.05  # 5% impact if not enough liquidity
        
        avg_price = total_cost / size
        current_price = self.last_price
        
        return abs(avg_price - current_price) / current_price


class ExecutionAlgorithm(ABC):
    """Base class for execution algorithms"""
    
    def __init__(self, market_data: MarketDataFeed, exchange_api):
        self.market_data = market_data
        self.exchange_api = exchange_api
        self.active_orders = {}
        self.execution_history = []
    
    @abstractmethod
    async def execute(self, order: SmartOrder) -> ExecutionMetrics:
        """Execute the smart order"""
        pass
    
    def calculate_metrics(self, order: SmartOrder, fills: List[Dict]) -> ExecutionMetrics:
        """Calculate execution performance metrics"""
        if not fills:
            return ExecutionMetrics(0, 0, 0, 0, 0, 0, 0)
        
        # Calculate metrics
        arrival_price = fills[0].get("arrival_price", self.market_data.last_price)
        avg_fill_price = sum(f["price"] * f["size"] for f in fills) / sum(f["size"] for f in fills)
        
        implementation_shortfall = (avg_fill_price - arrival_price) / arrival_price
        if order.side.lower() == "sell":
            implementation_shortfall *= -1
        
        fill_rate = sum(f["size"] for f in fills) / order.total_size
        execution_time = (datetime.now() - order.created_time).total_seconds()
        
        # Estimate slippage and market impact
        price_move = (self.market_data.last_price - arrival_price) / arrival_price
        slippage = implementation_shortfall - price_move
        
        return ExecutionMetrics(
            implementation_shortfall=implementation_shortfall,
            volume_participation=self.market_data.get_participation_rate(sum(f["size"] for f in fills)),
            fill_rate=fill_rate,
            time_to_complete=execution_time,
            slippage=slippage,
            market_impact=abs(slippage),
            vwap_performance=(avg_fill_price - self.market_data.get_vwap()) / self.market_data.get_vwap()
        )


class TWAPExecutor(ExecutionAlgorithm):
    """Time-Weighted Average Price execution"""
    
    async def execute(self, order: SmartOrder) -> ExecutionMetrics:
        """Execute TWAP strategy"""
        logger.info(f"Starting TWAP execution for {order.total_size} {order.symbol}")
        
        duration = order.twap_duration or 300  # Default 5 minutes
        intervals = order.twap_intervals
        interval_size = duration / intervals
        slice_size = order.total_size / intervals
        
        fills = []
        start_time = time.time()
        
        for i in range(intervals):
            if order.remaining_size <= 0:
                break
            
            # Calculate slice size with randomization to avoid predictability
            current_slice = min(slice_size * (0.8 + 0.4 * random.random()), order.remaining_size)
            
            try:
                # Place limit order slightly aggressive to ensure fill
                current_price = self.market_data.last_price
                if order.side.lower() == "buy":
                    limit_price = current_price * 1.001  # 0.1% above market
                else:
                    limit_price = current_price * 0.999  # 0.1% below market
                
                # Simulate order placement and fill
                fill_price = limit_price
                fills.append({
                    "price": fill_price,
                    "size": current_slice,
                    "timestamp": time.time(),
                    "arrival_price": self.market_data.last_price
                })
                
                order.remaining_size -= current_slice
                order.filled_size += current_slice
                
                logger.debug(f"TWAP slice {i+1}/{intervals}: {current_slice:.6f} @ {fill_price:.2f}")
                
            except Exception as e:
                logger.error(f"TWAP execution error on slice {i+1}: {e}")
                break
            
            # Wait for next interval (unless last slice)
            if i < intervals - 1:
                await asyncio.sleep(interval_size)
        
        order.status = OrderStatus.FILLED if order.remaining_size == 0 else OrderStatus.PARTIALLY_FILLED
        return self.calculate_metrics(order, fills)


class VWAPExecutor(ExecutionAlgorithm):
    """Volume-Weighted Average Price execution"""
    
    async def execute(self, order: SmartOrder) -> ExecutionMetrics:
        """Execute VWAP strategy"""
        logger.info(f"Starting VWAP execution for {order.total_size} {order.symbol}")
        
        fills = []
        volume_profile = self._build_volume_profile()
        max_participation = order.max_participation_rate
        
        start_time = time.time()
        execution_start_price = self.market_data.last_price
        
        while order.remaining_size > 0 and (time.time() - start_time) < (order.time_limit or 1800):
            try:
                # Get current market volume rate
                current_volume_rate = self._estimate_current_volume_rate(volume_profile)
                
                # Calculate target participation for this period
                target_size = min(
                    current_volume_rate * max_participation * 60,  # Per minute
                    order.remaining_size,
                    order.total_size * 0.1  # Max 10% per iteration
                )
                
                if target_size < order.min_fill_size:
                    await asyncio.sleep(5)  # Wait for better opportunity
                    continue
                
                # Execute slice
                current_price = self.market_data.last_price
                vwap_benchmark = self.market_data.get_vwap(order.vwap_lookback)
                
                # Price slightly better than VWAP to improve performance
                if order.side.lower() == "buy":
                    if current_price < vwap_benchmark:
                        limit_price = current_price * 1.0005  # Slightly aggressive
                    else:
                        limit_price = current_price * 0.9995  # More passive
                else:
                    if current_price > vwap_benchmark:
                        limit_price = current_price * 0.9995  # Slightly aggressive
                    else:
                        limit_price = current_price * 1.0005  # More passive
                
                # Simulate fill
                fill_size = target_size
                fills.append({
                    "price": limit_price,
                    "size": fill_size,
                    "timestamp": time.time(),
                    "arrival_price": execution_start_price,
                    "vwap_benchmark": vwap_benchmark
                })
                
                order.remaining_size -= fill_size
                order.filled_size += fill_size
                
                logger.debug(f"VWAP slice: {fill_size:.6f} @ {limit_price:.2f} (VWAP: {vwap_benchmark:.2f})")
                
                # Dynamic wait based on market conditions
                wait_time = self._calculate_wait_time(current_volume_rate, order.urgency)
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"VWAP execution error: {e}")
                await asyncio.sleep(10)
        
        order.status = OrderStatus.FILLED if order.remaining_size == 0 else OrderStatus.PARTIALLY_FILLED
        return self.calculate_metrics(order, fills)
    
    def _build_volume_profile(self) -> Dict[int, float]:
        """Build historical volume profile by hour"""
        profile = {}
        
        # Use recent volume history to estimate hourly patterns
        if len(self.market_data.volume_history) >= 24:
            volumes = list(self.market_data.volume_history)[-24:]  # Last 24 periods
            for i, vol in enumerate(volumes):
                hour = i % 24
                profile[hour] = profile.get(hour, 0) + vol
        
        # Normalize
        total_volume = sum(profile.values())
        if total_volume > 0:
            profile = {hour: vol/total_volume for hour, vol in profile.items()}
        
        return profile
    
    def _estimate_current_volume_rate(self, volume_profile: Dict[int, float]) -> float:
        """Estimate current volume rate"""
        current_hour = datetime.now().hour
        base_rate = self.market_data.last_volume
        
        # Adjust by hourly profile
        hourly_multiplier = volume_profile.get(current_hour, 1.0)
        return base_rate * hourly_multiplier
    
    def _calculate_wait_time(self, volume_rate: float, urgency: float) -> float:
        """Calculate optimal wait time between slices"""
        base_wait = 60  # 1 minute base
        
        # Adjust for volume (less wait in high volume)
        volume_adjustment = max(0.5, 1.0 / (volume_rate + 0.1))
        
        # Adjust for urgency (less wait for urgent orders)
        urgency_adjustment = 2.0 - urgency
        
        return base_wait * volume_adjustment * urgency_adjustment


class IcebergExecutor(ExecutionAlgorithm):
    """Iceberg order execution with hidden size"""
    
    async def execute(self, order: SmartOrder) -> ExecutionMetrics:
        """Execute Iceberg strategy"""
        logger.info(f"Starting Iceberg execution for {order.total_size} {order.symbol}")
        
        fills = []
        base_visible_size = order.iceberg_visible_size
        randomness = order.iceberg_randomness
        
        start_time = time.time()
        execution_start_price = self.market_data.last_price
        
        while order.remaining_size > 0 and (time.time() - start_time) < (order.time_limit or 3600):
            try:
                # Calculate current slice size with randomization
                visible_size = base_visible_size * (1 + randomness * (random.random() - 0.5))
                slice_size = min(visible_size, order.remaining_size)
                
                if slice_size < order.min_fill_size:
                    break
                
                # Place limit order at or near best price
                current_price = self.market_data.last_price
                
                if order.side.lower() == "buy":
                    # Try to get filled at best bid or slightly better
                    limit_price = current_price * 0.9998
                else:
                    # Try to get filled at best ask or slightly better
                    limit_price = current_price * 1.0002
                
                # Simulate order placement and potential fill
                fill_probability = self._calculate_fill_probability(slice_size, limit_price, order.side)
                
                if random.random() < fill_probability:
                    # Order gets filled
                    fills.append({
                        "price": limit_price,
                        "size": slice_size,
                        "timestamp": time.time(),
                        "arrival_price": execution_start_price
                    })
                    
                    order.remaining_size -= slice_size
                    order.filled_size += slice_size
                    
                    logger.debug(f"Iceberg slice filled: {slice_size:.6f} @ {limit_price:.2f}")
                    
                    # Quick refresh for next slice
                    await asyncio.sleep(random.uniform(1, 3))
                else:
                    # Order not filled, wait and try again
                    await asyncio.sleep(random.uniform(10, 30))
                
            except Exception as e:
                logger.error(f"Iceberg execution error: {e}")
                await asyncio.sleep(10)
        
        order.status = OrderStatus.FILLED if order.remaining_size == 0 else OrderStatus.PARTIALLY_FILLED
        return self.calculate_metrics(order, fills)
    
    def _calculate_fill_probability(self, size: float, price: float, side: str) -> float:
        """Estimate probability of fill based on market conditions"""
        current_price = self.market_data.last_price
        
        # Distance from market price
        price_distance = abs(price - current_price) / current_price
        
        # Size relative to typical market volume
        size_impact = size / (self.market_data.last_volume + 0.001)
        
        # Base probability decreases with distance and size
        base_prob = 0.8
        distance_penalty = price_distance * 50  # Penalty for being away from market
        size_penalty = size_impact * 0.5
        
        probability = max(0.1, base_prob - distance_penalty - size_penalty)
        return probability


class AdaptiveExecutor(ExecutionAlgorithm):
    """AI-driven adaptive execution algorithm"""
    
    def __init__(self, market_data: MarketDataFeed, exchange_api):
        super().__init__(market_data, exchange_api)
        self.market_impact_model = self._initialize_impact_model()
        self.execution_tactics = [
            "aggressive_market",
            "passive_limit", 
            "mid_point",
            "volume_matching",
            "momentum_following"
        ]
    
    async def execute(self, order: SmartOrder) -> ExecutionMetrics:
        """Execute using adaptive algorithm"""
        logger.info(f"Starting Adaptive execution for {order.total_size} {order.symbol}")
        
        fills = []
        start_time = time.time()
        execution_start_price = self.market_data.last_price
        
        # Initial market state assessment
        market_state = self._assess_market_state()
        
        while order.remaining_size > 0 and (time.time() - start_time) < (order.time_limit or 1800):
            try:
                # Dynamic strategy selection based on current conditions
                current_tactic = self._select_execution_tactic(market_state, order)
                slice_size = self._calculate_optimal_slice_size(order, market_state)
                
                if slice_size < order.min_fill_size:
                    await asyncio.sleep(5)
                    market_state = self._assess_market_state()  # Refresh state
                    continue
                
                # Execute slice using selected tactic
                fill_result = await self._execute_slice(order, slice_size, current_tactic)
                
                if fill_result:
                    fills.append({
                        "price": fill_result["price"],
                        "size": fill_result["size"],
                        "timestamp": time.time(),
                        "arrival_price": execution_start_price,
                        "tactic": current_tactic
                    })
                    
                    order.remaining_size -= fill_result["size"]
                    order.filled_size += fill_result["size"]
                    
                    logger.debug(f"Adaptive slice: {fill_result['size']:.6f} @ {fill_result['price']:.2f} using {current_tactic}")
                
                # Update market state and wait
                market_state = self._assess_market_state()
                wait_time = self._calculate_adaptive_wait_time(market_state, order.urgency)
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Adaptive execution error: {e}")
                await asyncio.sleep(10)
        
        order.status = OrderStatus.FILLED if order.remaining_size == 0 else OrderStatus.PARTIALLY_FILLED
        return self.calculate_metrics(order, fills)
    
    def _assess_market_state(self) -> Dict[str, float]:
        """Assess current market conditions"""
        if len(self.market_data.price_history) < 20:
            return {"volatility": 0.02, "trend": 0.0, "liquidity": 0.5, "momentum": 0.0}
        
        prices = np.array(list(self.market_data.price_history)[-20:])
        volumes = np.array(list(self.market_data.volume_history)[-20:])
        
        # Volatility (rolling standard deviation)
        returns = np.diff(np.log(prices))
        volatility = np.std(returns)
        
        # Trend (linear regression slope)
        x = np.arange(len(prices))
        trend_slope = np.polyfit(x, prices, 1)[0] / prices[-1]  # Normalized slope
        
        # Liquidity (based on recent volume)
        avg_volume = np.mean(volumes)
        current_volume = volumes[-1] if len(volumes) > 0 else avg_volume
        liquidity_score = min(1.0, current_volume / (avg_volume + 0.001))
        
        # Momentum (price acceleration)
        if len(prices) >= 10:
            recent_momentum = (prices[-1] - prices[-5]) / prices[-5]
            older_momentum = (prices[-5] - prices[-10]) / prices[-10]
            momentum = recent_momentum - older_momentum
        else:
            momentum = 0.0
        
        return {
            "volatility": volatility,
            "trend": trend_slope,
            "liquidity": liquidity_score,
            "momentum": momentum
        }
    
    def _select_execution_tactic(self, market_state: Dict[str, float], order: SmartOrder) -> str:
        """Select optimal execution tactic based on market conditions"""
        volatility = market_state["volatility"]
        liquidity = market_state["liquidity"]
        momentum = abs(market_state["momentum"])
        urgency = order.urgency
        
        # Decision logic based on market conditions
        if urgency > 0.8 or volatility > 0.03:
            return "aggressive_market"  # High urgency or volatility
        elif liquidity > 0.7 and volatility < 0.015:
            return "passive_limit"  # Good liquidity, low volatility
        elif momentum > 0.01:
            return "momentum_following"  # Strong momentum
        elif liquidity > 0.5:
            return "volume_matching"  # Decent liquidity
        else:
            return "mid_point"  # Default conservative approach
    
    def _calculate_optimal_slice_size(self, order: SmartOrder, market_state: Dict[str, float]) -> float:
        """Calculate optimal slice size based on market impact model"""
        base_size = order.total_size * 0.1  # 10% base slice
        
        # Adjust for market conditions
        volatility_adjustment = 1.0 / (1.0 + market_state["volatility"] * 50)
        liquidity_adjustment = market_state["liquidity"]
        urgency_adjustment = 1.0 + order.urgency * 0.5
        
        optimal_size = base_size * volatility_adjustment * liquidity_adjustment * urgency_adjustment
        
        # Bounds checking
        optimal_size = max(order.min_fill_size, min(optimal_size, order.remaining_size))
        optimal_size = min(optimal_size, order.total_size * 0.25)  # Max 25% per slice
        
        return optimal_size
    
    async def _execute_slice(self, order: SmartOrder, size: float, tactic: str) -> Optional[Dict]:
        """Execute a single slice using specified tactic"""
        current_price = self.market_data.last_price
        
        try:
            if tactic == "aggressive_market":
                # Market order simulation
                if order.side.lower() == "buy":
                    fill_price = current_price * 1.002  # 0.2% slippage
                else:
                    fill_price = current_price * 0.998
                return {"price": fill_price, "size": size}
                
            elif tactic == "passive_limit":
                # Limit order at best price
                if order.side.lower() == "buy":
                    limit_price = current_price * 0.9995
                else:
                    limit_price = current_price * 1.0005
                
                # Simulate fill probability
                if random.random() < 0.7:  # 70% fill rate for passive orders
                    return {"price": limit_price, "size": size}
                
            elif tactic == "mid_point":
                # Mid-point execution
                mid_price = current_price  # Simplified - would use bid-ask mid in reality
                return {"price": mid_price, "size": size}
                
            elif tactic == "volume_matching":
                # Match recent volume patterns
                recent_volume = self.market_data.last_volume
                participation_rate = min(0.3, size / (recent_volume + 0.001))
                
                if participation_rate < 0.2:  # Low impact
                    fill_price = current_price * (1.0001 if order.side.lower() == "buy" else 0.9999)
                    return {"price": fill_price, "size": size}
                
            elif tactic == "momentum_following":
                # Follow price momentum
                momentum = self.market_state.get("momentum", 0)
                if (momentum > 0 and order.side.lower() == "buy") or (momentum < 0 and order.side.lower() == "sell"):
                    # Momentum favorable
                    fill_price = current_price * (1.0005 if order.side.lower() == "buy" else 0.9995)
                    return {"price": fill_price, "size": size}
            
            return None  # No fill this round
            
        except Exception as e:
            logger.error(f"Slice execution error: {e}")
            return None
    
    def _calculate_adaptive_wait_time(self, market_state: Dict[str, float], urgency: float) -> float:
        """Calculate adaptive wait time between slices"""
        base_wait = 30  # 30 seconds base
        
        # Adjust for market volatility (wait less in volatile markets)
        volatility_factor = max(0.5, 1.0 - market_state["volatility"] * 10)
        
        # Adjust for liquidity (wait more in illiquid markets)
        liquidity_factor = 2.0 - market_state["liquidity"]
        
        # Adjust for urgency
        urgency_factor = 2.0 - urgency
        
        adaptive_wait = base_wait * volatility_factor * liquidity_factor * urgency_factor
        return max(5, min(300, adaptive_wait))  # Between 5 seconds and 5 minutes
    
    def _initialize_impact_model(self):
        """Initialize market impact prediction model"""
        # Simplified impact model - in practice would use ML model
        return {
            "linear_coefficient": 0.001,
            "square_root_coefficient": 0.0001,
            "volatility_multiplier": 2.0
        }


class SmartExecutionEngine:
    """Main execution engine coordinating all execution strategies"""
    
    def __init__(self, exchange_api, market_data_feed: MarketDataFeed):
        self.exchange_api = exchange_api
        self.market_data = market_data_feed
        
        # Initialize execution algorithms
        self.executors = {
            ExecutionStrategy.TWAP: TWAPExecutor(market_data_feed, exchange_api),
            ExecutionStrategy.VWAP: VWAPExecutor(market_data_feed, exchange_api),
            ExecutionStrategy.ICEBERG: IcebergExecutor(market_data_feed, exchange_api),
            ExecutionStrategy.ADAPTIVE: AdaptiveExecutor(market_data_feed, exchange_api)
        }
        
        self.active_orders = {}
        self.execution_history = []
        self.performance_stats = {}
    
    async def submit_smart_order(self, order: SmartOrder) -> str:
        """Submit smart order for execution"""
        try:
            logger.info(f"Received smart order: {order.strategy.value} {order.side} {order.total_size} {order.symbol}")
            
            # Validate order
            if not self._validate_order(order):
                order.status = OrderStatus.REJECTED
                return order.order_id
            
            # Select appropriate executor
            if order.strategy in self.executors:
                executor = self.executors[order.strategy]
            else:
                # Default to adaptive execution
                executor = self.executors[ExecutionStrategy.ADAPTIVE]
                logger.warning(f"Unknown strategy {order.strategy}, using adaptive execution")
            
            # Store order
            order.status = OrderStatus.ACTIVE
            self.active_orders[order.order_id] = order
            
            # Execute asynchronously
            asyncio.create_task(self._execute_order_async(order, executor))
            
            return order.order_id
            
        except Exception as e:
            logger.error(f"Smart order submission error: {e}")
            order.status = OrderStatus.REJECTED
            return order.order_id
    
    async def _execute_order_async(self, order: SmartOrder, executor: ExecutionAlgorithm):
        """Execute order asynchronously"""
        try:
            start_time = datetime.now()
            
            # Execute the order
            metrics = await executor.execute(order)
            
            # Update order status
            order.last_update = datetime.now()
            
            # Store execution results
            execution_record = {
                "order_id": order.order_id,
                "strategy": order.strategy.value,
                "metrics": metrics,
                "start_time": start_time,
                "end_time": datetime.now(),
                "final_status": order.status.value
            }
            
            self.execution_history.append(execution_record)
            
            # Update performance statistics
            self._update_performance_stats(order.strategy, metrics)
            
            logger.info(f"Order {order.order_id} completed: {order.status.value}, "
                       f"filled {order.filled_size:.6f}/{order.total_size:.6f}")
            
        except Exception as e:
            logger.error(f"Order execution error for {order.order_id}: {e}")
            order.status = OrderStatus.REJECTED
        finally:
            # Remove from active orders
            if order.order_id in self.active_orders:
                del self.active_orders[order.order_id]
    
    def _validate_order(self, order: SmartOrder) -> bool:
        """Validate order parameters"""
        if order.total_size <= 0:
            logger.error("Invalid order size")
            return False
        
        if order.side.lower() not in ["buy", "sell"]:
            logger.error("Invalid order side")
            return False
        
        if order.limit_price and order.limit_price <= 0:
            logger.error("Invalid limit price")
            return False
        
        return True
    
    def _update_performance_stats(self, strategy: ExecutionStrategy, metrics: ExecutionMetrics):
        """Update performance statistics for strategy"""
        if strategy not in self.performance_stats:
            self.performance_stats[strategy] = []
        
        self.performance_stats[strategy].append({
            "timestamp": datetime.now(),
            "implementation_shortfall": metrics.implementation_shortfall,
            "fill_rate": metrics.fill_rate,
            "market_impact": metrics.market_impact,
            "vwap_performance": metrics.vwap_performance
        })
        
        # Keep only recent performance (last 100 executions)
        self.performance_stats[strategy] = self.performance_stats[strategy][-100:]
    
    def get_execution_status(self, order_id: str) -> Optional[SmartOrder]:
        """Get current status of smart order"""
        return self.active_orders.get(order_id)
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel active smart order"""
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
            order.status = OrderStatus.CANCELLED
            logger.info(f"Order {order_id} cancelled")
            return True
        return False
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate execution performance report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "active_orders": len(self.active_orders),
            "total_executions": len(self.execution_history),
            "strategy_performance": {}
        }
        
        for strategy, stats in self.performance_stats.items():
            if stats:
                recent_stats = stats[-20:]  # Last 20 executions
                
                report["strategy_performance"][strategy.value] = {
                    "avg_implementation_shortfall": np.mean([s["implementation_shortfall"] for s in recent_stats]),
                    "avg_fill_rate": np.mean([s["fill_rate"] for s in recent_stats]),
                    "avg_market_impact": np.mean([s["market_impact"] for s in recent_stats]),
                    "avg_vwap_performance": np.mean([s["vwap_performance"] for s in recent_stats]),
                    "execution_count": len(recent_stats)
                }
        
        return report