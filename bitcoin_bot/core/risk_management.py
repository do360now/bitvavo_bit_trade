"""
Enhanced Risk Management System with circuit breakers, position limits,
and dynamic risk assessment.
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class RiskAction(Enum):
    ALLOW = "allow"
    REDUCE = "reduce"
    BLOCK = "block"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class RiskLimits:
    """Risk limits configuration"""
    max_position_size_pct: float = 0.15  # 15% max position
    max_daily_loss_pct: float = 0.05     # 5% max daily loss
    max_drawdown_pct: float = 0.10       # 10% max drawdown
    max_trades_per_hour: int = 20        # Rate limiting
    max_order_value_eur: float = 10000   # Single order limit
    
    # Volatility-based limits
    vol_threshold_high: float = 0.06     # 6% volatility = high risk
    vol_threshold_extreme: float = 0.10  # 10% volatility = extreme risk
    
    # Correlation limits
    max_correlation_exposure: float = 0.8  # Max correlation to market


@dataclass
class RiskMetrics:
    """Current risk metrics"""
    current_exposure: float
    daily_pnl: float
    max_drawdown: float
    current_volatility: float
    risk_level: RiskLevel
    trades_last_hour: int
    correlation_score: float
    
    timestamp: datetime = field(default_factory=datetime.now)


class PositionTracker:
    """Track positions and calculate risk metrics"""
    
    def __init__(self, initial_balance: Dict[str, float]):
        self.initial_balance = initial_balance.copy()
        self.current_balance = initial_balance.copy()
        self.trades = deque(maxlen=1000)
        self.equity_history = deque(maxlen=2000)  # Keep 2000 points
        self.peak_equity = sum(initial_balance.values())
        
    def update_balance(self, new_balance: Dict[str, float], btc_price: float):
        """Update current balance and calculate metrics"""
        self.current_balance = new_balance.copy()
        
        # Calculate total equity in EUR
        total_equity = new_balance.get("EUR", 0) + new_balance.get("BTC", 0) * btc_price
        
        self.equity_history.append({
            "timestamp": datetime.now(),
            "equity": total_equity,
            "btc_price": btc_price
        })
        
        # Update peak equity
        if total_equity > self.peak_equity:
            self.peak_equity = total_equity
    
    def add_trade(self, trade: Dict[str, Any]):
        """Add a trade to the history"""
        self.trades.append({
            **trade,
            "timestamp": datetime.now()
        })
    
    def get_daily_pnl(self) -> float:
        """Calculate daily P&L"""
        if len(self.equity_history) < 2:
            return 0.0
        
        # Find equity 24 hours ago
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        start_equity = None
        for point in self.equity_history:
            if point["timestamp"] >= cutoff_time:
                start_equity = point["equity"]
                break
        
        if start_equity is None:
            start_equity = self.equity_history[0]["equity"]
        
        current_equity = self.equity_history[-1]["equity"]
        return (current_equity - start_equity) / start_equity
    
    def get_max_drawdown(self) -> float:
        """Calculate maximum drawdown from peak"""
        if not self.equity_history:
            return 0.0
        
        current_equity = self.equity_history[-1]["equity"]
        return (self.peak_equity - current_equity) / self.peak_equity
    
    def get_current_exposure(self, btc_price: float) -> float:
        """Calculate current BTC exposure as percentage of total equity"""
        btc_balance = self.current_balance.get("BTC", 0)
        eur_balance = self.current_balance.get("EUR", 0)
        
        total_value = eur_balance + btc_balance * btc_price
        btc_value = btc_balance * btc_price
        
        return btc_value / total_value if total_value > 0 else 0
    
    def get_trades_last_hour(self) -> int:
        """Count trades in the last hour"""
        cutoff_time = datetime.now() - timedelta(hours=1)
        return sum(1 for trade in self.trades if trade["timestamp"] >= cutoff_time)


class VolatilityCalculator:
    """Calculate market volatility metrics"""
    
    def __init__(self, window_size: int = 100):
        self.price_history = deque(maxlen=window_size)
        self.window_size = window_size
    
    def update_price(self, price: float):
        """Update with new price"""
        self.price_history.append(price)
    
    def get_volatility(self, periods: int = 20) -> float:
        """Calculate rolling volatility"""
        if len(self.price_history) < periods + 1:
            return 0.02  # Default volatility
        
        prices = list(self.price_history)[-periods-1:]
        returns = np.diff(np.log(prices))
        
        # Annualized volatility (assuming 15-minute intervals)
        volatility = np.std(returns) * np.sqrt(96 * 365)  # 96 15-min periods per day
        return volatility
    
    def get_volatility_regime(self) -> str:
        """Classify volatility regime"""
        vol = self.get_volatility()
        
        if vol > 0.8:
            return "extreme"
        elif vol > 0.4:
            return "high"
        elif vol > 0.2:
            return "medium"
        else:
            return "low"


class RiskManager:
    """
    Enhanced risk management system with multiple safety layers
    """
    
    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self.position_tracker = None
        self.volatility_calculator = VolatilityCalculator()
        
        # Emergency state
        self.emergency_stop = False
        self.emergency_reason = ""
        
        # Risk metrics history
        self.risk_history = deque(maxlen=1000)
        
        # Circuit breaker
        self.consecutive_losses = 0
        self.max_consecutive_losses = 5
        
        logger.info("Enhanced Risk Manager initialized")
    
    def initialize_tracking(self, initial_balance: Dict[str, float]):
        """Initialize position tracking"""
        self.position_tracker = PositionTracker(initial_balance)
    
    def update_market_data(self, price: float, balance: Dict[str, float]):
        """Update with latest market data"""
        if self.position_tracker:
            self.position_tracker.update_balance(balance, price)
        
        self.volatility_calculator.update_price(price)
    
    def assess_trade_risk(
        self,
        trade_amount: float,
        trade_side: str,
        current_price: float,
        confidence: float
    ) -> Tuple[RiskAction, str, float]:
        """
        Assess risk for a proposed trade
        
        Returns:
            (action, reason, recommended_size)
        """
        if self.emergency_stop:
            return RiskAction.BLOCK, f"Emergency stop: {self.emergency_reason}", 0.0
        
        if not self.position_tracker:
            return RiskAction.BLOCK, "Risk tracking not initialized", 0.0
        
        # Calculate current metrics
        metrics = self._calculate_risk_metrics(current_price)
        
        # Risk assessment logic
        risk_issues = []
        recommended_size = trade_amount
        
        # 1. Check position size limits
        trade_value = trade_amount * current_price
        total_equity = (
            self.position_tracker.current_balance.get("EUR", 0) +
            self.position_tracker.current_balance.get("BTC", 0) * current_price
        )
        
        position_pct = trade_value / total_equity if total_equity > 0 else 0
        
        if position_pct > self.limits.max_position_size_pct:
            max_allowed = (self.limits.max_position_size_pct * total_equity) / current_price
            recommended_size = max_allowed
            risk_issues.append(f"Position size reduced: {position_pct:.1%} -> {self.limits.max_position_size_pct:.1%}")
        
        # 2. Check single order value limit
        if trade_value > self.limits.max_order_value_eur:
            max_allowed = self.limits.max_order_value_eur / current_price
            recommended_size = min(recommended_size, max_allowed)
            risk_issues.append(f"Order size limited to €{self.limits.max_order_value_eur}")
        
        # 3. Check daily loss limits
        if metrics.daily_pnl < -self.limits.max_daily_loss_pct:
            if trade_side == "buy":
                return RiskAction.BLOCK, f"Daily loss limit exceeded: {metrics.daily_pnl:.1%}", 0.0
            else:
                # Allow sells to reduce risk
                pass
        
        # 4. Check drawdown limits
        if metrics.max_drawdown > self.limits.max_drawdown_pct:
            if trade_side == "buy":
                recommended_size *= 0.5  # Reduce size significantly
                risk_issues.append(f"Drawdown limit exceeded: {metrics.max_drawdown:.1%}, reducing size")
        
        # 5. Check trade frequency
        if metrics.trades_last_hour >= self.limits.max_trades_per_hour:
            return RiskAction.BLOCK, f"Hourly trade limit exceeded: {metrics.trades_last_hour}", 0.0
        
        # 6. Volatility-based adjustments
        if metrics.current_volatility > self.limits.vol_threshold_extreme:
            if confidence < 0.8:
                return RiskAction.BLOCK, f"Extreme volatility + low confidence: {metrics.current_volatility:.1%}", 0.0
            recommended_size *= 0.3  # Very conservative
            risk_issues.append("Extreme volatility: size reduced to 30%")
            
        elif metrics.current_volatility > self.limits.vol_threshold_high:
            recommended_size *= 0.6  # Conservative
            risk_issues.append("High volatility: size reduced to 60%")
        
        # 7. Confidence-based adjustments
        if confidence < 0.4:
            return RiskAction.BLOCK, f"Confidence too low: {confidence:.1%}", 0.0
        elif confidence < 0.6:
            recommended_size *= 0.7
            risk_issues.append(f"Low confidence: size reduced to 70%")
        
        # Determine final action
        if recommended_size < trade_amount * 0.1:  # Less than 10% of original
            return RiskAction.BLOCK, "Recommended size too small", 0.0
        elif recommended_size < trade_amount * 0.8:  # Reduced by more than 20%
            reason = "Risk limits applied: " + "; ".join(risk_issues[:2])
            return RiskAction.REDUCE, reason, recommended_size
        else:
            return RiskAction.ALLOW, "Trade approved", recommended_size
    
    def _calculate_risk_metrics(self, current_price: float) -> RiskMetrics:
        """Calculate current risk metrics"""
        if not self.position_tracker:
            return RiskMetrics(0, 0, 0, 0.02, RiskLevel.LOW, 0, 0)
        
        exposure = self.position_tracker.get_current_exposure(current_price)
        daily_pnl = self.position_tracker.get_daily_pnl()
        max_drawdown = self.position_tracker.get_max_drawdown()
        volatility = self.volatility_calculator.get_volatility()
        trades_last_hour = self.position_tracker.get_trades_last_hour()
        
        # Determine risk level
        risk_score = 0
        if abs(daily_pnl) > 0.03:
            risk_score += 2
        if max_drawdown > 0.05:
            risk_score += 2
        if volatility > 0.06:
            risk_score += 2
        if trades_last_hour > 15:
            risk_score += 1
        
        if risk_score >= 5:
            risk_level = RiskLevel.EXTREME
        elif risk_score >= 3:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 1:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        metrics = RiskMetrics(
            current_exposure=exposure,
            daily_pnl=daily_pnl,
            max_drawdown=max_drawdown,
            current_volatility=volatility,
            risk_level=risk_level,
            trades_last_hour=trades_last_hour,
            correlation_score=0.0  # TODO: Implement correlation tracking
        )
        
        self.risk_history.append(metrics)
        return metrics
    
    def record_trade_outcome(self, trade: Dict[str, Any], pnl: float):
        """Record trade outcome for risk tracking"""
        if self.position_tracker:
            trade_record = {**trade, "pnl": pnl}
            self.position_tracker.add_trade(trade_record)
        
        # Track consecutive losses for circuit breaker
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.trigger_emergency_stop(f"Too many consecutive losses: {self.consecutive_losses}")
        else:
            self.consecutive_losses = 0
    
    def trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop"""
        self.emergency_stop = True
        self.emergency_reason = reason
        logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
    
    def reset_emergency_stop(self, reason: str = "Manual reset"):
        """Reset emergency stop (manual intervention required)"""
        self.emergency_stop = False
        self.emergency_reason = ""
        self.consecutive_losses = 0
        logger.warning(f"Emergency stop reset: {reason}")
    
    def get_risk_status(self) -> Dict[str, Any]:
        """Get comprehensive risk status"""
        if not self.risk_history:
            return {"status": "No data"}
        
        latest_metrics = self.risk_history[-1]
        
        return {
            "emergency_stop": self.emergency_stop,
            "emergency_reason": self.emergency_reason,
            "risk_level": latest_metrics.risk_level.value,
            "current_exposure": f"{latest_metrics.current_exposure:.1%}",
            "daily_pnl": f"{latest_metrics.daily_pnl:.2%}",
            "max_drawdown": f"{latest_metrics.max_drawdown:.2%}",
            "volatility": f"{latest_metrics.current_volatility:.1%}",
            "trades_last_hour": latest_metrics.trades_last_hour,
            "consecutive_losses": self.consecutive_losses,
            "risk_score": self._calculate_risk_score(latest_metrics)
        }
    
    def _calculate_risk_score(self, metrics: RiskMetrics) -> float:
        """Calculate normalized risk score (0-100)"""
        score = 0
        
        # Exposure risk (0-30 points)
        score += min(30, metrics.current_exposure * 30)
        
        # Drawdown risk (0-25 points)
        score += min(25, metrics.max_drawdown * 250)
        
        # Volatility risk (0-25 points)
        score += min(25, metrics.current_volatility * 250)
        
        # Trading frequency risk (0-20 points)
        score += min(20, metrics.trades_last_hour)
        
        return min(100, score)