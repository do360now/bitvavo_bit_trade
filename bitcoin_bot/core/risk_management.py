# core/risk_management.py
"""
Institutional-Grade Risk Management System
Implements sophisticated risk controls used by top-tier trading firms
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from collections import deque
import json

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    EXTREME = 0.9


class RiskAlert(Enum):
    GREEN = "green"      # Normal operations
    YELLOW = "yellow"    # Caution advised
    ORANGE = "orange"    # Risk elevated
    RED = "red"          # Halt trading


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    var_1d: float  # 1-day Value at Risk
    var_1w: float  # 1-week Value at Risk
    expected_shortfall: float  # Conditional VaR
    max_drawdown: float
    sharpe_ratio: float
    volatility: float
    beta: float  # Market beta
    correlation_btc: float
    portfolio_heat: float  # Overall risk temperature
    margin_utilization: float
    leverage_ratio: float
    concentration_risk: float


@dataclass
class PositionRisk:
    """Individual position risk assessment"""
    symbol: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    risk_contribution: float  # Contribution to portfolio risk
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    time_decay_risk: float = 0.0  # For options/futures
    liquidity_risk: float = 0.0


@dataclass
class RiskLimits:
    """Risk limits configuration"""
    max_position_size_pct: float = 0.20  # Max 20% of portfolio per position
    max_daily_var_pct: float = 0.05      # Max 5% daily VaR
    max_portfolio_leverage: float = 2.0   # Max 2x leverage
    max_drawdown_pct: float = 0.15       # Max 15% drawdown
    max_correlation: float = 0.8         # Max correlation with market
    max_concentration_pct: float = 0.5   # Max 50% in any single asset class
    min_liquidity_ratio: float = 0.1    # Min 10% cash buffer
    max_trades_per_day: int = 100
    max_trade_size_pct: float = 0.05     # Max 5% per trade


class VolatilityRegimeDetector:
    """Detect volatility regimes for dynamic risk adjustment"""
    
    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.price_history = deque(maxlen=lookback)
        self.vol_history = deque(maxlen=lookback)
        
    def update(self, price: float):
        """Update with new price"""
        self.price_history.append(price)
        
        if len(self.price_history) > 1:
            returns = np.diff(np.log(list(self.price_history)))
            if len(returns) >= 10:
                current_vol = np.std(returns[-10:])  # 10-period vol
                self.vol_history.append(current_vol)
    
    def get_volatility_regime(self) -> Dict[str, float]:
        """Classify current volatility regime"""
        if len(self.vol_history) < 30:
            return {"regime": "normal", "percentile": 0.5, "multiplier": 1.0}
        
        try:
            current_vol = self.vol_history[-1]
            vol_array = np.array(self.vol_history)
            
            # Calculate percentile of current volatility
            percentile = np.percentile(vol_array, np.searchsorted(np.sort(vol_array), current_vol) / len(vol_array) * 100)
            
            # Classify regime
            if percentile > 90:
                regime = "extreme"
                multiplier = 2.0
            elif percentile > 75:
                regime = "high"
                multiplier = 1.5
            elif percentile < 25:
                regime = "low"
                multiplier = 0.7
            else:
                regime = "normal"
                multiplier = 1.0
            
            return {
                "regime": regime,
                "percentile": percentile / 100,
                "multiplier": multiplier,
                "current_vol": current_vol
            }
            
        except Exception as e:
            logger.error(f"Volatility regime detection error: {e}")
            return {"regime": "normal", "percentile": 0.5, "multiplier": 1.0}


class DrawdownTracker:
    """Track and analyze drawdowns"""
    
    def __init__(self):
        self.peak_equity = 0
        self.current_drawdown = 0
        self.max_drawdown = 0
        self.drawdown_duration = 0
        self.equity_history = deque(maxlen=1000)
        self.drawdown_history = []
    
    def update(self, current_equity: float):
        """Update with new equity value"""
        self.equity_history.append({
            'timestamp': datetime.now(),
            'equity': current_equity
        })
        
        # Update peak
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            # End of drawdown period
            if self.current_drawdown < 0:
                self.drawdown_history.append({
                    'max_drawdown': self.current_drawdown,
                    'duration': self.drawdown_duration,
                    'recovery_date': datetime.now()
                })
            self.current_drawdown = 0
            self.drawdown_duration = 0
        else:
            # In drawdown
            self.current_drawdown = (current_equity - self.peak_equity) / self.peak_equity
            self.max_drawdown = min(self.max_drawdown, self.current_drawdown)
            self.drawdown_duration += 1
    
    def get_drawdown_stats(self) -> Dict[str, float]:
        """Get comprehensive drawdown statistics"""
        return {
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "drawdown_duration": self.drawdown_duration,
            "avg_drawdown": np.mean([dd['max_drawdown'] for dd in self.drawdown_history]) if self.drawdown_history else 0,
            "max_duration": max([dd['duration'] for dd in self.drawdown_history]) if self.drawdown_history else 0,
            "recovery_factor": abs(self.max_drawdown) / len(self.drawdown_history) if self.drawdown_history and self.max_drawdown != 0 else 0
        }


class PortfolioRiskAnalyzer:
    """Comprehensive portfolio risk analysis"""
    
    def __init__(self):
        self.positions = {}
        self.historical_returns = deque(maxlen=252)  # 1 year of daily returns
        self.correlation_matrix = {}
        self.var_confidence_level = 0.05  # 95% VaR
        
    def update_position(self, symbol: str, size: float, entry_price: float, current_price: float):
        """Update position information"""
        unrealized_pnl = (current_price - entry_price) * size
        unrealized_pnl_pct = (current_price - entry_price) / entry_price
        
        self.positions[symbol] = PositionRisk(
            symbol=symbol,
            size=size,
            entry_price=entry_price,
            current_price=current_price,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            risk_contribution=0.0  # Will be calculated
        )
    
    def calculate_var(self, confidence_level: float = 0.05) -> Tuple[float, float]:
        """Calculate Value at Risk using historical simulation"""
        if len(self.historical_returns) < 30:
            return 0.0, 0.0  # Insufficient data
        
        try:
            returns = np.array(self.historical_returns)
            
            # 1-day VaR
            var_1d = np.percentile(returns, confidence_level * 100)
            
            # 1-week VaR (assuming sqrt(5) scaling)
            var_1w = var_1d * np.sqrt(5)
            
            return abs(var_1d), abs(var_1w)
            
        except Exception as e:
            logger.error(f"VaR calculation error: {e}")
            return 0.0, 0.0
    
    def calculate_expected_shortfall(self, confidence_level: float = 0.05) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        if len(self.historical_returns) < 30:
            return 0.0
        
        try:
            returns = np.array(self.historical_returns)
            var_threshold = np.percentile(returns, confidence_level * 100)
            
            # Expected value of returns below VaR threshold
            tail_returns = returns[returns <= var_threshold]
            expected_shortfall = np.mean(tail_returns) if len(tail_returns) > 0 else 0
            
            return abs(expected_shortfall)
            
        except Exception as e:
            logger.error(f"Expected Shortfall calculation error: {e}")
            return 0.0
    
    def calculate_portfolio_metrics(self, total_equity: float, benchmark_returns: List[float] = None) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            # VaR calculations
            var_1d, var_1w = self.calculate_var()
            expected_shortfall = self.calculate_expected_shortfall()
            
            # Volatility
            if len(self.historical_returns) > 1:
                volatility = np.std(self.historical_returns) * np.sqrt(252)  # Annualized
                sharpe_ratio = np.mean(self.historical_returns) / np.std(self.historical_returns) * np.sqrt(252) if np.std(self.historical_returns) > 0 else 0
            else:
                volatility = 0.0
                sharpe_ratio = 0.0
            
            # Beta calculation (if benchmark provided)
            if benchmark_returns and len(benchmark_returns) >= len(self.historical_returns):
                portfolio_returns = np.array(self.historical_returns)
                benchmark_returns = np.array(benchmark_returns[-len(portfolio_returns):])
                
                if len(portfolio_returns) > 10 and np.var(benchmark_returns) > 0:
                    beta = np.cov(portfolio_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
                    correlation = np.corrcoef(portfolio_returns, benchmark_returns)[0, 1]
                else:
                    beta = 1.0
                    correlation = 0.0
            else:
                beta = 1.0
                correlation = 0.0
            
            # Portfolio concentration
            if self.positions:
                position_values = [abs(pos.unrealized_pnl) for pos in self.positions.values()]
                total_position_value = sum(position_values)
                concentration_risk = max(position_values) / total_position_value if total_position_value > 0 else 0
            else:
                concentration_risk = 0.0
            
            # Portfolio heat (overall risk temperature)
            portfolio_heat = min(1.0, (var_1d * 10) + (volatility * 2) + (concentration_risk * 0.5))
            
            return RiskMetrics(
                var_1d=var_1d,
                var_1w=var_1w,
                expected_shortfall=expected_shortfall,
                max_drawdown=0.0,  # Will be set externally by drawdown tracker
                sharpe_ratio=sharpe_ratio,
                volatility=volatility,
                beta=beta,
                correlation_btc=correlation,
                portfolio_heat=portfolio_heat,
                margin_utilization=0.0,  # To be implemented based on exchange data
                leverage_ratio=1.0,      # To be calculated based on positions
                concentration_risk=concentration_risk
            )
            
        except Exception as e:
            logger.error(f"Portfolio metrics calculation error: {e}")
            return RiskMetrics(
                var_1d=0.0, var_1w=0.0, expected_shortfall=0.0, max_drawdown=0.0,
                sharpe_ratio=0.0, volatility=0.0, beta=1.0, correlation_btc=0.0,
                portfolio_heat=0.5, margin_utilization=0.0, leverage_ratio=1.0,
                concentration_risk=0.0
            )


class RiskManager:
    """Main risk management system"""
    
    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self.portfolio_analyzer = PortfolioRiskAnalyzer()
        self.drawdown_tracker = DrawdownTracker()
        self.volatility_detector = VolatilityRegimeDetector()
        
        # Risk state tracking
        self.current_alert_level = RiskAlert.GREEN
        self.daily_trade_count = 0
        self.daily_trade_reset = datetime.now().date()
        self.emergency_stop = False
        
        # Performance tracking
        self.risk_adjustments = []
        self.violation_history = []
    
    def update_market_data(self, price: float, volume: float, portfolio_value: float):
        """Update with new market data"""
        self.volatility_detector.update(price)
        self.drawdown_tracker.update(portfolio_value)
        
        # Reset daily counters
        if datetime.now().date() > self.daily_trade_reset:
            self.daily_trade_count = 0
            self.daily_trade_reset = datetime.now().date()
    
    def assess_trade_risk(self, trade_size: float, trade_direction: str, current_price: float, 
                         portfolio_value: float, confidence: float) -> Dict[str, Any]:
        """Comprehensive pre-trade risk assessment"""
        try:
            risk_assessment = {
                "approved": True,
                "risk_level": RiskLevel.LOW,
                "adjustments": [],
                "warnings": [],
                "max_recommended_size": trade_size,
                "stop_loss": None,
                "take_profit": None
            }
            
            # Check daily trade limit
            if self.daily_trade_count >= self.limits.max_trades_per_day:
                risk_assessment["approved"] = False
                risk_assessment["warnings"].append("Daily trade limit exceeded")
                return risk_assessment
            
            # Check emergency stop
            if self.emergency_stop:
                risk_assessment["approved"] = False
                risk_assessment["warnings"].append("Emergency stop activated")
                return risk_assessment
            
            # Position size check
            position_size_pct = (trade_size * current_price) / portfolio_value
            if position_size_pct > self.limits.max_position_size_pct:
                new_size = (self.limits.max_position_size_pct * portfolio_value) / current_price
                risk_assessment["max_recommended_size"] = new_size
                risk_assessment["adjustments"].append(f"Reduced size from {trade_size:.6f} to {new_size:.6f}")
                risk_assessment["risk_level"] = RiskLevel.MEDIUM
            
            # Trade size check
            trade_size_pct = (trade_size * current_price) / portfolio_value
            if trade_size_pct > self.limits.max_trade_size_pct:
                new_size = (self.limits.max_trade_size_pct * portfolio_value) / current_price
                risk_assessment["max_recommended_size"] = min(risk_assessment["max_recommended_size"], new_size)
                risk_assessment["adjustments"].append(f"Trade size too large, reduced to {new_size:.6f}")
                risk_assessment["risk_level"] = RiskLevel.HIGH
            
            # Volatility-based adjustments
            vol_regime = self.volatility_detector.get_volatility_regime()
            if vol_regime["regime"] in ["high", "extreme"]:
                vol_adjustment = 1.0 / vol_regime["multiplier"]
                adjusted_size = risk_assessment["max_recommended_size"] * vol_adjustment
                risk_assessment["max_recommended_size"] = adjusted_size
                risk_assessment["adjustments"].append(f"Volatility adjustment: {vol_regime['regime']} vol, size reduced by {(1-vol_adjustment)*100:.1f}%")
                risk_assessment["risk_level"] = RiskLevel.HIGH
            
            # Confidence-based adjustments
            if confidence < 0.6:
                confidence_adjustment = confidence / 0.6  # Scale down for low confidence
                adjusted_size = risk_assessment["max_recommended_size"] * confidence_adjustment
                risk_assessment["max_recommended_size"] = adjusted_size
                risk_assessment["adjustments"].append(f"Low confidence ({confidence:.1%}), size reduced")
                risk_assessment["risk_level"] = RiskLevel.MEDIUM
            
            # Drawdown-based restrictions
            drawdown_stats = self.drawdown_tracker.get_drawdown_stats()
            if abs(drawdown_stats["current_drawdown"]) > self.limits.max_drawdown_pct * 0.8:  # 80% of max
                drawdown_adjustment = 0.5  # Halve position sizes
                adjusted_size = risk_assessment["max_recommended_size"] * drawdown_adjustment
                risk_assessment["max_recommended_size"] = adjusted_size
                risk_assessment["adjustments"].append("Large drawdown detected, reducing position size")
                risk_assessment["risk_level"] = RiskLevel.HIGH
                risk_assessment["warnings"].append(f"Current drawdown: {drawdown_stats['current_drawdown']:.1%}")
            
            # Set stop loss and take profit based on volatility
            current_vol = vol_regime.get("current_vol", 0.02)
            if trade_direction.upper() == "BUY":
                risk_assessment["stop_loss"] = current_price * (1 - 2 * current_vol)
                risk_assessment["take_profit"] = current_price * (1 + 3 * current_vol)
            else:  # SELL
                risk_assessment["stop_loss"] = current_price * (1 + 2 * current_vol)
                risk_assessment["take_profit"] = current_price * (1 - 3 * current_vol)
            
            # Final approval check
            if risk_assessment["max_recommended_size"] < trade_size * 0.1:  # Less than 10% of original
                risk_assessment["approved"] = False
                risk_assessment["warnings"].append("Recommended size too small after risk adjustments")
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Trade risk assessment error: {e}")
            return {
                "approved": False,
                "risk_level": RiskLevel.EXTREME,
                "adjustments": [],
                "warnings": [f"Risk assessment failed: {e}"],
                "max_recommended_size": 0.0
            }
    
    def monitor_portfolio_risk(self, portfolio_value: float, positions: Dict[str, Any]) -> RiskAlert:
        """Continuous portfolio risk monitoring"""
        try:
            # Update portfolio analyzer
            for symbol, position in positions.items():
                self.portfolio_analyzer.update_position(
                    symbol=symbol,
                    size=position.get("size", 0),
                    entry_price=position.get("entry_price", 0),
                    current_price=position.get("current_price", 0)
                )
            
            # Calculate risk metrics
            risk_metrics = self.portfolio_analyzer.calculate_portfolio_metrics(portfolio_value)
            
            # Determine alert level
            alert_level = RiskAlert.GREEN
            violations = []
            
            # Check VaR limits
            if risk_metrics.var_1d > self.limits.max_daily_var_pct:
                violations.append(f"Daily VaR ({risk_metrics.var_1d:.1%}) exceeds limit ({self.limits.max_daily_var_pct:.1%})")
                alert_level = max(alert_level, RiskAlert.ORANGE)
            
            # Check drawdown
            drawdown_stats = self.drawdown_tracker.get_drawdown_stats()
            if abs(drawdown_stats["current_drawdown"]) > self.limits.max_drawdown_pct:
                violations.append(f"Drawdown ({abs(drawdown_stats['current_drawdown']):.1%}) exceeds limit ({self.limits.max_drawdown_pct:.1%})")
                alert_level = max(alert_level, RiskAlert.RED)
                self.emergency_stop = True
            
            # Check concentration
            if risk_metrics.concentration_risk > self.limits.max_concentration_pct:
                violations.append(f"Concentration risk ({risk_metrics.concentration_risk:.1%}) too high")
                alert_level = max(alert_level, RiskAlert.YELLOW)
            
            # Check portfolio heat
            if risk_metrics.portfolio_heat > 0.8:
                violations.append("Portfolio heat elevated")
                alert_level = max(alert_level, RiskAlert.ORANGE)
            
            # Log violations
            if violations:
                self.violation_history.append({
                    "timestamp": datetime.now(),
                    "violations": violations,
                    "alert_level": alert_level,
                    "risk_metrics": risk_metrics
                })
                
                for violation in violations:
                    logger.warning(f"Risk violation: {violation}")
            
            self.current_alert_level = alert_level
            return alert_level
            
        except Exception as e:
            logger.error(f"Portfolio risk monitoring error: {e}")
            return RiskAlert.RED
    
    def get_position_sizing_recommendation(self, signal_confidence: float, portfolio_value: float, 
                                         volatility: float, market_regime: str = "normal") -> float:
        """Advanced position sizing using Kelly Criterion and risk parity"""
        try:
            # Base position size from configuration
            base_size_pct = 0.05  # 5% base allocation
            
            # Kelly Criterion adjustment
            # Assuming win rate correlates with confidence and average win/loss ratio
            win_rate = 0.4 + (signal_confidence * 0.4)  # 40-80% based on confidence
            avg_win_loss_ratio = 1.5  # Average win is 1.5x average loss
            
            kelly_fraction = (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio
            kelly_fraction = max(0, min(0.25, kelly_fraction))  # Cap at 25%
            
            # Volatility adjustment
            vol_adjustment = 0.02 / max(0.01, volatility)  # Inverse relationship with volatility
            vol_adjustment = max(0.5, min(2.0, vol_adjustment))  # Bound between 50% and 200%
            
            # Market regime adjustment
            regime_multipliers = {
                "trending": 1.2,
                "mean_reverting": 1.0,
                "volatile": 0.6,
                "normal": 1.0
            }
            regime_mult = regime_multipliers.get(market_regime, 1.0)
            
            # Confidence adjustment
            confidence_mult = 0.5 + (signal_confidence * 1.0)  # 0.5x to 1.5x based on confidence
            
            # Combine all factors
            final_size_pct = base_size_pct * kelly_fraction * vol_adjustment * regime_mult * confidence_mult
            
            # Apply limits
            final_size_pct = max(0.001, min(self.limits.max_trade_size_pct, final_size_pct))
            
            # Convert to absolute size
            recommended_size = (final_size_pct * portfolio_value)
            
            logger.debug(f"Position sizing: base={base_size_pct:.3f}, kelly={kelly_fraction:.3f}, "
                        f"vol_adj={vol_adjustment:.3f}, regime={regime_mult:.3f}, "
                        f"conf={confidence_mult:.3f}, final={final_size_pct:.3f}")
            
            return recommended_size
            
        except Exception as e:
            logger.error(f"Position sizing error: {e}")
            return portfolio_value * 0.01  # 1% fallback
    
    def generate_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        try:
            # Get current metrics
            risk_metrics = self.portfolio_analyzer.calculate_portfolio_metrics(100000)  # Dummy portfolio value
            drawdown_stats = self.drawdown_tracker.get_drawdown_stats()
            vol_regime = self.volatility_detector.get_volatility_regime()
            
            # Recent violations
            recent_violations = [v for v in self.violation_history if 
                               (datetime.now() - v["timestamp"]).days <= 7]
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "alert_level": self.current_alert_level.value,
                "emergency_stop": self.emergency_stop,
                "daily_trades": self.daily_trade_count,
                "risk_metrics": {
                    "var_1d": f"{risk_metrics.var_1d:.2%}",
                    "var_1w": f"{risk_metrics.var_1w:.2%}",
                    "expected_shortfall": f"{risk_metrics.expected_shortfall:.2%}",
                    "sharpe_ratio": round(risk_metrics.sharpe_ratio, 2),
                    "volatility": f"{risk_metrics.volatility:.2%}",
                    "portfolio_heat": f"{risk_metrics.portfolio_heat:.1%}",
                    "concentration_risk": f"{risk_metrics.concentration_risk:.1%}"
                },
                "drawdown_stats": {
                    "current": f"{drawdown_stats['current_drawdown']:.2%}",
                    "maximum": f"{drawdown_stats['max_drawdown']:.2%}",
                    "duration": drawdown_stats["drawdown_duration"]
                },
                "volatility_regime": vol_regime,
                "recent_violations": len(recent_violations),
                "risk_limits": {
                    "max_position_size": f"{self.limits.max_position_size_pct:.1%}",
                    "max_daily_var": f"{self.limits.max_daily_var_pct:.1%}",
                    "max_drawdown": f"{self.limits.max_drawdown_pct:.1%}",
                    "max_trades_daily": self.limits.max_trades_per_day
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Risk report generation error: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def reset_emergency_stop(self):
        """Reset emergency stop (manual intervention required)"""
        logger.warning("Emergency stop manually reset")
        self.emergency_stop = False
        self.current_alert_level = RiskAlert.YELLOW  # Start with caution
    
    def optimize_risk_limits(self, performance_data: List[Dict], lookback_days: int = 30):
        """Dynamically optimize risk limits based on performance"""
        try:
            if len(performance_data) < 10:
                return
            
            # Filter recent performance
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            recent_performance = [p for p in performance_data 
                                if datetime.fromisoformat(p["timestamp"]) > cutoff_date]
            
            if not recent_performance:
                return
            
            # Calculate performance metrics
            returns = [p.get("return", 0) for p in recent_performance]
            avg_return = np.mean(returns)
            volatility = np.std(returns)
            
            # Adjust limits based on performance
            if avg_return > 0 and volatility < 0.02:  # Good performance, low vol
                # Slightly relax limits
                self.limits.max_position_size_pct = min(0.25, self.limits.max_position_size_pct * 1.1)
                self.limits.max_trade_size_pct = min(0.07, self.limits.max_trade_size_pct * 1.1)
            elif avg_return < 0 or volatility > 0.05:  # Poor performance or high vol
                # Tighten limits
                self.limits.max_position_size_pct = max(0.10, self.limits.max_position_size_pct * 0.9)
                self.limits.max_trade_size_pct = max(0.03, self.limits.max_trade_size_pct * 0.9)
            
            logger.info(f"Risk limits optimized: pos_size={self.limits.max_position_size_pct:.1%}, "
                       f"trade_size={self.limits.max_trade_size_pct:.1%}")
            
        except Exception as e:
            logger.error(f"Risk limit optimization error: {e}")
    
    def record_trade_outcome(self, trade_size: float, pnl: float, confidence: float):
        """Record trade outcome for risk model improvement"""
        try:
            self.daily_trade_count += 1
            
            # Add to portfolio return history
            if trade_size > 0:
                return_pct = pnl / (trade_size * 100)  # Approximate return percentage
                self.portfolio_analyzer.historical_returns.append(return_pct)
            
            # Log for analysis
            logger.debug(f"Trade recorded: size={trade_size:.6f}, pnl={pnl:.2f}, conf={confidence:.2f}")
            
        except Exception as e:
            logger.error(f"Trade recording error: {e}")