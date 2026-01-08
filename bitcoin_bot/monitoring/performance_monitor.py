# monitoring/performance_monitor.py
import asyncio
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    timestamp: datetime
    total_btc_balance: float
    total_eur_balance: float
    total_value_eur: float
    btc_price: float
    
    # Trading metrics
    trades_today: int
    total_trades: int
    successful_trades: int
    success_rate: float
    
    # P&L metrics
    realized_pnl_eur: float
    unrealized_pnl_eur: float
    total_pnl_eur: float
    pnl_percentage: float
    
    # Risk metrics
    max_drawdown: float
    current_drawdown: float
    volatility: float
    
    # Accumulation metrics
    btc_accumulated_today: float
    btc_accumulated_total: float
    accumulation_rate_monthly: float
    
    # Optional metrics (must come last due to default values)
    sharpe_ratio: Optional[float] = None

class PerformanceMonitor:
    """Advanced performance monitoring with real-time alerts."""
    
    def __init__(self, data_dir: Path = Path("./monitoring_data")):
        self.data_dir = data_dir
        self.data_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.metrics_history: List[PerformanceMetrics] = []
        self.peak_value = 0.0
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0)
        
        # Alerts
        self.alert_thresholds = {
            "max_drawdown": 0.05,      # 5% max drawdown
            "daily_loss_limit": 0.02,  # 2% daily loss limit
            "success_rate_min": 0.4,   # 40% minimum success rate
            "accumulation_rate_min": 0.01  # 1% monthly minimum accumulation
        }
        
        self.alerts_sent = set()
        
        # Load historical data
        self._load_historical_data()
    
    async def update_metrics(
        self,
        btc_balance: float,
        eur_balance: float,
        btc_price: float,
        trades_today: int,
        total_trades: int,
        successful_trades: int,
        **kwargs
    ) -> PerformanceMetrics:
        """Update performance metrics and check for alerts."""
        
        current_time = datetime.now()
        
        # Calculate derived metrics
        total_value_eur = eur_balance + (btc_balance * btc_price)
        success_rate = successful_trades / max(1, total_trades)
        
        # Calculate drawdown
        self.peak_value = max(self.peak_value, total_value_eur)
        current_drawdown = (self.peak_value - total_value_eur) / self.peak_value if self.peak_value > 0 else 0
        
        # Calculate max drawdown
        max_drawdown = 0.0
        if self.metrics_history:
            historical_values = [m.total_value_eur for m in self.metrics_history[-100:]]
            if historical_values:
                historical_peak = max(historical_values)
                historical_min = min(historical_values[historical_values.index(max(historical_values)):])
                max_drawdown = (historical_peak - historical_min) / historical_peak if historical_peak > 0 else 0
        
        # Calculate accumulation metrics
        btc_accumulated_today = kwargs.get("btc_accumulated_today", 0.0)
        btc_accumulated_total = kwargs.get("btc_accumulated_total", 0.0)
        
        # Estimate monthly accumulation rate
        if len(self.metrics_history) >= 30:
            thirty_days_ago = self.metrics_history[-30]
            btc_growth = btc_balance - thirty_days_ago.total_btc_balance
            accumulation_rate_monthly = btc_growth / thirty_days_ago.total_btc_balance if thirty_days_ago.total_btc_balance > 0 else 0
        else:
            accumulation_rate_monthly = 0.0
        
        # Calculate Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        # Create metrics object
        metrics = PerformanceMetrics(
            timestamp=current_time,
            total_btc_balance=btc_balance,
            total_eur_balance=eur_balance,
            total_value_eur=total_value_eur,
            btc_price=btc_price,
            trades_today=trades_today,
            total_trades=total_trades,
            successful_trades=successful_trades,
            success_rate=success_rate,
            realized_pnl_eur=kwargs.get("realized_pnl_eur", 0.0),
            unrealized_pnl_eur=kwargs.get("unrealized_pnl_eur", 0.0),
            total_pnl_eur=kwargs.get("total_pnl_eur", 0.0),
            pnl_percentage=kwargs.get("pnl_percentage", 0.0),
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            volatility=kwargs.get("volatility", 0.0),
            sharpe_ratio=sharpe_ratio,
            btc_accumulated_today=btc_accumulated_today,
            btc_accumulated_total=btc_accumulated_total,
            accumulation_rate_monthly=accumulation_rate_monthly
        )
        
        # Add to history
        self.metrics_history.append(metrics)
        
        # Keep only recent history in memory
        if len(self.metrics_history) > 10000:
            self.metrics_history = self.metrics_history[-5000:]
        
        # Check for alerts
        await self._check_alerts(metrics)
        
        # Save metrics periodically
        if len(self.metrics_history) % 10 == 0:
            await self._save_metrics()
        
        return metrics
    
    def _calculate_sharpe_ratio(self, periods: int = 100) -> Optional[float]:
        """Calculate Sharpe ratio from recent performance."""
        if len(self.metrics_history) < periods:
            return None
        
        try:
            recent_values = [m.total_value_eur for m in self.metrics_history[-periods:]]
            returns = np.diff(recent_values) / recent_values[:-1]
            
            if len(returns) < 10:
                return None
            
            # Annualized Sharpe ratio (assuming 15-minute intervals)
            returns_mean = np.mean(returns)
            returns_std = np.std(returns)
            
            if returns_std == 0:
                return None
            
            # Annualize (96 periods per day, 365 days per year)
            sharpe = (returns_mean * 96 * 365) / (returns_std * np.sqrt(96 * 365))
            
            return float(sharpe)
            
        except Exception as e:
            logger.warning(f"Sharpe ratio calculation failed: {e}")
            return None
    
    async def _check_alerts(self, metrics: PerformanceMetrics):
        """Check for alert conditions and send notifications."""
        alerts = []
        
        # Drawdown alerts
        if metrics.current_drawdown > self.alert_thresholds["max_drawdown"]:
            alert_key = f"drawdown_{datetime.now().date()}"
            if alert_key not in self.alerts_sent:
                alerts.append({
                    "type": "HIGH_DRAWDOWN",
                    "message": f"High drawdown detected: {metrics.current_drawdown:.1%}",
                    "severity": "HIGH",
                    "metrics": metrics
                })
                self.alerts_sent.add(alert_key)
        
        # Success rate alerts
        if metrics.total_trades >= 10 and metrics.success_rate < self.alert_thresholds["success_rate_min"]:
            alert_key = f"success_rate_{datetime.now().date()}"
            if alert_key not in self.alerts_sent:
                alerts.append({
                    "type": "LOW_SUCCESS_RATE",
                    "message": f"Low success rate: {metrics.success_rate:.1%}",
                    "severity": "MEDIUM",
                    "metrics": metrics
                })
                self.alerts_sent.add(alert_key)
        
        # Accumulation rate alerts
        if metrics.accumulation_rate_monthly < self.alert_thresholds["accumulation_rate_min"]:
            alert_key = f"accumulation_{datetime.now().date()}"
            if alert_key not in self.alerts_sent:
                alerts.append({
                    "type": "LOW_ACCUMULATION",
                    "message": f"Low accumulation rate: {metrics.accumulation_rate_monthly:.1%}/month",
                    "severity": "MEDIUM",
                    "metrics": metrics
                })
                self.alerts_sent.add(alert_key)
        
        # Send alerts
        for alert in alerts:
            await self._send_alert(alert)
    
    async def _send_alert(self, alert: Dict[str, Any]):
        """Send alert notification."""
        logger.warning(f"ALERT [{alert['type']}]: {alert['message']}")
        
        # Here you could add more sophisticated alerting:
        # - Email notifications
        # - Slack/Discord webhooks
        # - SMS alerts
        # - Trading halt triggers
        
        # Save alert to file
        alert_file = self.data_dir / "alerts.jsonl"
        with open(alert_file, "a") as f:
            alert_record = {
                "timestamp": datetime.now().isoformat(),
                **alert
            }
            # Remove metrics object for JSON serialization
            alert_record.pop("metrics", None)
            f.write(json.dumps(alert_record) + "\n")
    
    async def _save_metrics(self):
        """Save metrics to persistent storage."""
        try:
            metrics_file = self.data_dir / f"metrics_{datetime.now().strftime('%Y%m')}.jsonl"
            
            # Save recent metrics
            recent_metrics = self.metrics_history[-10:]
            with open(metrics_file, "a") as f:
                for metric in recent_metrics:
                    record = asdict(metric)
                    record["timestamp"] = record["timestamp"].isoformat()
                    f.write(json.dumps(record) + "\n")
            
            logger.debug(f"Saved {len(recent_metrics)} metrics to {metrics_file}")
            
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def _load_historical_data(self):
        """Load historical metrics data."""
        try:
            # Load metrics from current month
            current_month = datetime.now().strftime('%Y%m')
            metrics_file = self.data_dir / f"metrics_{current_month}.jsonl"
            
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
                            metrics = PerformanceMetrics(**data)
                            self.metrics_history.append(metrics)
                        except Exception as e:
                            logger.warning(f"Failed to parse metrics line: {e}")
                
                logger.info(f"Loaded {len(self.metrics_history)} historical metrics")
                
                # Update peak value
                if self.metrics_history:
                    self.peak_value = max(m.total_value_eur for m in self.metrics_history)
        
        except Exception as e:
            logger.warning(f"Failed to load historical data: {e}")
    
    def get_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get performance summary for the specified period."""
        if not self.metrics_history:
            return {"error": "No performance data available"}
        
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {"error": f"No data available for the last {days} days"}
        
        latest = recent_metrics[-1]
        earliest = recent_metrics[0]
        
        # Calculate period performance
        period_return = (latest.total_value_eur - earliest.total_value_eur) / earliest.total_value_eur
        btc_accumulated = latest.total_btc_balance - earliest.total_btc_balance
        
        # Calculate average metrics
        avg_success_rate = np.mean([m.success_rate for m in recent_metrics if m.total_trades > 0])
        avg_daily_trades = np.mean([m.trades_today for m in recent_metrics])
        
        return {
            "period_days": days,
            "period_return": period_return,
            "btc_accumulated": btc_accumulated,
            "avg_success_rate": avg_success_rate,
            "avg_daily_trades": avg_daily_trades,
            "current_sharpe_ratio": latest.sharpe_ratio,
            "max_drawdown": latest.max_drawdown,
            "current_drawdown": latest.current_drawdown,
            "total_trades": latest.total_trades - earliest.total_trades,
            "current_accumulation_rate": latest.accumulation_rate_monthly
        }