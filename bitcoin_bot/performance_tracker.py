import json
import numpy as np
from typing import Dict
from datetime import datetime
from utils.logger import logger


class PerformanceTracker:
    """Track and analyze trading bot performance"""

    def __init__(
        self, initial_btc_balance: float = 0.0, initial_eur_balance: float = 0.0
    ):
        self.initial_btc_balance = initial_btc_balance
        self.initial_eur_balance = initial_eur_balance
        self.trades = []
        self.equity_curve = []
        self.performance_file = "./performance_history.json"
        self._load_performance_history()

    def _load_performance_history(self):
        """Load performance history from file"""
        try:
            with open(self.performance_file, "r") as f:
                data = json.load(f)
                self.trades = data.get("trades", [])
                self.equity_curve = data.get("equity_curve", [])
                logger.info(f"Loaded {len(self.trades)} historical trades")
        except FileNotFoundError:
            logger.info("No performance history found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load performance history: {e}")

    def _save_performance_history(self):
        """Save performance history to file"""
        try:
            data = {
                "trades": self.trades[-1000:],  # Keep last 1000 trades
                "equity_curve": self.equity_curve[-10000:],  # Keep last 10000 points
                "last_updated": datetime.now().isoformat(),
            }
            with open(self.performance_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save performance history: {e}")

    def record_trade(
        self,
        order_id: str,
        side: str,
        volume: float,
        price: float,
        fee: float,
        timestamp: float = None,
    ):
        """Record a completed trade"""
        trade = {
            "order_id": order_id,
            "side": side,
            "volume": volume,
            "price": price,
            "fee": fee,
            "timestamp": timestamp or datetime.now().timestamp(),
            "value": volume * price,
        }

        self.trades.append(trade)
        self._save_performance_history()
        logger.debug(f"Recorded {side} trade: {volume:.8f} BTC @ €{price:.2f}")

    def update_equity(
        self, btc_balance: float, eur_balance: float, current_btc_price: float
    ):
        """Update equity curve"""
        total_equity_eur = eur_balance + (btc_balance * current_btc_price)

        equity_point = {
            "timestamp": datetime.now().timestamp(),
            "btc_balance": btc_balance,
            "eur_balance": eur_balance,
            "btc_price": current_btc_price,
            "total_equity_eur": total_equity_eur,
        }

        self.equity_curve.append(equity_point)

        # Save periodically (every 10 updates)
        if len(self.equity_curve) % 10 == 0:
            self._save_performance_history()

    def calculate_returns(self, period_hours: int = 24) -> Dict[str, float]:
        """Calculate returns over a specified period"""
        if not self.equity_curve:
            initial_equity = self.initial_eur_balance + (
                self.initial_btc_balance * 0
            )  # Assume price 0 if no data
            return {
                "period_return": 0.0,
                "total_return": 0.0,
                "current_equity": initial_equity,
                "initial_equity": initial_equity,
            }

        current_equity = self.equity_curve[-1]["total_equity_eur"]
        initial_equity = (
            self.initial_eur_balance
            + self.initial_btc_balance * self.equity_curve[0]["btc_price"]
        )

        # Find equity from period_hours ago
        cutoff_time = datetime.now().timestamp() - (period_hours * 3600)
        period_equity = initial_equity

        for point in self.equity_curve:
            if point["timestamp"] >= cutoff_time:
                period_equity = point["total_equity_eur"]
                break

        period_return = (
            (current_equity - period_equity) / period_equity if period_equity > 0 else 0
        )
        total_return = (
            (current_equity - initial_equity) / initial_equity
            if initial_equity > 0
            else 0
        )

        return {
            "period_return": period_return,
            "total_return": total_return,
            "current_equity": current_equity,
            "initial_equity": initial_equity,
        }

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        returns = []
        for i in range(1, len(self.equity_curve)):
            if self.equity_curve[i - 1]["total_equity_eur"] > 0:
                hourly_return = (
                    self.equity_curve[i]["total_equity_eur"]
                    - self.equity_curve[i - 1]["total_equity_eur"]
                ) / self.equity_curve[i - 1]["total_equity_eur"]
                returns.append(hourly_return)
        if not returns or np.std(returns) == 0:
            return 0.0
        avg_return = np.mean(returns) * 24 * 365  # Annualize hourly returns
        std_return = np.std(returns) * np.sqrt(24 * 365)
        return (avg_return - risk_free_rate) / std_return if std_return > 0 else 0.0

    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve"""
        if len(self.equity_curve) < 2:
            return 0.0

        # Filter out zero equity values
        equities = np.array(
            [
                point["total_equity_eur"]
                for point in self.equity_curve
                if point["total_equity_eur"] > 0
            ]
        )
        if len(equities) < 2:
            logger.warning("Insufficient non-zero equity values to calculate drawdown")
            return 0.0

        peaks = np.maximum.accumulate(equities)
        drawdowns = (equities - peaks) / peaks

        return abs(np.min(drawdowns)) if np.isfinite(drawdowns).any() else 0.0

    def calculate_win_rate(self) -> float:
        """Calculate win rate from trades"""
        if not self.trades:
            return 0.0

        # Group trades by buy/sell pairs
        buy_trades = [t for t in self.trades if t["side"] == "buy"]
        sell_trades = [t for t in self.trades if t["side"] == "sell"]

        if not sell_trades:
            return 0.0  # No completed cycles

        wins = 0
        total = len(sell_trades)

        for sell in sell_trades:
            # Find average buy price before this sell
            relevant_buys = [
                b for b in buy_trades if b["timestamp"] < sell["timestamp"]
            ]
            if relevant_buys:
                avg_buy_price = sum(
                    b["price"] * b["volume"] for b in relevant_buys
                ) / sum(b["volume"] for b in relevant_buys)
                if sell["price"] > avg_buy_price:
                    wins += 1

        return wins / total if total > 0 else 0.0

    def get_trade_statistics(self) -> Dict:
        """Get comprehensive trade statistics"""
        if not self.trades:
            return {
                "total_trades": 0,
                "buy_trades": 0,
                "sell_trades": 0,
                "total_volume": 0.0,
                "total_fees": 0.0,
                "avg_trade_size": 0.0,
            }

        buy_trades = [t for t in self.trades if t["side"] == "buy"]
        sell_trades = [t for t in self.trades if t["side"] == "sell"]

        total_volume = sum(t["volume"] for t in self.trades)
        total_fees = sum(t["fee"] for t in self.trades)

        return {
            "total_trades": len(self.trades),
            "buy_trades": len(buy_trades),
            "sell_trades": len(sell_trades),
            "total_volume": total_volume,
            "total_fees": total_fees,
            "avg_trade_size": total_volume / len(self.trades) if self.trades else 0,
            "avg_buy_price": (
                sum(t["price"] for t in buy_trades) / len(buy_trades)
                if buy_trades
                else 0
            ),
            "avg_sell_price": (
                sum(t["price"] for t in sell_trades) / len(sell_trades)
                if sell_trades
                else 0
            ),
        }

    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        returns = self.calculate_returns(24)  # 24 hour returns
        weekly_returns = self.calculate_returns(168)  # 7 day returns

        report = {
            "timestamp": datetime.now().isoformat(),
            "returns": {
                "24h": f"{returns['period_return']:.2%}",
                "7d": f"{weekly_returns['period_return']:.2%}",
                "total": f"{returns['total_return']:.2%}",
            },
            "risk_metrics": {
                "sharpe_ratio": round(self.calculate_sharpe_ratio(), 2),
                "max_drawdown": f"{self.calculate_max_drawdown():.2%}",
                "win_rate": f"{self.calculate_win_rate():.2%}",
            },
            "trade_stats": self.get_trade_statistics(),
            "equity": {
                "current": f"€{returns.get('current_equity', 0.0):.2f}",
                "initial": f"€{returns.get('initial_equity', 0.0):.2f}",
            },
        }

        return report

    def print_performance_summary(self):
        """Print a formatted performance summary"""
        report = self.generate_performance_report()

        logger.info("=" * 50)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 50)
        logger.info(
            f"Equity: {report['equity']['current']} (from {report['equity']['initial']})"
        )
        logger.info(
            f"Returns: 24h: {report['returns']['24h']}, 7d: {report['returns']['7d']}, Total: {report['returns']['total']}"
        )
        logger.info(f"Sharpe Ratio: {report['risk_metrics']['sharpe_ratio']}")
        logger.info(f"Max Drawdown: {report['risk_metrics']['max_drawdown']}")
        logger.info(f"Win Rate: {report['risk_metrics']['win_rate']}")
        logger.info(
            f"Total Trades: {report['trade_stats']['total_trades']} (Buy: {report['trade_stats']['buy_trades']}, Sell: {report['trade_stats']['sell_trades']})"
        )
        logger.info(f"Total Fees Paid: €{report['trade_stats']['total_fees']:.2f}")
        logger.info("=" * 50)
