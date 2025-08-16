import time
import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands, calculate_moving_average, calculate_vwap, fetch_latest_news, calculate_sentiment, fetch_enhanced_news, calculate_enhanced_sentiment, calculate_risk_adjusted_indicators
from data_manager import DataManager
from order_manager import OrderManager
from trade_executor import TradeExecutor
from onchain_analyzer import OnChainAnalyzer
from config import GLOBAL_TRADE_COOLDOWN, BOT_LOGS_FILE, BITVAVO_CONFIG
from logger_config import logger
from performance_tracker import PerformanceTracker
# from metrics_server import MetricsServer
import requests



class TradingBot:
    def __init__(self, data_manager: DataManager, trade_executor: TradeExecutor, onchain_analyzer: OnChainAnalyzer, order_manager: OrderManager = None):
        # CRITICAL: Assign dependencies FIRST before using them
        self.data_manager = data_manager
        self.trade_executor = trade_executor
        self.onchain_analyzer = onchain_analyzer
        self.order_manager = order_manager
        
        # Then initialize other attributes
        self.max_position_size = 0.15
        self.stop_loss_percentage = 0.03
        self.take_profit_percentage = 0.10
        self.max_daily_trades = 8
        self.daily_trade_count = 0
        self.last_trade_time = 0
        self.max_cash_allocation = 0.9
        self.min_eur_for_trade = 5.0
        self.min_trade_volume = BITVAVO_CONFIG['MIN_ORDER_SIZE']
        self.recent_buys = []
        self._load_recent_buys()
        
        # NOW we can safely use self.trade_executor
        self.performance_tracker = PerformanceTracker(
            initial_btc_balance=self.trade_executor.get_total_btc_balance() or 0,
            initial_eur_balance=self.trade_executor.get_available_balance("EUR") or 0
        )
        
        self.lookback_period = 96  # hours
        self.price_history = []
        self._initialize_price_history()
        self.start_time = time.time()
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model_name = "gemma3:4b"
        self.last_rsi = 0
        self.last_sentiment = 0

        self.last_daily_reset = datetime.now().date()
        self.trade_session_file = "./trade_session.json"

    def _initialize_price_history(self):
        prices, _ = self.data_manager.load_price_history()
        if not prices:
            logger.warning("No price history available. Fetching initial data...")
            self._fetch_historical_data()
            prices, _ = self.data_manager.load_price_history()

        required_candles = self.lookback_period * 4  # 15-min candles
        current_time = int(time.time())
        required_time = current_time - (self.lookback_period * 3600)

        if prices:
            with open(self.data_manager.price_history_file, 'r') as f:
                data = json.load(f)
            if data and isinstance(data, list) and len(data) > 0 and isinstance(data[0], list) and len(data[0]) > 0:
                earliest_timestamp = data[0][0]
                if earliest_timestamp > required_time:
                    logger.info(f"Price history only goes back to {datetime.fromtimestamp(earliest_timestamp).isoformat()}. Fetching more data...")
                    self._fetch_historical_data(since=earliest_timestamp - (self.lookback_period * 3600))
                    prices, _ = self.data_manager.load_price_history()

        self._load_price_history_for_lookback()

    def _fetch_historical_data(self, since: Optional[int] = None):
        if since is None:
            since = int(time.time()) - (self.lookback_period * 3600 * 2)
        logger.info(f"Fetching historical OHLC data from {datetime.fromtimestamp(since).isoformat()}")
        # Updated to use new trade_executor method
        ohlc = self.trade_executor.get_ohlc_data(pair="BTC/EUR", interval='15m', since=int(time.time() - 7200))
        if ohlc:
            self.data_manager.append_ohlc_data(ohlc)

    def _load_price_history_for_lookback(self):
        try:
            prices, _ = self.data_manager.load_price_history()
            if prices:
                num_candles = self.lookback_period * 4
                self.price_history = [float(p) for p in prices[-num_candles:]]
                logger.info(f"Loaded {len(self.price_history)} prices into price_history for lookback")
                if self.price_history:
                    logger.info(f"Price history range: min={min(self.price_history):.2f}, max={max(self.price_history):.2f}")
                else:
                    logger.warning("Price history is empty after loading")
        except Exception as e:
            logger.error(f"Failed to load price history for lookback: {e}")

    

    def _detect_market_regime(self, prices: List[float]) -> str:
        if len(prices) < 50:
            return "unknown"
        ma_50 = calculate_moving_average(prices, 50)
        ma_200 = calculate_moving_average(prices, 200) if len(prices) >= 200 else ma_50
        if ma_50 > ma_200 * 1.02:
            return "strong_uptrend"
        elif ma_50 < ma_200 * 0.98:
            return "strong_downtrend"
        else:
            return "ranging"

    def _load_recent_buys(self):
        try:
            recent_buys_file = "recent_buys.json"
            if os.path.exists(recent_buys_file):
                with open(recent_buys_file, 'r') as f:
                    self.recent_buys = json.load(f)
                logger.debug(f"Loaded {len(self.recent_buys)} recent buys from {recent_buys_file}")
        except Exception as e:
            logger.error(f"Failed to load recent_buys: {e}")

    def _save_recent_buys(self):
        try:
            recent_buys_file = "recent_buys.json"
            with open(recent_buys_file, 'w') as f:
                json.dump(self.recent_buys, f)
            logger.debug(f"Saved {len(self.recent_buys)} recent buys to {recent_buys_file}")
        except Exception as e:
            logger.error(f"Failed to save recent_buys: {e}")

    def _calculate_volatility(self, prices: List[float]) -> float:
        if len(prices) < 2:
            return 0.0
        returns = np.diff(prices) / prices[:-1]
        return float(np.std(returns[-20:])) if len(returns) >= 20 else 0.01

    def _estimate_avg_buy_price(self) -> Optional[float]:
        try:
            if not os.path.exists(BOT_LOGS_FILE):
                logger.error(f"Bot logs file not found at {BOT_LOGS_FILE}")
                return self._fallback_avg_buy_price()
            logger.debug(f"Reading bot_logs.csv from {BOT_LOGS_FILE}")
            df = pd.read_csv(BOT_LOGS_FILE, dtype={"buy_decision": str, "sell_decision": str}, encoding='utf-8', on_bad_lines='warn')
            logger.debug(f"CSV shape: {df.shape}")
            expected_columns = ["timestamp", "price", "trade_volume", "side", "buy_decision"]
            if not all(col in df.columns for col in expected_columns):
                logger.error(f"Missing required columns in bot_logs.csv: {set(expected_columns) - set(df.columns)}")
                return self._fallback_avg_buy_price()
            df['timestamp'] = df['timestamp'].astype(str)
            valid_rows = df[df['timestamp'].notna()]
            buy_trades = df[
                (df["buy_decision"].str.lower().isin(["true", "1"])) & 
                (df["side"].astype(str).str.lower() == "buy")
            ]
            if buy_trades.empty:
                logger.debug("No valid buy trades found in logs")
                return self._fallback_avg_buy_price()
            buy_trades = buy_trades.copy()
            buy_trades["price"] = pd.to_numeric(buy_trades["price"], errors="coerce")
            buy_trades["trade_volume"] = pd.to_numeric(buy_trades["trade_volume"], errors="coerce")
            buy_trades = buy_trades.dropna(subset=["price", "trade_volume"])
            if buy_trades.empty:
                logger.debug("No valid buy trades after numeric conversion")
                return self._fallback_avg_buy_price()
            total_cost = (buy_trades["price"] * buy_trades["trade_volume"]).sum()
            total_volume = buy_trades["trade_volume"].sum()
            avg_price = total_cost / total_volume if total_volume > 0 else None
            logger.debug(f"Estimated avg buy price from logs: {avg_price}")
            return avg_price
        except Exception as e:
            logger.error(f"Failed to estimate avg buy price: {e}", exc_info=True)
            return self._fallback_avg_buy_price()

    def _fallback_avg_buy_price(self) -> Optional[float]:
        if not self.recent_buys:
            logger.debug("No recent buys available")
            return None
        total_cost = sum(price * volume for price, volume in self.recent_buys)
        total_volume = sum(volume for _, volume in self.recent_buys)
        avg_price = total_cost / total_volume if total_volume > 0 else None
        logger.debug(f"Fallback avg buy price from recent_buys: {avg_price}")
        return avg_price

    def enhanced_decide_action_with_risk_override(self, indicators_data: Dict) -> str:
        """
        Advanced decision making with comprehensive risk management.
        """
        # Extract all key indicators
        news_analysis = indicators_data.get('news_analysis', {})
        risk_off_prob = news_analysis.get('risk_off_probability', 0)
        sentiment = indicators_data.get('sentiment', 0)
        current_price = indicators_data.get('current_price', 0)
        avg_buy_price = indicators_data.get('avg_buy_price', 0)
        market_trend = indicators_data.get('market_trend', 'ranging')
        netflow = indicators_data.get('netflow', 0)
        rsi = indicators_data.get('rsi', 50)
        macd = indicators_data.get('macd', 0)
        signal = indicators_data.get('signal', 0)
        vwap = indicators_data.get('vwap', current_price)
        volatility = indicators_data.get('volatility', 0)
        
        # Performance metrics
        performance = indicators_data.get('performance_report', {})
        win_rate = float(performance.get('risk_metrics', {}).get('win_rate', '0%').rstrip('%')) / 100
        total_return = performance.get('returns', {}).get('total', '0%')
        
        # Calculate profit/loss position
        if avg_buy_price and avg_buy_price > 0:
            profit_margin = (current_price - avg_buy_price) / avg_buy_price * 100
        else:
            profit_margin = 0
        
        # Calculate position relative to VWAP
        vwap_distance = (current_price - vwap) / vwap * 100
        
        logger.info(f"🔍 ANALYSIS - Price: €{current_price:.0f}, P&L: {profit_margin:.1f}%, "
                    f"Risk-off: {risk_off_prob*100:.0f}%, Sentiment: {sentiment:.3f}, "
                    f"RSI: {rsi:.1f}, VWAP: {vwap_distance:+.1f}%")
        
        # EMERGENCY CONDITIONS - Override everything else
        
        # 1. EXTREME RISK CONDITIONS
        if risk_off_prob > 0.8:  # 80%+ risk-off probability
            logger.error(f"🚨 EXTREME RISK: Risk-off {risk_off_prob*100:.0f}% - EMERGENCY SELL")
            return 'sell'
        
        # 2. LIQUIDATION CASCADE WARNING
        if volatility > 0.08 and sentiment < -0.1 and current_price < vwap * 0.95:
            logger.error(f"🚨 LIQUIDATION CASCADE: High vol + negative sentiment + below VWAP - SELL")
            return 'sell'
        
        # HIGH RISK CONDITIONS - Very conservative approach
        
        # 3. HIGH MACRO RISK
        if risk_off_prob > 0.6:  # 60%+ risk-off probability
            if profit_margin > 2:  # If profitable, take some profits
                logger.warning(f"⚠️ HIGH MACRO RISK: Risk-off {risk_off_prob*100:.0f}% + profitable - SELL")
                return 'sell'
            else:
                logger.warning(f"⚠️ HIGH MACRO RISK: Risk-off {risk_off_prob*100:.0f}% - NO NEW POSITIONS")
                return 'hold'
        
        # 4. STRONG DOWNTREND + UNDERWATER
        if market_trend == 'strong_downtrend' and profit_margin < -2:
            if rsi > 60:  # Overbought in downtrend = bad
                logger.warning(f"📉 DOWNTREND + UNDERWATER + OVERBOUGHT - SELL")
                return 'sell'
            else:
                logger.warning(f"📉 STRONG DOWNTREND + UNDERWATER - HOLD")
                return 'hold'
        
        # 5. POOR PERFORMANCE ADJUSTMENT
        if win_rate < 0.3 and total_return.startswith('-'):  # Win rate < 30% and negative returns
            # Be more conservative
            if risk_off_prob > 0.4:  # Lower threshold when performing poorly
                logger.warning(f"📊 POOR PERFORMANCE: Win rate {win_rate*100:.0f}% - DEFENSIVE MODE")
                return 'hold'
        
        # MODERATE RISK CONDITIONS
        
        # 6. MODERATE RISK - SELECTIVE TRADING
        if risk_off_prob > 0.4:  # 40-60% risk-off probability
            # Only trade in very favorable conditions
            strong_buy_signals = sum([
            rsi < 35 if rsi is not None else False,  # Very oversold
            current_price < vwap * 0.98 if current_price and vwap else False,  # Significantly below VWAP
            netflow < -10000 if netflow is not None else False,  # Very strong accumulation
            sentiment > 0.05 if sentiment is not None else False,  # Positive sentiment despite risk
            (macd > signal) if (macd is not None and signal is not None) else False,  # MACD bullish crossover
        ])
            
            if strong_buy_signals >= 4:  # Need 4/5 strong signals
                logger.info(f"⚡ MODERATE RISK BUT STRONG SIGNALS ({strong_buy_signals}/5) - BUY")
                return 'buy'
            
            # Check for profit taking
            if profit_margin > 5:
                logger.info(f"💰 MODERATE RISK + PROFITABLE ({profit_margin:.1f}%) - TAKE PROFITS")
                return 'sell'
            
            logger.info(f"⚠️ MODERATE RISK: Only {strong_buy_signals}/5 strong signals - HOLD")
            return 'hold'
        
        # LOW RISK CONDITIONS - Normal trading
        
        # 7. PROFIT TAKING RULES
        if profit_margin > 8:  # 8%+ profit
            logger.info(f"💰 STRONG PROFITS: {profit_margin:.1f}% - TAKE PROFITS")
            return 'sell'
        elif profit_margin > 5 and (rsi > 70 or current_price > vwap * 1.02):
            logger.info(f"💰 GOOD PROFITS + OVERBOUGHT: {profit_margin:.1f}% - TAKE PROFITS")
            return 'sell'
        
        # 8. BUY CONDITIONS (Low risk environment)
        # Replace the buy_signals list around line 330 in enhanced_decide_action_with_risk_override

        buy_signals = [
            rsi < 45,  # Oversold or neutral
            current_price < vwap,  # Below VWAP
            netflow < -3000,  # Accumulation happening
            sentiment > -0.05,  # Not too negative
            # FIXED: Handle None values for MACD
            (macd is not None and signal is not None and macd > signal) or (macd is None or signal is None),  # MACD neutral or bullish
        ]
        
        
        buy_score = sum(buy_signals)
        
        # Additional boost for very oversold conditions
        if rsi < 30:
            buy_score += 1
            logger.info(f"🔥 VERY OVERSOLD BOOST: RSI {rsi:.1f}")
        
        # Additional boost for strong accumulation
        if netflow < -8000:
            buy_score += 1
            logger.info(f"🐋 STRONG ACCUMULATION BOOST: Netflow {netflow:.0f}")
        
        logger.info(f"📊 BUY SIGNALS: {buy_score}/5 base + bonuses")
        
        if buy_score >= 4:  # Need 4+ signals for buy
            logger.info(f"✅ BUY CONDITIONS MET: {buy_score} signals")
            return 'buy'
        
        # 9. SELL CONDITIONS (Technical)
        sell_signals = [
        rsi > 75,  # Very overbought
        current_price > vwap * 1.05,  # 5% above VWAP
        market_trend == 'strong_downtrend',
        sentiment < -0.1,  # Very negative sentiment
        # macd < signal and signal > 0,  # Bearish MACD crossover - COMMENTED OUT FOR NOW
]
        
        sell_score = sum(sell_signals)
        
        if sell_score >= 3 and profit_margin > 0:  # Need 3+ signals and be profitable
            logger.info(f"📉 TECHNICAL SELL: {sell_score}/5 signals + profitable")
            return 'sell'
        
        # 10. DEFAULT HOLD WITH REASONING
        hold_reasons = []
        if buy_score < 4:
            hold_reasons.append(f"insufficient buy signals ({buy_score}/4)")
        if risk_off_prob > 0.3:
            hold_reasons.append(f"elevated risk ({risk_off_prob*100:.0f}%)")
        if profit_margin < 0 and sell_score < 3:
            hold_reasons.append("underwater position, waiting for recovery")
        
        logger.info(f"⏸️ HOLD: {', '.join(hold_reasons) if hold_reasons else 'market conditions neutral'}")
        return 'hold'

    def calculate_risk_adjusted_position_size(self, action: str, indicators_data: Dict, 
                                            btc_balance: float, eur_balance: float) -> float:
        """
        Calculate position size based on comprehensive risk assessment.
        """
        if action == 'hold':
            return 0.0
        
        # Base position sizes
        base_buy_pct = 0.08  # 8% of EUR balance
        base_sell_pct = 0.12  # 12% of BTC balance
        
        # Risk adjustments
        news_analysis = indicators_data.get('news_analysis', {})
        risk_off_prob = news_analysis.get('risk_off_probability', 0)
        volatility = indicators_data.get('volatility', 0.02)
        
        # Performance adjustment
        performance = indicators_data.get('performance_report', {})
        win_rate = float(performance.get('risk_metrics', {}).get('win_rate', '0%').rstrip('%')) / 100
        
        # Calculate risk multiplier
        risk_multiplier = 1.0
        
        # Reduce size for high risk-off probability
        if risk_off_prob > 0.4:
            risk_multiplier *= (1 - risk_off_prob)  # Scale down proportionally
        
        # Reduce size for high volatility
        if volatility > 0.05:
            risk_multiplier *= 0.7
        
        # Reduce size for poor performance
        if win_rate < 0.3:
            risk_multiplier *= 0.6
        
        # Increase size for very favorable conditions
        rsi = indicators_data.get('rsi', 50)
        netflow = indicators_data.get('netflow', 0)
        
        if action == 'buy' and rsi < 30 and netflow < -8000 and risk_off_prob < 0.2:
            risk_multiplier *= 1.5  # Aggressive buying in very favorable conditions
        
        # Calculate final position
        if action == 'buy':
            position_eur = eur_balance * base_buy_pct * risk_multiplier
            current_price = indicators_data.get('current_price', 1)
            position_btc = position_eur / current_price
            
            # Ensure we have enough EUR and respect minimums
            max_affordable = eur_balance * 0.9 / current_price
            position_btc = min(position_btc, max_affordable)
            position_btc = max(position_btc, self.min_trade_volume)
            
        elif action == 'sell':
            position_btc = btc_balance * base_sell_pct * risk_multiplier
            position_btc = min(position_btc, btc_balance * 0.8)  # Max 80% of holdings
            position_btc = max(position_btc, self.min_trade_volume)
        
        else:
            position_btc = 0.0
        
        logger.info(f"📏 POSITION SIZING: Action={action}, Risk multiplier={risk_multiplier:.2f}, "
                    f"Size={position_btc:.8f} BTC")
        
        return position_btc

    def decide_amount(self, action: str, indicators_data: Dict, btc_balance: float, eur_balance: float) -> float:
        """
        Calculate position size using the enhanced risk-adjusted method.
        """
        return self.calculate_risk_adjusted_position_size(action, indicators_data, btc_balance, eur_balance)

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json().get('response', '').strip()
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return '' 
    
    

    def _log_trade(self, timestamp: str, price: float, volume: float, side: str, reason: str):
        self.data_manager.log_strategy(
            timestamp=timestamp,
            price=price,
            trade_volume=volume,
            side=side,
            reason=reason,
            buy_decision=side == "buy",
            sell_decision=side == "sell"
        )

    def should_wait_for_pending_orders(self, side: str) -> bool:
        if not self.order_manager:
            return False
        pending = self.order_manager.get_pending_orders()
        same_side_pending = [o for o in pending.values() if o['side'] == side]
        if same_side_pending:
            logger.info(f"⏳ Waiting for {len(same_side_pending)} pending {side} order(s) to fill before placing new ones")
            return True
        return False

    

    def log_risk_decision(self, action: str, indicators_data: Dict, reasoning: str):
        """
        Log detailed risk analysis for monitoring and backtesting.
        """
        import json
        from datetime import datetime
        
        risk_log_file = "./risk_decisions.json"
        
        # Extract key risk factors
        news_analysis = indicators_data.get('news_analysis', {})
        risk_off_prob = news_analysis.get('risk_off_probability', 0)
        sentiment = indicators_data.get('sentiment', 0)
        current_price = indicators_data.get('current_price', 0)
        avg_buy_price = indicators_data.get('avg_buy_price', 0)
        profit_margin = ((current_price - avg_buy_price) / avg_buy_price * 100) if avg_buy_price and avg_buy_price > 0 else 0
        
        risk_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "reasoning": reasoning,
            "market_data": {
                "price": current_price,
                "profit_margin": profit_margin,
                "rsi": indicators_data.get('rsi', 50),
                "market_trend": indicators_data.get('market_trend', 'unknown'),
                "vwap_distance": ((current_price - indicators_data.get('vwap', current_price)) 
                                / indicators_data.get('vwap', current_price) * 100)
            },
            "risk_factors": {
                "risk_off_probability": risk_off_prob,
                "sentiment": sentiment,
                "netflow": indicators_data.get('netflow', 0),
                "volatility": indicators_data.get('volatility', 0),
                "macro_articles": news_analysis.get('macro_articles', 0),
                "total_articles": news_analysis.get('total_articles', 0)
            },
            "performance": {
                "win_rate": indicators_data.get('performance_report', {}).get('risk_metrics', {}).get('win_rate', '0%'),
                "total_return": indicators_data.get('performance_report', {}).get('returns', {}).get('total', '0%')
            }
        }
        
        # Load existing log
        try:
            with open(risk_log_file, 'r') as f:
                risk_log = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            risk_log = []
        
        # Add new entry
        risk_log.append(risk_entry)
        
        # Keep only last 1000 entries
        risk_log = risk_log[-1000:]
        
        # Save updated log
        try:
            with open(risk_log_file, 'w') as f:
                json.dump(risk_log, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save risk log: {e}")

    def get_risk_summary(self) -> Dict:
        """
        Get summary of recent risk decisions for monitoring.
        """
        import json
        from datetime import datetime, timedelta
        
        risk_log_file = "./risk_decisions.json"
        
        try:
            with open(risk_log_file, 'r') as f:
                risk_log = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"error": "No risk log available"}
        
        if not risk_log:
            return {"error": "Empty risk log"}
        
        # Analyze last 24 hours
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_decisions = [
            entry for entry in risk_log 
            if datetime.fromisoformat(entry['timestamp']) > cutoff_time
        ]
        
        if not recent_decisions:
            return {"error": "No recent decisions"}
        
        # Calculate statistics
        total_decisions = len(recent_decisions)
        action_counts = {}
        risk_levels = []
        
        for decision in recent_decisions:
            action = decision['action']
            action_counts[action] = action_counts.get(action, 0) + 1
            risk_levels.append(decision['risk_factors']['risk_off_probability'])
        
        avg_risk_level = sum(risk_levels) / len(risk_levels) if risk_levels else 0
        max_risk_level = max(risk_levels) if risk_levels else 0
        
        return {
            "total_decisions_24h": total_decisions,
            "action_breakdown": action_counts,
            "avg_risk_off_probability": avg_risk_level,
            "max_risk_off_probability": max_risk_level,
            "latest_decision": recent_decisions[-1] if recent_decisions else None,
            "risk_status": (
                "HIGH RISK" if avg_risk_level > 0.6 else
                "MODERATE RISK" if avg_risk_level > 0.4 else
                "LOW RISK"
            )
        }

    def implement_dynamic_stop_loss(self, entry_price: float, side: str, indicators_data: Dict) -> float:
        """
        Implement dynamic stop-loss based on market conditions.
        """
        base_stop_pct = 0.03  # 3% base stop loss
        
        # Get risk factors
        news_analysis = indicators_data.get('news_analysis', {})
        liquidation_signals = indicators_data.get('liquidation_signals', {})
        volatility = indicators_data.get('volatility', 0.02)
        
        # Tighten stops during high risk periods
        risk_off_prob = news_analysis.get('risk_off_probability', 0)
        if risk_off_prob > 0.6:
            base_stop_pct = 0.02  # 2% stop loss
        elif risk_off_prob > 0.3:
            base_stop_pct = 0.025  # 2.5% stop loss
        
        # Adjust for liquidation risk
        cascade_prob = liquidation_signals.get('cascade_probability', 0)
        if cascade_prob > 0.5:
            base_stop_pct = 0.015  # Very tight 1.5% stop
        
        # Adjust for volatility to avoid whipsaws
        volatility_adjustment = min(2.0, 1 + volatility * 5)  # Up to 2x adjustment
        final_stop_pct = base_stop_pct * volatility_adjustment
        
        # Calculate stop price
        if side == 'buy':
            stop_price = entry_price * (1 - final_stop_pct)
        else:  # sell
            stop_price = entry_price * (1 + final_stop_pct)
        
        logger.info(f"Dynamic stop-loss - Entry: €{entry_price:.2f}, Stop: €{stop_price:.2f}, "
                    f"Distance: {final_stop_pct*100:.1f}%")
        
        return stop_price

    def check_risk_status(self):
        """
        Quick CLI command to check current risk status.
        Usage: Add this as a method and call it manually when needed.
        """
        risk_summary = self.get_risk_summary()
        
        print("\n=== RISK STATUS SUMMARY ===")
        print(f"Status: {risk_summary.get('risk_status', 'UNKNOWN')}")
        print(f"24h Decisions: {risk_summary.get('total_decisions_24h', 0)}")
        print(f"Actions: {risk_summary.get('action_breakdown', {})}")
        print(f"Avg Risk Level: {risk_summary.get('avg_risk_off_probability', 0)*100:.1f}%")
        print(f"Max Risk Level: {risk_summary.get('max_risk_off_probability', 0)*100:.1f}%")
        
        latest = risk_summary.get('latest_decision')
        if latest:
            print(f"\nLatest Decision: {latest['action'].upper()} at {latest['timestamp']}")
            print(f"Reasoning: {latest['reasoning']}")
        print("===========================\n")

    def _load_trade_session(self):
        """Load daily trade count and session data"""
        try:
            if os.path.exists(self.trade_session_file):
                with open(self.trade_session_file, 'r') as f:
                    session_data = json.load(f)
                    
                session_date = datetime.fromisoformat(session_data.get('date', '1970-01-01')).date()
                if session_date == datetime.now().date():
                    self.daily_trade_count = session_data.get('daily_trade_count', 0)
                    logger.info(f"Loaded session: {self.daily_trade_count} trades today")
                else:
                    logger.info("New trading day started, resetting counters")
                    self.daily_trade_count = 0
                    self._save_trade_session()
        except Exception as e:
            logger.error(f"Failed to load trade session: {e}")
            self.daily_trade_count = 0

    def _save_trade_session(self):
        """Save daily trade count and session data"""
        try:
            session_data = {
                'date': datetime.now().date().isoformat(),
                'daily_trade_count': self.daily_trade_count,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.trade_session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save trade session: {e}")

    def _reset_daily_counters_if_needed(self):
        """Reset daily counters if it's a new day"""
        current_date = datetime.now().date()
        if current_date != self.last_daily_reset:
            logger.info("New trading day - resetting daily counters")
            self.daily_trade_count = 0
            self.last_daily_reset = current_date
            self._save_trade_session()

    def check_pending_orders(self):
        """Enhanced order checking with better trade recording"""
        if not self.order_manager:
            return
            
        try:
            self._reset_daily_counters_if_needed()
            
            logger.debug("Checking pending orders...")
            results = self.order_manager.check_and_update_orders()
            
            # Process filled orders
            if results['filled']:
                logger.info(f"🎯 Orders FILLED: {results['filled']}")
                
                for order_id in results['filled']:
                    order_info = self.order_manager.filled_orders.get(order_id)
                    if order_info:
                        # Record in performance tracker
                        self.performance_tracker.record_trade(
                            order_id=order_id,
                            side=order_info['side'],
                            volume=order_info.get('executed_volume', order_info['volume']),
                            price=order_info.get('average_price', order_info['price']),
                            fee=order_info.get('fee', 0),
                            timestamp=order_info.get('filled_at', time.time())
                        )
                        
                        # Update daily counter
                        self.daily_trade_count += 1
                        self._save_trade_session()
                        
                        # Update recent buys for profit calculation
                        if order_info['side'] == 'buy':
                            self.recent_buys.append((
                                order_info.get('average_price', order_info['price']),
                                order_info.get('executed_volume', order_info['volume'])
                            ))
                            self.recent_buys = self.recent_buys[-10:]  # Keep last 10
                            self._save_recent_buys()
                            
                            logger.info(f"✅ Buy recorded: {order_info.get('executed_volume', 0):.8f} BTC @ €{order_info.get('average_price', 0):.2f}")
                            
                        elif order_info['side'] == 'sell':
                            # Calculate and log profit
                            avg_buy_price = self._estimate_avg_buy_price()
                            if avg_buy_price:
                                profit_pct = (order_info.get('average_price', 0) - avg_buy_price) / avg_buy_price * 100
                                logger.info(f"💰 Sell completed with {profit_pct:.2f}% profit")
                            
                            logger.info(f"✅ Sell recorded: {order_info.get('executed_volume', 0):.8f} BTC @ €{order_info.get('average_price', 0):.2f}")
                        
                        # Log the trade in our detailed logs
                        self._log_completed_trade(order_info)
            
            # Log other order states
            if results['cancelled']:
                logger.info(f"❌ Orders CANCELLED: {results['cancelled']}")
                
            if results['partial']:
                logger.info(f"📊 Orders PARTIALLY filled: {results['partial']}")
                for order_id in results['partial']:
                    if order_id in self.order_manager.pending_orders:
                        order_info = self.order_manager.pending_orders[order_id]
                        executed = order_info.get('executed_volume', 0)
                        total = order_info['volume']
                        pct_filled = (executed / total * 100) if total > 0 else 0
                        logger.info(f"  {order_id}: {executed:.8f}/{total:.8f} BTC ({pct_filled:.1f}% filled)")
                        
        except Exception as e:
            logger.error(f"Error checking pending orders: {e}", exc_info=True)

    def _log_completed_trade(self, order_info: Dict):
        """Log completed trade with full details"""
        try:
            trade_details = {
                'timestamp': datetime.now().isoformat(),
                'order_id': order_info['id'],
                'side': order_info['side'],
                'volume': order_info.get('executed_volume', order_info['volume']),
                'price': order_info.get('average_price', order_info['price']),
                'fee': order_info.get('fee', 0),
                'total_value': order_info.get('executed_volume', order_info['volume']) * order_info.get('average_price', order_info['price']),
                'fill_time_seconds': order_info.get('filled_at', 0) - order_info.get('timestamp', 0),
                'daily_trade_number': self.daily_trade_count
            }
            
            # Log to our main data manager
            self.data_manager.log_strategy(
                timestamp=trade_details['timestamp'],
                price=trade_details['price'],
                trade_volume=trade_details['volume'],
                side=trade_details['side'],
                reason=f"ORDER_FILLED: {order_info['id']}",
                buy_decision=trade_details['side'] == 'buy',
                sell_decision=trade_details['side'] == 'sell'
            )
            
            logger.info(f"📝 Trade logged: {trade_details['side'].upper()} #{trade_details['daily_trade_number']}")
            
        except Exception as e:
            logger.error(f"Failed to log completed trade: {e}")

    def get_order_summary(self):
        """Enhanced order summary with better statistics"""
        if not self.order_manager:
            logger.info("No order manager available")
            return
            
        try:
            # Current pending orders
            pending = self.order_manager.get_pending_orders()
            if pending:
                logger.info(f"=== PENDING ORDERS ({len(pending)}) ===")
                total_buy_volume = 0
                total_sell_volume = 0
                total_buy_value = 0
                total_sell_value = 0
                
                for order_id, order_info in pending.items():
                    age = time.time() - order_info['timestamp']
                    remaining = max(0, order_info['timeout'] - age)
                    executed = order_info.get('executed_volume', 0)
                    total = order_info['volume']
                    value = total * order_info['price']
                    
                    status_indicator = "🟡" if executed == 0 else "🔵"  # Yellow for new, blue for partial
                    
                    logger.info(f"  {status_indicator} {order_id}: {order_info['side'].upper()} "
                              f"{total:.8f} BTC @ €{order_info['price']:.2f} "
                              f"(€{value:.2f}) - expires in {remaining:.0f}s")
                    
                    if executed > 0:
                        pct_filled = (executed / total * 100) if total > 0 else 0
                        logger.info(f"    📊 Partially filled: {executed:.8f} BTC ({pct_filled:.1f}%)")
                    
                    if order_info['side'] == 'buy':
                        total_buy_volume += total
                        total_buy_value += value
                    else:
                        total_sell_volume += total
                        total_sell_value += value
                
                logger.info(f"  📊 Total pending: BUY {total_buy_volume:.8f} BTC (€{total_buy_value:.2f}), "
                          f"SELL {total_sell_volume:.8f} BTC (€{total_sell_value:.2f})")
            
            # Recent filled orders
            recent_fills = self.order_manager.get_filled_orders(hours=24)
            if recent_fills:
                logger.info(f"=== RECENT FILLS (Last 24h: {len(recent_fills)}) ===")
                total_buy_volume = 0
                total_sell_volume = 0
                total_fees = 0
                
                # Show last 5 fills
                recent_items = list(recent_fills.items())[-5:]
                for order_id, order_info in recent_items:
                    fill_time = datetime.fromtimestamp(order_info.get('filled_at', 0))
                    executed_vol = order_info.get('executed_volume', order_info['volume'])
                    avg_price = order_info.get('average_price', order_info['price'])
                    fee = order_info.get('fee', 0)
                    value = executed_vol * avg_price
                    
                    side_indicator = "🟢" if order_info['side'] == 'buy' else "🔴"
                    
                    logger.info(f"  {side_indicator} {order_info['side'].upper()} {executed_vol:.8f} BTC @ "
                              f"€{avg_price:.2f} (€{value:.2f}) at {fill_time.strftime('%H:%M:%S')}")
                    
                    if order_info['side'] == 'buy':
                        total_buy_volume += executed_vol
                    else:
                        total_sell_volume += executed_vol
                    total_fees += fee
                
                logger.info(f"  📊 24h totals: BUY {total_buy_volume:.8f} BTC, SELL {total_sell_volume:.8f} BTC")
                logger.info(f"  💸 24h fees: €{total_fees:.4f}")
            
            # Order statistics
            stats = self.order_manager.get_order_statistics()
            logger.info(f"=== ORDER STATISTICS ===")
            logger.info(f"  📈 Fill rate: {stats['fill_rate']:.1%}")
            logger.info(f"  ⏱️  Avg fill time: {stats['avg_time_to_fill']:.0f}s")
            logger.info(f"  💰 Total fees paid: €{stats['total_fees_paid']:.4f}")
            logger.info(f"  📊 Total orders: {stats['total_filled_orders']} filled, {stats['total_cancelled_orders']} cancelled")
            logger.info(f"  🔄 Today's trades: {self.daily_trade_count}")
            
            # Recent trade count verification
            recent_count = self.order_manager.get_recent_trade_count(24)
            if recent_count != len(recent_fills):
                logger.warning(f"⚠️ Trade count mismatch: recent_count={recent_count}, recent_fills={len(recent_fills)}")
            
        except Exception as e:
            logger.error(f"Error generating order summary: {e}")

    def execute_strategy(self):
        """Enhanced strategy execution with better error handling and logging"""
        try:
            # Reset daily counters if needed
            self._reset_daily_counters_if_needed()
            
            # Check pending orders first
            self.check_pending_orders()
            
            # Get order summary for context
            self.get_order_summary()
            
            # Load price data
            prices, volumes = self.data_manager.load_price_history()
            if not prices or len(prices) < 10:
                logger.warning("Insufficient price data for strategy execution")
                return

            # Get current market data
            current_price, current_volume = self.trade_executor.fetch_current_price()
            if not current_price:
                logger.error("Failed to fetch current price")
                return

            # Update price history
            self.price_history.append(current_price)
            self.price_history = self.price_history[-self.lookback_period * 4:]

            # Fetch fresh OHLC data
            ohlc = self.trade_executor.get_ohlc_data(pair="BTC/EUR", interval='15m', since=int(time.time() - 7200))
            if ohlc:
                added_candles = self.data_manager.append_ohlc_data(ohlc)
                logger.debug(f"Added {added_candles} new OHLC candles")

            # Prepare data for analysis
            prices = [float(p) for p in prices]
            volumes = [float(v) for v in volumes]
            
            if len(prices) != len(volumes):
                logger.error(f"Price/volume length mismatch: {len(prices)} vs {len(volumes)}")
                return

            # Calculate technical indicators
            rsi = calculate_rsi(prices) or 50
            macd, signal = calculate_macd(prices) or (0, 0)
            upper_band, ma_short, lower_band = calculate_bollinger_bands(prices) or (current_price, current_price, current_price)
            ma_long = calculate_moving_average(prices, 50) or current_price
            vwap = calculate_vwap(prices, volumes) or current_price
            volatility = self._calculate_volatility(prices)
            market_trend = self._detect_market_regime(prices)

            # Enhanced news and sentiment analysis
            try:
                articles = fetch_enhanced_news(top_n=20)
                news_analysis = calculate_enhanced_sentiment(articles)
                sentiment = news_analysis.get('sentiment', 0)
                
                logger.info(f"📰 News analysis: {news_analysis.get('total_articles', 0)} articles, "
                          f"sentiment: {sentiment:.3f}, risk-off: {news_analysis.get('risk_off_probability', 0)*100:.0f}%")
            except Exception as e:
                logger.warning(f"News analysis failed: {e}")
                news_analysis = {'sentiment': 0, 'risk_off_probability': 0, 'macro_weight': 1.0}
                sentiment = 0

            # Enhanced risk-adjusted indicators
            try:
                enhanced_indicators = calculate_risk_adjusted_indicators(prices, volumes, news_analysis)
            except Exception as e:
                logger.warning(f"Enhanced indicators failed, using basic ones: {e}")
                enhanced_indicators = {
                    'rsi': rsi, 'macd': macd, 'signal': signal,
                    'ma_short': ma_short, 'ma_long': ma_long, 'vwap': vwap,
                    'correlations': {}, 'liquidation_signals': {}, 'risk_factor': 1.0
                }

            # Get on-chain signals
            try:
                onchain_signals = self.onchain_analyzer.get_onchain_signals()
            except Exception as e:
                logger.warning(f"On-chain analysis failed: {e}")
                onchain_signals = {"fee_rate": 0, "netflow": 0, "volume": 0, "old_utxos": 0}

            # Get current balances
            btc_balance = self.trade_executor.get_total_btc_balance() or 0
            eur_balance = self.trade_executor.get_available_balance("EUR") or 0

            # Update performance tracking
            self.performance_tracker.update_equity(btc_balance, eur_balance, current_price)
            performance_report = self.performance_tracker.generate_performance_report()
            avg_buy_price = self._estimate_avg_buy_price()

            # Prepare comprehensive indicators data
            indicators_data = {
                'current_price': current_price,
                'news_analysis': news_analysis,
                'correlations': enhanced_indicators.get('correlations', {}),
                'liquidation_signals': enhanced_indicators.get('liquidation_signals', {}),
                'rsi': enhanced_indicators.get('rsi', rsi),
                'macd': enhanced_indicators.get('macd', macd),
                'signal': enhanced_indicators.get('signal', signal),
                'ma_short': enhanced_indicators.get('ma_short', ma_short),
                'ma_long': enhanced_indicators.get('ma_long', ma_long),
                'vwap': enhanced_indicators.get('vwap', vwap),
                'upper_band': upper_band,
                'lower_band': lower_band,
                'volatility': volatility,
                'sentiment': sentiment,
                'fee_rate': onchain_signals.get("fee_rate", 0),
                'netflow': onchain_signals.get("netflow", 0),
                'onchain_volume': onchain_signals.get("volume", 0),
                'old_utxos': onchain_signals.get("old_utxos", 0),
                'market_trend': market_trend,
                'performance_report': performance_report,
                'avg_buy_price': avg_buy_price
            }

            # Store current indicators for metrics
            self.last_rsi = indicators_data['rsi']
            self.last_sentiment = sentiment

            # Log current market state
            logger.info(f"💹 MARKET STATE: Price=€{current_price:.0f}, RSI={indicators_data['rsi']:.1f}, "
                       f"Sentiment={sentiment:.3f}, Risk-off={news_analysis.get('risk_off_probability', 0)*100:.0f}%, "
                       f"Trend={market_trend}")

            # Enhanced decision making
            action = self.enhanced_decide_action_with_risk_override(indicators_data)
            reason = f"{action.upper()} decision from enhanced risk system"

            # Calculate position size if trading
            trade_volume = 0
            if action in ['buy', 'sell']:
                trade_volume = self.calculate_risk_adjusted_position_size(action, indicators_data, btc_balance, eur_balance)
                
                if trade_volume <= self.min_trade_volume:
                    action = 'hold'
                    reason = f"Position size too small ({trade_volume:.8f} < {self.min_trade_volume:.8f})"
                    logger.info(reason)

            # Check trading limits
            if action != 'hold' and self.daily_trade_count >= self.max_daily_trades:
                action = 'hold'
                reason = f"Daily trade limit reached ({self.daily_trade_count}/{self.max_daily_trades})"
                logger.info(reason)

            # Execute trading decision
            execution_result = self._execute_trading_decision(action, trade_volume, reason, indicators_data)
            
            # Log strategy execution
            self._log_strategy_execution(action, trade_volume, reason, indicators_data, execution_result)

            logger.info(f"🎯 Strategy execution complete: {action.upper()} - {reason}")

        except Exception as e:
            logger.error(f"Strategy execution failed: {e}", exc_info=True)

    def _execute_trading_decision(self, action: str, trade_volume: float, reason: str, indicators_data: Dict) -> Dict:
        """Execute the trading decision with comprehensive error handling"""
        execution_result = {
            'executed': False,
            'order_id': None,
            'price': None,
            'reason': reason
        }

        if action == 'hold':
            execution_result['executed'] = True
            execution_result['reason'] = reason
            return execution_result

        try:
            # Check for pending orders of the same side
            if self.should_wait_for_pending_orders(action):
                execution_result['reason'] = f"Waiting for pending {action} orders"
                logger.info(execution_result['reason'])
                return execution_result

            # Get order book and optimal price
            order_book = self.trade_executor.get_btc_order_book()
            if not order_book:
                execution_result['reason'] = f"{action} failed: No order book available"
                logger.error(execution_result['reason'])
                return execution_result

            optimal_price = self.trade_executor.get_optimal_price(order_book, action)
            if not optimal_price:
                execution_result['reason'] = f"{action} failed: No optimal price found"
                logger.error(execution_result['reason'])
                return execution_result

            execution_result['price'] = optimal_price

            # Execute the trade
            if self.order_manager:
                # Use order manager for limit orders
                order_id = self.order_manager.place_limit_order_with_timeout(
                    volume=trade_volume,
                    side=action,
                    price=optimal_price,
                    timeout=300,
                    post_only=False
                )

                if order_id:
                    execution_result['executed'] = True
                    execution_result['order_id'] = order_id
                    execution_result['reason'] = f"{action} order placed successfully"
                    
                    self.last_trade_time = time.time()
                    
                    logger.info(f"✅ {action.upper()} order placed: {trade_volume:.8f} BTC at €{optimal_price:.2f}, ID: {order_id}")
                else:
                    execution_result['reason'] = f"{action} failed: Could not place order"
                    logger.error(execution_result['reason'])
            else:
                # Direct execution fallback
                if self.trade_executor.execute_trade(trade_volume, action, optimal_price):
                    execution_result['executed'] = True
                    execution_result['reason'] = f"{action} executed directly"
                    
                    self.last_trade_time = time.time()
                    self.daily_trade_count += 1
                    self._save_trade_session()
                    
                    # Update recent buys for tracking
                    if action == 'buy':
                        self.recent_buys.append((optimal_price, trade_volume))
                        self.recent_buys = self.recent_buys[-10:]
                        self._save_recent_buys()
                    
                    logger.info(f"✅ {action.upper()} executed directly: {trade_volume:.8f} BTC at €{optimal_price:.2f}")
                else:
                    execution_result['reason'] = f"{action} failed: Direct execution failed"
                    logger.error(execution_result['reason'])

        except Exception as e:
            execution_result['reason'] = f"{action} failed: {str(e)}"
            logger.error(f"Trade execution error: {e}", exc_info=True)

        return execution_result

    def _log_strategy_execution(self, action: str, trade_volume: float, reason: str, 
                              indicators_data: Dict, execution_result: Dict):
        """Log comprehensive strategy execution details"""
        try:
            current_price = indicators_data.get('current_price', 0)
            avg_buy_price = indicators_data.get('avg_buy_price', 0)
            profit_margin = None
            
            if avg_buy_price and avg_buy_price > 0:
                profit_margin = (current_price - avg_buy_price) / avg_buy_price

            # Get current balances
            btc_balance = self.trade_executor.get_total_btc_balance() or 0
            eur_balance = self.trade_executor.get_available_balance("EUR") or 0

            log_data = {
                "timestamp": datetime.now().isoformat(),
                "price": current_price,
                "trade_volume": trade_volume if execution_result['executed'] and action != 'hold' else 0,
                "side": action if execution_result['executed'] and action != 'hold' else "",
                "reason": execution_result['reason'],
                "dip": (indicators_data.get('ma_short', current_price) - current_price) / indicators_data.get('ma_short', current_price) if indicators_data.get('ma_short') else 0,
                "rsi": indicators_data.get('rsi', 50),
                "macd": indicators_data.get('macd', 0),
                "signal": indicators_data.get('signal', 0),
                "ma_short": indicators_data.get('ma_short', current_price),
                "ma_long": indicators_data.get('ma_long', current_price),
                "upper_band": indicators_data.get('upper_band', current_price),
                "lower_band": indicators_data.get('lower_band', current_price),
                "sentiment": indicators_data.get('sentiment', 0),
                "fee_rate": indicators_data.get('fee_rate', 0),
                "netflow": indicators_data.get('netflow', 0),
                "volume": indicators_data.get('onchain_volume', 0),
                "old_utxos": indicators_data.get('old_utxos', 0),
                "buy_decision": action == 'buy' and execution_result['executed'],
                "sell_decision": action == 'sell' and execution_result['executed'],
                "btc_balance": btc_balance,
                "eur_balance": eur_balance,
                "avg_buy_price": avg_buy_price,
                "profit_margin": profit_margin
            }

            self.data_manager.log_strategy(**log_data)
            
            # Log risk decision
            self.log_risk_decision(action, indicators_data, execution_result['reason'])

        except Exception as e:
            logger.error(f"Failed to log strategy execution: {e}")

    def get_trading_status_summary(self) -> Dict:
        """Get comprehensive trading status for monitoring"""
        try:
            # Get current balances
            btc_balance = self.trade_executor.get_total_btc_balance() or 0
            eur_balance = self.trade_executor.get_available_balance("EUR") or 0
            current_price, _ = self.trade_executor.fetch_current_price()
            
            # Calculate portfolio value
            total_value = eur_balance + (btc_balance * (current_price or 0))
            
            # Get order statistics
            order_stats = self.order_manager.get_order_statistics() if self.order_manager else {}
            
            # Get performance report
            performance = self.performance_tracker.generate_performance_report()
            
            # Get recent trades
            recent_trades = self.order_manager.get_recent_trade_count(24) if self.order_manager else 0
            
            status = {
                'timestamp': datetime.now().isoformat(),
                'balances': {
                    'btc': btc_balance,
                    'eur': eur_balance,
                    'total_value_eur': total_value,
                    'current_btc_price': current_price or 0
                },
                'trading': {
                    'daily_trades': self.daily_trade_count,
                    'recent_trades_24h': recent_trades,
                    'max_daily_trades': self.max_daily_trades,
                    'last_trade_time': self.last_trade_time
                },
                'orders': {
                    'pending_count': len(self.order_manager.get_pending_orders()) if self.order_manager else 0,
                    'fill_rate': order_stats.get('fill_rate', 0),
                    'total_fees': order_stats.get('total_fees_paid', 0)
                },
                'performance': performance,
                'indicators': {
                    'last_rsi': getattr(self, 'last_rsi', 0),
                    'last_sentiment': getattr(self, 'last_sentiment', 0)
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to generate status summary: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    def print_trading_status(self):
        """Print a comprehensive trading status report"""
        try:
            status = self.get_trading_status_summary()
            
            print("\n" + "="*60)
            print("🤖 BITCOIN TRADING BOT STATUS")
            print("="*60)
            
            if 'error' in status:
                print(f"❌ Error: {status['error']}")
                return
            
            # Balances
            balances = status['balances']
            print(f"💰 Portfolio:")
            print(f"  BTC: {balances['btc']:.8f}")
            print(f"  EUR: €{balances['eur']:.2f}")
            print(f"  Total Value: €{balances['total_value_eur']:.2f}")
            print(f"  BTC Price: €{balances['current_btc_price']:.2f}")
            
            # Trading activity
            trading = status['trading']
            print(f"\n📊 Trading Activity:")
            print(f"  Today's Trades: {trading['daily_trades']}/{trading['max_daily_trades']}")
            print(f"  24h Trades: {trading['recent_trades_24h']}")
            if trading['last_trade_time'] > 0:
                last_trade = datetime.fromtimestamp(trading['last_trade_time'])
                print(f"  Last Trade: {last_trade.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Orders
            orders = status['orders']
            print(f"\n📋 Orders:")
            print(f"  Pending: {orders['pending_count']}")
            print(f"  Fill Rate: {orders['fill_rate']:.1%}")
            print(f"  Total Fees: €{orders['total_fees']:.4f}")
            
            # Performance
            perf = status['performance']
            if 'returns' in perf:
                print(f"\n📈 Performance:")
                print(f"  24h Return: {perf['returns']['24h']}")
                print(f"  7d Return: {perf['returns']['7d']}")
                print(f"  Total Return: {perf['returns']['total']}")
                
                if 'risk_metrics' in perf:
                    risk = perf['risk_metrics']
                    print(f"  Win Rate: {risk['win_rate']}")
                    print(f"  Sharpe Ratio: {risk['sharpe_ratio']}")
                    print(f"  Max Drawdown: {risk['max_drawdown']}")
            
            # Technical indicators
            indicators = status['indicators']
            print(f"\n📊 Last Indicators:")
            print(f"  RSI: {indicators['last_rsi']:.1f}")
            print(f"  Sentiment: {indicators['last_sentiment']:.3f}")
            
            print("="*60 + "\n")
            
        except Exception as e:
            logger.error(f"Failed to print trading status: {e}")
            print(f"❌ Failed to generate status report: {e}")

# Usage improvements for main.py