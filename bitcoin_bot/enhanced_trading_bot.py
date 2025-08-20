import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import os
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import time


@dataclass
class TradeSignal:
    """Enhanced trade signal with confidence scores"""

    action: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0-1
    urgency: float  # 0-1 (how quickly to act)
    position_size: float  # 0-1 (fraction of available balance)
    reasoning: List[str]
    risk_level: str  # 'low', 'medium', 'high'
    expected_duration: int  # expected hold time in minutes


class AdaptiveLearningEngine:
    """Machine learning engine that learns from past trades"""

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            "rsi",
            "macd",
            "signal",
            "bb_position",
            "volume_ratio",
            "price_momentum",
            "volatility",
            "sentiment",
            "risk_off_prob",
            "hour_of_day",
            "day_of_week",
            "recent_performance",
        ]
        self.model_file = "trading_model.pkl"
        self.performance_window = 50  # Last N trades to consider
        self._load_model()

    def _load_model(self):
        """Load previously trained model"""
        try:
            if os.path.exists(self.model_file):
                with open(self.model_file, "rb") as f:
                    data = pickle.load(f)
                    self.model = data["model"]
                    self.scaler = data["scaler"]
                    self.is_trained = data["is_trained"]
                print("✅ Loaded existing ML model")
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")

    def _save_model(self):
        """Save trained model"""
        try:
            data = {
                "model": self.model,
                "scaler": self.scaler,
                "is_trained": self.is_trained,
            }
            with open(self.model_file, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"❌ Could not save model: {e}")

    def extract_features(
        self, indicators_data: Dict, current_time: datetime = None
    ) -> np.array:
        """Extract features for ML model"""
        if current_time is None:
            current_time = datetime.now()

        # Technical indicators
        rsi = indicators_data.get("rsi", 50) / 100.0  # Normalize to 0-1
        macd = np.tanh(indicators_data.get("macd", 0) / 1000)  # Bounded between -1,1
        signal = np.tanh(indicators_data.get("signal", 0) / 1000)

        # Bollinger Band position
        current_price = indicators_data.get("current_price", 0)
        upper_band = indicators_data.get("upper_band", current_price)
        lower_band = indicators_data.get("lower_band", current_price)
        bb_position = (
            ((current_price - lower_band) / (upper_band - lower_band))
            if upper_band > lower_band
            else 0.5
        )

        # Volume and momentum
        volume_ratio = (
            min(2.0, indicators_data.get("volume_ratio", 1.0)) / 2.0
        )  # Normalize
        price_momentum = np.tanh(
            indicators_data.get("price_momentum", 0) / 0.1
        )  # Bounded
        volatility = min(
            1.0, indicators_data.get("volatility", 0.02) / 0.1
        )  # Normalize

        # Sentiment and news
        sentiment = (indicators_data.get("sentiment", 0) + 1) / 2  # Convert -1,1 to 0,1
        risk_off_prob = indicators_data.get("news_analysis", {}).get(
            "risk_off_probability", 0
        )

        # Time features
        hour_of_day = current_time.hour / 24.0
        day_of_week = current_time.weekday() / 7.0

        # Recent performance
        recent_performance = (
            indicators_data.get("recent_win_rate", 0.5) - 0.5
        ) * 2  # Convert to -1,1

        features = np.array(
            [
                rsi,
                macd,
                signal,
                bb_position,
                volume_ratio,
                price_momentum,
                volatility,
                sentiment,
                risk_off_prob,
                hour_of_day,
                day_of_week,
                recent_performance,
            ]
        )

        return features

    def train_on_historical_data(self, bot_logs_df: pd.DataFrame):
        """Train model on historical trading data"""
        if len(bot_logs_df) < 50:
            print("⚠️ Insufficient historical data for training")
            return

        # Prepare training data
        X = []
        y = []

        for i in range(len(bot_logs_df) - 1):
            row = bot_logs_df.iloc[i]
            next_row = bot_logs_df.iloc[i + 1]

            # Extract features at time of decision
            features = self._extract_features_from_log(row)
            if features is None:
                continue

            # Determine if the trade was profitable (label)
            current_price = row.get("price", 0)
            next_price = next_row.get("price", current_price)

            if row.get("buy_decision") == "True":
                # For buy decisions, profitable if price went up
                profitable = (
                    1 if next_price > current_price * 1.002 else 0
                )  # 0.2% threshold
            elif row.get("sell_decision") == "True":
                # For sell decisions, profitable if price went down
                profitable = 1 if next_price < current_price * 0.998 else 0
            else:
                continue  # Skip hold decisions for now

            X.append(features)
            y.append(profitable)

        if len(X) < 20:
            print("⚠️ Insufficient valid training examples")
            return

        X = np.array(X)
        y = np.array(y)

        # Scale features and train model
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        self._save_model()

        # Print training results
        score = self.model.score(X_scaled, y)
        print(f"✅ Model trained on {len(X)} examples, accuracy: {score:.2%}")

    def _extract_features_from_log(self, row) -> Optional[np.array]:
        """Extract features from a log row"""
        try:
            indicators_data = {
                "rsi": row.get("rsi", 50),
                "macd": row.get("macd", 0),
                "signal": row.get("signal", 0),
                "current_price": row.get("price", 0),
                "upper_band": row.get("upper_band", row.get("price", 0)),
                "lower_band": row.get("lower_band", row.get("price", 0)),
                "volume_ratio": 1.0,  # Not available in logs
                "price_momentum": 0.0,  # Would need to calculate
                "volatility": 0.02,  # Default
                "sentiment": row.get("sentiment", 0),
                "news_analysis": {"risk_off_probability": 0.0},
                "recent_win_rate": 0.5,
            }

            timestamp = pd.to_datetime(row.get("timestamp"))
            return self.extract_features(indicators_data, timestamp)
        except Exception:
            return None

    def predict_trade_success(self, indicators_data: Dict) -> Tuple[float, float]:
        """Predict probability of trade success for buy and sell"""
        if not self.is_trained:
            return 0.5, 0.5  # Neutral if not trained

        features = self.extract_features(indicators_data).reshape(1, -1)
        features_scaled = self.scaler.transform(features)

        # Get probability predictions
        proba = self.model.predict_proba(features_scaled)[0]

        # Return probabilities for [unsuccessful, successful]
        return proba[0], proba[1]


class MarketRegimeDetector:
    """Detect current market regime for adaptive strategy"""

    def __init__(self, lookback_periods: int = 200):
        self.lookback_periods = lookback_periods

    def detect_regime(
        self, prices: List[float], volumes: List[float]
    ) -> Dict[str, any]:
        """Detect current market regime"""
        if len(prices) < 50:
            return {
                "regime": "unknown",
                "confidence": 0.0,
                "volatility_regime": "normal",
            }

        recent_prices = np.array(prices[-self.lookback_periods :])
        recent_volumes = np.array(volumes[-self.lookback_periods :])

        # Calculate various metrics
        returns = np.diff(recent_prices) / recent_prices[:-1]
        volatility = np.std(returns) * np.sqrt(96)  # Annualized (15min intervals)

        # Trend detection
        short_ma = np.mean(recent_prices[-20:])
        long_ma = np.mean(recent_prices[-50:]) if len(recent_prices) >= 50 else short_ma
        trend_strength = (short_ma - long_ma) / long_ma if long_ma > 0 else 0

        # Volume profile
        avg_volume = np.mean(recent_volumes)
        recent_volume = np.mean(recent_volumes[-10:])
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0

        # Determine regime
        regime = "ranging"
        confidence = 0.5

        if abs(trend_strength) > 0.05:  # 5% trend
            if trend_strength > 0:
                regime = "bull_trend"
                confidence = min(0.9, abs(trend_strength) * 10)
            else:
                regime = "bear_trend"
                confidence = min(0.9, abs(trend_strength) * 10)

        # Volatility regime
        vol_regime = "normal"
        if volatility > 0.8:  # 80% annualized
            vol_regime = "high"
        elif volatility < 0.3:  # 30% annualized
            vol_regime = "low"

        # Market structure
        structure = self._analyze_market_structure(recent_prices)

        return {
            "regime": regime,
            "confidence": confidence,
            "volatility_regime": vol_regime,
            "trend_strength": trend_strength,
            "volume_ratio": volume_ratio,
            "volatility": volatility,
            "structure": structure,
        }

    def _analyze_market_structure(self, prices: np.array) -> Dict[str, any]:
        """Analyze market microstructure"""
        if len(prices) < 20:
            return {
                "support_levels": [],
                "resistance_levels": [],
                "breakout_potential": 0.0,
            }

        # Simple support/resistance detection
        highs = []
        lows = []

        for i in range(2, len(prices) - 2):
            # Local high
            if (
                prices[i] > prices[i - 1]
                and prices[i] > prices[i - 2]
                and prices[i] > prices[i + 1]
                and prices[i] > prices[i + 2]
            ):
                highs.append(prices[i])

            # Local low
            if (
                prices[i] < prices[i - 1]
                and prices[i] < prices[i - 2]
                and prices[i] < prices[i + 1]
                and prices[i] < prices[i + 2]
            ):
                lows.append(prices[i])

        current_price = prices[-1]

        # Find nearest support/resistance
        support_levels = [low for low in lows if low < current_price]
        resistance_levels = [high for high in highs if high > current_price]

        # Breakout potential
        breakout_potential = 0.0
        if resistance_levels:
            nearest_resistance = min(resistance_levels)
            distance_to_resistance = (
                nearest_resistance - current_price
            ) / current_price
            if distance_to_resistance < 0.02:  # Within 2%
                breakout_potential = 0.8

        return {
            "support_levels": sorted(support_levels)[-3:],  # Top 3 nearest
            "resistance_levels": sorted(resistance_levels)[:3],  # Top 3 nearest
            "breakout_potential": breakout_potential,
        }


class AdaptivePositionSizer:
    """Dynamically adjust position sizes based on performance and market conditions"""

    def __init__(self, base_position_pct: float = 0.10):
        self.base_position_pct = base_position_pct
        self.performance_history = []
        self.max_history = 100

    def calculate_position_size(
        self,
        signal: TradeSignal,
        balance: float,
        market_regime: Dict,
        recent_performance: Dict,
    ) -> float:
        """Calculate optimal position size"""
        base_size = balance * self.base_position_pct

        # Confidence adjustment
        confidence_multiplier = 0.5 + (signal.confidence * 1.5)  # 0.5 to 2.0x

        # Urgency adjustment
        urgency_multiplier = 0.8 + (signal.urgency * 0.4)  # 0.8 to 1.2x

        # Performance adjustment
        win_rate = recent_performance.get("win_rate", 0.5)
        performance_multiplier = 0.5 + win_rate  # 0.5 to 1.5x

        # Market regime adjustment
        regime = market_regime.get("regime", "ranging")
        volatility = market_regime.get("volatility", 0.5)

        regime_multiplier = 1.0
        if regime == "bull_trend" and signal.action == "buy":
            regime_multiplier = 1.3
        elif regime == "bear_trend" and signal.action == "sell":
            regime_multiplier = 1.3
        elif volatility > 0.8:  # High volatility
            regime_multiplier = 0.7

        # Risk adjustment
        risk_multiplier = 1.0
        if signal.risk_level == "high":
            risk_multiplier = 0.6
        elif signal.risk_level == "low":
            risk_multiplier = 1.2

        # Calculate final size
        final_size = (
            base_size
            * confidence_multiplier
            * urgency_multiplier
            * performance_multiplier
            * regime_multiplier
            * risk_multiplier
        )

        # Apply bounds
        max_size = balance * 0.25  # Never more than 25% of balance
        min_size = balance * 0.01  # At least 1% if trading

        final_size = max(min_size, min(max_size, final_size))

        return final_size


class EnhancedTradingBot:
    """Enhanced trading bot with adaptive learning capabilities"""

    def __init__(self, original_bot):
        self.original_bot = original_bot
        self.learning_engine = AdaptiveLearningEngine()
        self.regime_detector = MarketRegimeDetector()
        self.position_sizer = AdaptivePositionSizer()

        # Performance tracking
        self.trade_history = []
        self.performance_metrics = {}

        # Initialize learning from historical data
        self._initialize_learning()

    def _initialize_learning(self):
        """Initialize learning from historical bot logs"""
        try:
            if os.path.exists(self.original_bot.data_manager.bot_logs_file):
                df = pd.read_csv(self.original_bot.data_manager.bot_logs_file)
                if len(df) > 20:
                    print(f"🧠 Training on {len(df)} historical trades...")
                    self.learning_engine.train_on_historical_data(df)
                else:
                    print("⚠️ Insufficient historical data for initial training")
            else:
                print("⚠️ No historical bot logs found")
        except Exception as e:
            print(f"❌ Error initializing learning: {e}")

    def generate_enhanced_signal(self, indicators_data: Dict) -> TradeSignal:
        """Generate enhanced trading signal with ML insights"""

        # Get basic signal from original bot logic
        basic_action = self.original_bot.enhanced_decide_action_with_risk_override(
            indicators_data
        )

        # Get ML prediction
        ml_fail_prob, ml_success_prob = self.learning_engine.predict_trade_success(
            indicators_data
        )

        # Get market regime
        prices, volumes = self.original_bot.data_manager.load_price_history()
        market_regime = self.regime_detector.detect_regime(prices, volumes)

        # Calculate confidence and urgency
        confidence = self._calculate_confidence(
            indicators_data, ml_success_prob, market_regime
        )
        urgency = self._calculate_urgency(indicators_data, market_regime)
        risk_level = self._assess_risk_level(indicators_data, market_regime)

        # Override basic action if ML strongly disagrees
        if ml_success_prob < 0.3 and basic_action in ["buy", "sell"]:
            action = "hold"
            reasoning = [f"ML model predicts low success rate ({ml_success_prob:.2%})"]
        elif ml_success_prob > 0.8 and basic_action == "hold":
            # If ML is very confident and we detect strong signals
            if self._detect_strong_signals(indicators_data, market_regime):
                action = self._suggest_ml_action(indicators_data, market_regime)
                reasoning = [
                    f"ML model predicts high success rate ({ml_success_prob:.2%})",
                    "Strong technical signals detected",
                ]
            else:
                action = basic_action
                reasoning = ["Following conservative approach despite ML confidence"]
        else:
            action = basic_action
            reasoning = ["Following enhanced risk system decision"]

        # Add detailed reasoning
        reasoning.extend(
            self._generate_reasoning(indicators_data, market_regime, ml_success_prob)
        )

        # Calculate expected duration
        expected_duration = self._estimate_trade_duration(
            indicators_data, market_regime
        )

        return TradeSignal(
            action=action,
            confidence=confidence,
            urgency=urgency,
            position_size=0.0,  # Will be calculated by position sizer
            reasoning=reasoning,
            risk_level=risk_level,
            expected_duration=expected_duration,
        )

    def _calculate_confidence(
        self, indicators_data: Dict, ml_success_prob: float, market_regime: Dict
    ) -> float:
        """Calculate confidence score for the trade signal"""
        confidence_factors = []

        # ML confidence
        confidence_factors.append(ml_success_prob)

        # Market regime confidence
        regime_confidence = market_regime.get("confidence", 0.5)
        confidence_factors.append(regime_confidence)

        # Technical indicator alignment
        technical_alignment = self._calculate_technical_alignment(indicators_data)
        confidence_factors.append(technical_alignment)

        # News sentiment clarity
        news_analysis = indicators_data.get("news_analysis", {})
        sentiment = abs(
            indicators_data.get("sentiment", 0)
        )  # Strong sentiment either way
        confidence_factors.append(min(1.0, sentiment * 2))

        # Volume confirmation
        volume_ratio = market_regime.get("volume_ratio", 1.0)
        volume_confidence = min(1.0, volume_ratio / 2.0) if volume_ratio > 1 else 0.5
        confidence_factors.append(volume_confidence)

        # Average with weights
        weights = [0.3, 0.2, 0.2, 0.15, 0.15]  # ML gets highest weight
        confidence = sum(f * w for f, w in zip(confidence_factors, weights))

        return max(0.0, min(1.0, confidence))

    def _calculate_urgency(self, indicators_data: Dict, market_regime: Dict) -> float:
        """Calculate urgency score (how quickly to act)"""
        urgency_factors = []

        # Volatility - higher volatility = higher urgency
        volatility = market_regime.get("volatility", 0.5)
        urgency_factors.append(min(1.0, volatility * 2))

        # Breakout potential
        breakout_potential = market_regime.get("structure", {}).get(
            "breakout_potential", 0.0
        )
        urgency_factors.append(breakout_potential)

        # RSI extremes
        rsi = indicators_data.get("rsi", 50)
        if rsi < 25 or rsi > 75:
            rsi_urgency = 0.8
        elif rsi < 35 or rsi > 65:
            rsi_urgency = 0.6
        else:
            rsi_urgency = 0.3
        urgency_factors.append(rsi_urgency)

        # News risk-off probability
        risk_off_prob = indicators_data.get("news_analysis", {}).get(
            "risk_off_probability", 0
        )
        if risk_off_prob > 0.7:
            urgency_factors.append(0.9)  # High urgency to act on risk-off
        else:
            urgency_factors.append(0.4)

        return max(0.1, min(1.0, np.mean(urgency_factors)))

    def _assess_risk_level(self, indicators_data: Dict, market_regime: Dict) -> str:
        """Assess overall risk level"""
        risk_score = 0

        # Market volatility
        volatility = market_regime.get("volatility", 0.5)
        if volatility > 0.8:
            risk_score += 2
        elif volatility > 0.6:
            risk_score += 1

        # News risk-off probability
        risk_off_prob = indicators_data.get("news_analysis", {}).get(
            "risk_off_probability", 0
        )
        if risk_off_prob > 0.6:
            risk_score += 2
        elif risk_off_prob > 0.4:
            risk_score += 1

        # Current position P&L
        avg_buy_price = indicators_data.get("avg_buy_price", 0)
        current_price = indicators_data.get("current_price", 0)
        if avg_buy_price and current_price:
            pnl = (current_price - avg_buy_price) / avg_buy_price
            if pnl < -0.05:  # 5% underwater
                risk_score += 1

        # Market structure
        regime = market_regime.get("regime", "ranging")
        if regime == "bear_trend":
            risk_score += 1

        if risk_score >= 4:
            return "high"
        elif risk_score >= 2:
            return "medium"
        else:
            return "low"

    def _detect_strong_signals(
        self, indicators_data: Dict, market_regime: Dict
    ) -> bool:
        """Detect if we have strong confluence of signals"""
        signals = 0

        # RSI extremes
        rsi = indicators_data.get("rsi", 50)
        if rsi < 30 or rsi > 70:
            signals += 1

        # Strong trend
        trend_strength = abs(market_regime.get("trend_strength", 0))
        if trend_strength > 0.05:
            signals += 1

        # Volume confirmation
        volume_ratio = market_regime.get("volume_ratio", 1.0)
        if volume_ratio > 1.5:
            signals += 1

        # Clear sentiment
        sentiment = abs(indicators_data.get("sentiment", 0))
        if sentiment > 0.1:
            signals += 1

        # Breakout potential
        breakout_potential = market_regime.get("structure", {}).get(
            "breakout_potential", 0.0
        )
        if breakout_potential > 0.6:
            signals += 1

        return signals >= 3

    def _suggest_ml_action(self, indicators_data: Dict, market_regime: Dict) -> str:
        """Suggest action when ML is confident but basic system says hold"""
        regime = market_regime.get("regime", "ranging")
        trend_strength = market_regime.get("trend_strength", 0)
        rsi = indicators_data.get("rsi", 50)

        # In bull trend with oversold conditions
        if regime == "bull_trend" and rsi < 35:
            return "buy"

        # In bear trend with overbought conditions
        if regime == "bear_trend" and rsi > 65:
            return "sell"

        # Strong upward momentum
        if trend_strength > 0.03 and rsi < 60:
            return "buy"

        # Strong downward momentum
        if trend_strength < -0.03 and rsi > 40:
            return "sell"

        return "hold"

    def _calculate_technical_alignment(self, indicators_data: Dict) -> float:
        """Calculate how well technical indicators align"""
        buy_signals = 0
        sell_signals = 0
        total_signals = 0

        # RSI
        rsi = indicators_data.get("rsi", 50)
        if rsi < 40:
            buy_signals += 1
        elif rsi > 60:
            sell_signals += 1
        total_signals += 1

        # MACD
        macd = indicators_data.get("macd", 0)
        signal = indicators_data.get("signal", 0)
        if macd > signal:
            buy_signals += 1
        else:
            sell_signals += 1
        total_signals += 1

        # Price vs VWAP
        current_price = indicators_data.get("current_price", 0)
        vwap = indicators_data.get("vwap", current_price)
        if current_price < vwap:
            buy_signals += 1
        else:
            sell_signals += 1
        total_signals += 1

        # Calculate alignment (how strongly indicators agree)
        max_signals = max(buy_signals, sell_signals)
        alignment = max_signals / total_signals if total_signals > 0 else 0.5

        return alignment

    def _generate_reasoning(
        self, indicators_data: Dict, market_regime: Dict, ml_success_prob: float
    ) -> List[str]:
        """Generate detailed reasoning for the trade decision"""
        reasoning = []

        # Market regime
        regime = market_regime.get("regime", "ranging")
        regime_confidence = market_regime.get("confidence", 0.5)
        reasoning.append(
            f"Market regime: {regime} (confidence: {regime_confidence:.1%})"
        )

        # Volatility
        volatility = market_regime.get("volatility", 0.5)
        vol_regime = market_regime.get("volatility_regime", "normal")
        reasoning.append(f"Volatility: {vol_regime} ({volatility:.1%} annualized)")

        # ML prediction
        reasoning.append(f"ML success probability: {ml_success_prob:.1%}")

        # Key technical levels
        rsi = indicators_data.get("rsi", 50)
        reasoning.append(f"RSI: {rsi:.1f}")

        # Risk factors
        risk_off_prob = indicators_data.get("news_analysis", {}).get(
            "risk_off_probability", 0
        )
        if risk_off_prob > 0.3:
            reasoning.append(f"Elevated macro risk: {risk_off_prob:.1%}")

        return reasoning

    def _estimate_trade_duration(
        self, indicators_data: Dict, market_regime: Dict
    ) -> int:
        """Estimate expected trade duration in minutes"""
        base_duration = 180  # 3 hours default

        # Adjust for volatility
        volatility = market_regime.get("volatility", 0.5)
        if volatility > 0.8:
            duration = base_duration * 0.5  # Shorter in high vol
        elif volatility < 0.3:
            duration = base_duration * 2.0  # Longer in low vol
        else:
            duration = base_duration

        # Adjust for trend strength
        trend_strength = abs(market_regime.get("trend_strength", 0))
        if trend_strength > 0.05:
            duration *= 1.5  # Hold longer in strong trends

        # Adjust for urgency factors
        urgency = self._calculate_urgency(indicators_data, market_regime)
        if urgency > 0.8:
            duration *= 0.7  # Shorter duration for urgent trades

        return int(max(30, min(720, duration)))  # Between 30 minutes and 12 hours

    def execute_enhanced_strategy(self):
        """Execute the enhanced strategy with adaptive learning"""
        try:
            # Get all the data we need
            prices, volumes = self.original_bot.data_manager.load_price_history()
            current_price, current_volume = (
                self.original_bot.trade_executor.fetch_current_price()
            )

            if not current_price or len(prices) < 50:
                print("⚠️ Insufficient data for enhanced strategy")
                return

            # Update price history
            self.original_bot.price_history.append(current_price)
            self.original_bot.price_history = self.original_bot.price_history[
                -400:
            ]  # Keep last 400 prices

            # Get fresh OHLC data
            ohlc = self.original_bot.trade_executor.get_ohlc_data(
                pair="BTC/EUR", interval="15m", since=int(time.time() - 7200)
            )
            if ohlc:
                self.original_bot.data_manager.append_ohlc_data(ohlc)

            # Calculate all indicators (reusing original bot's logic)
            from core.indicators import (
                calculate_rsi,
                calculate_macd,
                calculate_bollinger_bands,
                calculate_moving_average,
                calculate_vwap,
                fetch_enhanced_news,
                calculate_enhanced_sentiment,
                calculate_risk_adjusted_indicators,
            )

            # Technical indicators
            rsi = calculate_rsi(prices) or 50
            macd, signal = calculate_macd(prices) or (0, 0)
            upper_band, ma_short, lower_band = calculate_bollinger_bands(prices) or (
                current_price,
                current_price,
                current_price,
            )
            ma_long = calculate_moving_average(prices, 50) or current_price
            vwap = calculate_vwap(prices, volumes) or current_price

            # Enhanced news analysis
            articles = fetch_enhanced_news(top_n=20)
            news_analysis = calculate_enhanced_sentiment(articles)
            sentiment = news_analysis.get("sentiment", 0)

            # Enhanced indicators with risk adjustment
            try:
                enhanced_indicators = calculate_risk_adjusted_indicators(
                    prices, volumes, news_analysis
                )
            except Exception as e:
                print(f"⚠️ Enhanced indicators failed: {e}")
                enhanced_indicators = {
                    "rsi": rsi,
                    "macd": macd,
                    "signal": signal,
                    "ma_short": ma_short,
                    "ma_long": ma_long,
                    "vwap": vwap,
                    "correlations": {},
                    "liquidation_signals": {},
                    "risk_factor": 1.0,
                }

            # Get on-chain signals
            onchain_signals = self.original_bot.onchain_analyzer.get_onchain_signals()

            # Get balances
            btc_balance = self.original_bot.trade_executor.get_total_btc_balance() or 0
            eur_balance = (
                self.original_bot.trade_executor.get_available_balance("EUR") or 0
            )

            # Update performance tracking
            self.original_bot.performance_tracker.update_equity(
                btc_balance, eur_balance, current_price
            )
            performance_report = (
                self.original_bot.performance_tracker.generate_performance_report()
            )
            avg_buy_price = self.original_bot._estimate_avg_buy_price()

            # Calculate recent performance metrics
            recent_performance = self._calculate_recent_performance()

            # Prepare indicators data with additional ML features
            indicators_data = {
                "current_price": current_price,
                "news_analysis": news_analysis,
                "rsi": enhanced_indicators.get("rsi", rsi),
                "macd": enhanced_indicators.get("macd", macd),
                "signal": enhanced_indicators.get("signal", signal),
                "ma_short": enhanced_indicators.get("ma_short", ma_short),
                "ma_long": enhanced_indicators.get("ma_long", ma_long),
                "vwap": enhanced_indicators.get("vwap", vwap),
                "upper_band": upper_band,
                "lower_band": lower_band,
                "sentiment": sentiment,
                "volatility": self.original_bot._calculate_volatility(prices),
                "avg_buy_price": avg_buy_price,
                "performance_report": performance_report,
                "recent_win_rate": recent_performance.get("win_rate", 0.5),
                "volume_ratio": (
                    current_volume / np.mean(volumes[-20:])
                    if len(volumes) >= 20
                    else 1.0
                ),
                "price_momentum": (
                    (current_price - prices[-10]) / prices[-10]
                    if len(prices) >= 10
                    else 0
                ),
                **onchain_signals,
            }

            # Generate enhanced signal
            signal = self.generate_enhanced_signal(indicators_data)

            # Get market regime
            market_regime = self.regime_detector.detect_regime(prices, volumes)

            # Calculate optimal position size
            if signal.action in ["buy", "sell"]:
                balance = eur_balance if signal.action == "buy" else btc_balance
                position_size = self.position_sizer.calculate_position_size(
                    signal, balance, market_regime, recent_performance
                )

                # Convert to BTC amount for buy orders
                if signal.action == "buy":
                    position_btc = min(
                        position_size / current_price, eur_balance * 0.9 / current_price
                    )
                else:
                    position_btc = min(position_size, btc_balance * 0.8)

                # Apply minimum trade size
                min_trade_size = 0.0001  # 0.0001 BTC minimum
                if position_btc < min_trade_size:
                    signal.action = "hold"
                    signal.reasoning.append(
                        f"Position size too small: {position_btc:.8f} BTC"
                    )
            else:
                position_btc = 0

            # Enhanced logging
            self._log_enhanced_decision(
                signal, market_regime, indicators_data, position_btc
            )

            # Execute the trade with enhanced logic
            if signal.action in ["buy", "sell"] and position_btc > 0:
                success = self._execute_enhanced_trade(
                    signal, position_btc, current_price
                )
                if success:
                    # Update trade history for learning
                    trade_record = {
                        "timestamp": datetime.now(),
                        "action": signal.action,
                        "price": current_price,
                        "volume": position_btc,
                        "confidence": signal.confidence,
                        "risk_level": signal.risk_level,
                        "market_regime": market_regime["regime"],
                        "expected_duration": signal.expected_duration,
                        "indicators": indicators_data.copy(),
                    }
                    self.trade_history.append(trade_record)

                    # Retrain model periodically
                    if len(self.trade_history) % 20 == 0:
                        self._retrain_model()

            # Always check pending orders
            self.original_bot.check_pending_orders()

        except Exception as e:
            print(f"❌ Enhanced strategy execution failed: {e}")
            import traceback

            traceback.print_exc()

    def _calculate_recent_performance(self) -> Dict[str, float]:
        """Calculate recent performance metrics for adaptive sizing"""
        try:
            if not os.path.exists(self.original_bot.data_manager.bot_logs_file):
                return {"win_rate": 0.5, "avg_return": 0.0, "sharpe": 0.0}

            df = pd.read_csv(self.original_bot.data_manager.bot_logs_file)
            if len(df) < 10:
                return {"win_rate": 0.5, "avg_return": 0.0, "sharpe": 0.0}

            # Get recent trades (last 50)
            recent_df = df.tail(50)

            # Calculate win rate from actual trades
            trades = recent_df[
                (recent_df["side"].isin(["buy", "sell"]))
                & (recent_df["trade_volume"] > 0)
            ]

            if len(trades) < 5:
                return {"win_rate": 0.5, "avg_return": 0.0, "sharpe": 0.0}

            # Simple win rate calculation (this could be improved)
            # For now, assume profitable if price moved in favor direction
            wins = 0
            total_return = 0
            returns = []

            for i in range(len(trades) - 1):
                current_trade = trades.iloc[i]
                next_price = (
                    trades.iloc[i + 1]["price"]
                    if i < len(trades) - 1
                    else current_trade["price"]
                )

                if current_trade["side"] == "buy":
                    trade_return = (
                        next_price - current_trade["price"]
                    ) / current_trade["price"]
                elif current_trade["side"] == "sell":
                    trade_return = (
                        current_trade["price"] - next_price
                    ) / current_trade["price"]
                else:
                    continue

                if trade_return > 0.001:  # 0.1% threshold
                    wins += 1

                returns.append(trade_return)
                total_return += trade_return

            win_rate = wins / len(returns) if returns else 0.5
            avg_return = total_return / len(returns) if returns else 0.0
            sharpe = (
                np.mean(returns) / np.std(returns)
                if len(returns) > 1 and np.std(returns) > 0
                else 0.0
            )

            return {
                "win_rate": win_rate,
                "avg_return": avg_return,
                "sharpe": sharpe,
                "total_trades": len(returns),
            }

        except Exception as e:
            print(f"⚠️ Error calculating recent performance: {e}")
            return {"win_rate": 0.5, "avg_return": 0.0, "sharpe": 0.0}

    def _execute_enhanced_trade(
        self, signal: TradeSignal, volume: float, current_price: float
    ) -> bool:
        """Execute trade with enhanced order management"""
        try:
            # Check if we should wait for pending orders
            if self.original_bot.should_wait_for_pending_orders(signal.action):
                print(f"⏳ Waiting for pending {signal.action} orders")
                return False

            # Get optimal price
            order_book = self.original_bot.trade_executor.get_btc_order_book()
            if not order_book:
                print("❌ Could not get order book")
                return False

            optimal_price = self.original_bot.trade_executor.get_optimal_price(
                order_book, signal.action
            )
            if not optimal_price:
                print("❌ Could not determine optimal price")
                return False

            # Adjust timeout based on urgency and expected duration
            base_timeout = 300  # 5 minutes
            if signal.urgency > 0.8:
                timeout = int(base_timeout * 0.6)  # Shorter timeout for urgent trades
            elif signal.urgency < 0.3:
                timeout = int(base_timeout * 1.5)  # Longer timeout for patient trades
            else:
                timeout = base_timeout

            # Place order with enhanced parameters
            if self.original_bot.order_manager:
                order_id = (
                    self.original_bot.order_manager.place_limit_order_with_timeout(
                        volume=volume,
                        side=signal.action,
                        price=optimal_price,
                        timeout=timeout,
                        post_only=(
                            signal.urgency < 0.5
                        ),  # Use post-only for non-urgent trades
                    )
                )

                if order_id:
                    self.original_bot.last_trade_time = time.time()
                    print(
                        f"✅ {signal.action.upper()} order placed: {volume:.8f} BTC at €{optimal_price:.2f}"
                    )
                    print(
                        f"   Confidence: {signal.confidence:.1%}, Risk: {signal.risk_level}"
                    )
                    print(f"   Expected duration: {signal.expected_duration}min")
                    print(f"   Reasoning: {', '.join(signal.reasoning[:2])}")
                    return True
                else:
                    print(f"❌ Failed to place {signal.action} order")
                    return False
            else:
                # Fallback to direct execution
                success = self.original_bot.trade_executor.execute_trade(
                    volume, signal.action, optimal_price
                )
                if success:
                    print(
                        f"✅ {signal.action.upper()} executed: {volume:.8f} BTC at €{optimal_price:.2f}"
                    )
                    return True
                else:
                    print(f"❌ Failed to execute {signal.action}")
                    return False

        except Exception as e:
            print(f"❌ Error executing enhanced trade: {e}")
            return False

    def _log_enhanced_decision(
        self,
        signal: TradeSignal,
        market_regime: Dict,
        indicators_data: Dict,
        position_btc: float,
    ):
        """Log enhanced decision with all context"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "action": signal.action,
            "confidence": signal.confidence,
            "urgency": signal.urgency,
            "risk_level": signal.risk_level,
            "position_size_btc": position_btc,
            "market_regime": market_regime.get("regime", "unknown"),
            "regime_confidence": market_regime.get("confidence", 0),
            "volatility": market_regime.get("volatility", 0),
            "price": indicators_data.get("current_price", 0),
            "rsi": indicators_data.get("rsi", 50),
            "sentiment": indicators_data.get("sentiment", 0),
            "risk_off_prob": indicators_data.get("news_analysis", {}).get(
                "risk_off_probability", 0
            ),
            "reasoning": " | ".join(signal.reasoning),
            "expected_duration": signal.expected_duration,
        }

        # Save to enhanced log file
        enhanced_log_file = "enhanced_decisions.json"
        try:
            if os.path.exists(enhanced_log_file):
                with open(enhanced_log_file, "r") as f:
                    log_history = json.load(f)
            else:
                log_history = []

            log_history.append(log_data)

            # Keep last 1000 entries
            log_history = log_history[-1000:]

            with open(enhanced_log_file, "w") as f:
                json.dump(log_history, f, indent=2)

        except Exception as e:
            print(f"⚠️ Could not save enhanced log: {e}")

    def _retrain_model(self):
        """Retrain the ML model with recent trade outcomes"""
        try:
            print("🧠 Retraining ML model with recent trade outcomes...")

            # Load updated bot logs
            if os.path.exists(self.original_bot.data_manager.bot_logs_file):
                df = pd.read_csv(self.original_bot.data_manager.bot_logs_file)
                if len(df) > 50:
                    self.learning_engine.train_on_historical_data(df)
                    print("✅ Model retrained successfully")
                else:
                    print("⚠️ Insufficient data for retraining")
        except Exception as e:
            print(f"❌ Error retraining model: {e}")

    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        try:
            # Get basic performance from original bot
            basic_report = (
                self.original_bot.performance_tracker.generate_performance_report()
            )

            # Add enhanced metrics
            recent_performance = self._calculate_recent_performance()

            # Market regime analysis
            prices, volumes = self.original_bot.data_manager.load_price_history()
            current_regime = self.regime_detector.detect_regime(prices, volumes)

            # ML model status
            ml_status = {
                "model_trained": self.learning_engine.is_trained,
                "recent_trades_analyzed": len(self.trade_history),
                "confidence_in_predictions": (
                    "High" if self.learning_engine.is_trained else "Building"
                ),
            }

            enhanced_summary = {
                "basic_performance": basic_report,
                "recent_performance": recent_performance,
                "current_market_regime": current_regime,
                "ml_status": ml_status,
                "enhancement_status": "Active",
                "last_update": datetime.now().isoformat(),
            }

            return enhanced_summary

        except Exception as e:
            print(f"❌ Error generating performance summary: {e}")
            return {"error": str(e)}

    def print_enhanced_status(self):
        """Print detailed status of the enhanced bot"""
        print("\n" + "=" * 60)
        print("🚀 ENHANCED TRADING BOT STATUS")
        print("=" * 60)

        try:
            # Current market conditions
            prices, volumes = self.original_bot.data_manager.load_price_history()
            current_price, _ = self.original_bot.trade_executor.fetch_current_price()

            if current_price:
                print(f"💰 Current BTC Price: €{current_price:.2f}")

            # Market regime
            regime = self.regime_detector.detect_regime(prices, volumes)
            print(
                f"📊 Market Regime: {regime['regime']} (confidence: {regime['confidence']:.1%})"
            )
            print(f"📈 Trend Strength: {regime['trend_strength']:+.2%}")
            print(
                f"🌊 Volatility: {regime['volatility_regime']} ({regime['volatility']:.1%})"
            )

            # ML Status
            print(
                f"🧠 ML Model: {'Trained' if self.learning_engine.is_trained else 'Learning'}"
            )
            print(f"📚 Trade History: {len(self.trade_history)} enhanced trades")

            # Recent performance
            recent_perf = self._calculate_recent_performance()
            print(f"🎯 Recent Win Rate: {recent_perf['win_rate']:.1%}")
            print(f"📊 Recent Trades: {recent_perf.get('total_trades', 0)}")

            # Current balances
            btc_balance = self.original_bot.trade_executor.get_total_btc_balance() or 0
            eur_balance = (
                self.original_bot.trade_executor.get_available_balance("EUR") or 0
            )

            print(f"💎 BTC Balance: {btc_balance:.8f}")
            print(f"💶 EUR Balance: €{eur_balance:.2f}")

            if current_price and btc_balance:
                total_value = eur_balance + (btc_balance * current_price)
                print(f"💰 Total Value: €{total_value:.2f}")

            # Pending orders
            if self.original_bot.order_manager:
                pending = self.original_bot.order_manager.get_pending_orders()
                print(f"⏳ Pending Orders: {len(pending)}")

        except Exception as e:
            print(f"❌ Error displaying status: {e}")

        print("=" * 60)


# Integration function to enhance existing bot
def enhance_existing_bot(original_bot):
    """
    Enhance an existing trading bot with adaptive learning capabilities.

    Usage:
    enhanced_bot = enhance_existing_bot(your_original_bot)
    enhanced_bot.execute_enhanced_strategy()
    """
    return EnhancedTradingBot(original_bot)


# Example usage and testing function
def test_enhanced_bot():
    """Test function to validate the enhanced bot"""
    print("🧪 Testing Enhanced Trading Bot Components...")

    # Test learning engine
    learning_engine = AdaptiveLearningEngine()
    print("✅ Learning engine initialized")

    # Test regime detector
    regime_detector = MarketRegimeDetector()
    test_prices = list(range(100, 150)) + list(
        range(150, 120, -1)
    )  # Trend then reversal
    test_volumes = [1000] * len(test_prices)
    regime = regime_detector.detect_regime(test_prices, test_volumes)
    print(f"✅ Market regime detected: {regime['regime']}")

    # Test position sizer
    position_sizer = AdaptivePositionSizer()
    test_signal = TradeSignal("buy", 0.8, 0.6, 0.1, ["test"], "medium", 180)
    size = position_sizer.calculate_position_size(
        test_signal, 1000, regime, {"win_rate": 0.6}
    )
    print(f"✅ Position size calculated: €{size:.2f}")

    print("🎉 All components tested successfully!")


if __name__ == "__main__":
    test_enhanced_bot()
