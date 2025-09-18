# bitcoin_bot/core/bot.py
"""
Unified Bitcoin Trading Bot
Consolidates all trading logic into a single, configurable class
"""

import time
import os
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class TradingAction(Enum):
    """Trading action enumeration"""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class RiskLevel(Enum):
    """Risk level enumeration"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class MarketIndicators:
    """Market indicators data structure"""

    current_price: float
    current_volume: float
    rsi: float = 50.0
    macd: float = 0.0
    signal: float = 0.0
    bollinger_upper: float = 0.0
    bollinger_middle: float = 0.0
    bollinger_lower: float = 0.0
    ma_short: float = 0.0
    ma_long: float = 0.0
    vwap: float = 0.0
    volatility: float = 0.02
    sentiment: float = 0.0
    risk_off_probability: float = 0.0
    ml_success_probability: Optional[float] = None
    peak_probability: Optional[float] = None
    peak_recommendation: Optional[str] = None
    # OnChain indicators
    netflow: float = 0.0
    fee_rate: float = 0.0
    onchain_volume: float = 0.0
    old_utxos: float = 0.0
    
    market_regime: str = "ranging"

@dataclass
class TradingSignal:
    """Trading signal with comprehensive details"""

    action: TradingAction
    confidence: float
    volume: float
    price: float
    reasoning: List[str] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.MEDIUM
    expected_duration_minutes: int = 180
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size_pct: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BotConfiguration:
    """Bot configuration parameters"""

    # Feature flags
    enable_ml: bool = True
    enable_peak_detection: bool = True
    enable_onchain_analysis: bool = True
    enable_news_sentiment: bool = True

    # Trading parameters
    max_daily_trades: int = 8
    max_position_size_pct: float = 0.25
    min_position_size_pct: float = 0.01
    base_position_size_pct: float = 0.10

    # Risk parameters
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.10
    max_risk_per_trade_pct: float = 0.02

    # Timing parameters
    trade_cooldown_seconds: int = 180
    order_timeout_seconds: int = 300

    # Technical parameters
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    min_confidence_threshold: float = 0.6

    # ML parameters
    ml_retrain_interval_trades: int = 50
    ml_min_training_samples: int = 100

    # Peak detection parameters
    peak_lookback_days: int = 30
    peak_avoid_threshold: float = 0.7


class TradingBot:
    """
    Unified trading bot with all features consolidated.
    Replaces TradingBot, EnhancedTradingBot, and UltimateAdaptiveBot.
    """

    def __init__(
        self,
        data_manager,
        trade_executor,
        order_manager,
        config: Optional[BotConfiguration] = None,
        onchain_analyzer=None,
    ):
        """
        Initialize the unified trading bot.

        Args:
            data_manager: Data management instance
            trade_executor: Trade execution instance
            order_manager: Order management instance
            config: Bot configuration (uses defaults if None)
            onchain_analyzer: Optional on-chain analysis instance
        """
        # Core components
        self.data_manager = data_manager
        self.trade_executor = trade_executor
        self.order_manager = order_manager
        self.onchain_analyzer = onchain_analyzer

        # Configuration
        self.config = config or BotConfiguration()

        # Optional components (lazy loaded)
        self._ml_engine = None
        self._peak_system = None
        self._performance_tracker = None

        # Trading state
        self.session_start = datetime.now()
        self.daily_trades = 0
        self.last_trade_time = 0
        self.last_daily_reset = datetime.now().date()

        # Price history cache
        self.price_history = deque(maxlen=1000)
        self.volume_history = deque(maxlen=1000)

        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0

        # Thread pool for concurrent operations
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Initialize components
        self._initialize_components()

        logger.info(f"Trading bot initialized with config: {self.config}")

    def _initialize_components(self):
        """Initialize optional components based on configuration"""
        # Initialize performance tracker
        self._init_performance_tracker()

        # Initialize ML if enabled
        if self.config.enable_ml:
            self._init_ml_engine()

        # Initialize peak detection if enabled
        if self.config.enable_peak_detection:
            self._init_peak_detection()

        # Load initial price history
        self._load_price_history()

    def _init_performance_tracker(self):
        """Initialize performance tracking"""
        try:
            from performance_tracker import PerformanceTracker

            btc_balance = self.trade_executor.get_total_btc_balance() or 0
            eur_balance = self.trade_executor.get_available_balance("EUR") or 0

            self._performance_tracker = PerformanceTracker(btc_balance, eur_balance)
            logger.info("Performance tracker initialized")
        except Exception as e:
            logger.warning(f"Performance tracker initialization failed: {e}")
            self._performance_tracker = None

    def _init_ml_engine(self):
        """Initialize machine learning engine"""
        try:
            from enhanced_trading_bot import AdaptiveLearningEngine

            self._ml_engine = AdaptiveLearningEngine()

            # Train on historical data if available
            if os.path.exists(self.data_manager.bot_logs_file):
                df = pd.read_csv(self.data_manager.bot_logs_file)
                if len(df) >= self.config.ml_min_training_samples:
                    self._ml_engine.train_on_historical_data(df)
                    logger.info(f"ML engine trained on {len(df)} samples")
                else:
                    logger.info(
                        f"Insufficient data for ML training ({len(df)} samples)"
                    )

            logger.info("ML engine initialized")
        except Exception as e:
            logger.warning(f"ML engine initialization failed: {e}")
            self._ml_engine = None

    def _init_peak_detection(self):
        """Initialize peak detection system"""
        try:
            from peak_avoidance_system import PeakAvoidanceSystem

            self._peak_system = PeakAvoidanceSystem(
                lookback_days=self.config.peak_lookback_days
            )

            # Analyze historical patterns if available
            prices, volumes = self.data_manager.load_price_history()
            if len(prices) > 100:
                self._peak_system.analyze_price_history(
                    prices[-500:],
                    volumes[-500:],
                    [
                        datetime.now() - timedelta(minutes=15 * i)
                        for i in range(min(500, len(prices)))
                    ][::-1],
                    [{}] * min(500, len(prices)),
                )

            logger.info("Peak detection system initialized")
        except Exception as e:
            logger.warning(f"Peak detection initialization failed: {e}")
            self._peak_system = None

    def _load_price_history(self):
        """Load historical price data into cache"""
        try:
            prices, volumes = self.data_manager.load_price_history()

            # Populate deques
            for price, volume in zip(prices[-1000:], volumes[-1000:]):
                self.price_history.append(price)
                self.volume_history.append(volume)

            logger.info(f"Loaded {len(self.price_history)} historical price points")
        except Exception as e:
            logger.error(f"Failed to load price history: {e}")

    def analyze_market(self) -> MarketIndicators:
        """
        Enhanced market analysis with real-time data updates
        """
        # Get current market data
        current_price, current_volume = self.trade_executor.fetch_current_price()
        if not current_price:
            raise ValueError("Cannot fetch current price")
        
        # CRITICAL FIX: Get recent OHLC data and append to history
        self._update_price_history()
        
        # Update history with current data point
        self.price_history.append(current_price)
        self.volume_history.append(current_volume)
        
        # Convert to lists for calculations
        prices = list(self.price_history)
        volumes = list(self.volume_history)

        if len(prices) < 50:
            raise ValueError(f"Insufficient price history: {len(prices)} points")

        # Import indicators
        from core.indicators import (
            calculate_rsi,
            calculate_macd,
            calculate_bollinger_bands,
            calculate_moving_average,
            calculate_vwap,
        )

        # Calculate technical indicators
        rsi = calculate_rsi(prices) or 50.0
        macd, signal = calculate_macd(prices) or (0.0, 0.0)
        upper, middle, lower = calculate_bollinger_bands(prices) or (
            current_price,
            current_price,
            current_price,
        )
        ma_short = calculate_moving_average(prices, 20) or current_price
        ma_long = calculate_moving_average(prices, 50) or current_price
        vwap = calculate_vwap(prices, volumes) or current_price

        volatility = self._calculate_volatility_safe(prices)

        # Determine market regime
        market_regime = self._determine_market_regime(prices, ma_short, ma_long)

        # Initialize indicators object
        indicators = MarketIndicators(
            current_price=current_price,
            current_volume=current_volume,
            rsi=rsi,
            macd=macd,
            signal=signal,
            bollinger_upper=upper,
            bollinger_middle=middle,
            bollinger_lower=lower,
            ma_short=ma_short,
            ma_long=ma_long,
            vwap=vwap,
            volatility=volatility,
            market_regime=market_regime,
        )

        # Add sentiment analysis if enabled
        if self.config.enable_news_sentiment:
            self._add_sentiment_analysis(indicators)

        # Add ML predictions if available
        if self.config.enable_ml and self._ml_engine and self._ml_engine.is_trained:
            self._add_ml_predictions(indicators)

        # Add peak detection if available
        if self.config.enable_peak_detection and self._peak_system:
            self._add_peak_analysis(indicators, prices, volumes)

        # Add on-chain analysis if available
        if self.config.enable_onchain_analysis and self.onchain_analyzer:
            self._add_onchain_analysis(indicators)

        logger.debug(
            f"Market analysis complete: Price={current_price:.2f}, RSI={rsi:.1f}, Volatility={volatility:.4f}"
        )

        return indicators
    
    def _update_price_history(self):
        """Update price history with recent OHLC data"""
        try:
            # Get recent OHLC data from the exchange
            recent_ohlc = self.trade_executor.get_ohlc_data(
                pair="BTC/EUR",
                interval='15m',
                limit=100  # Get last 100 candles
            )
            
            if recent_ohlc:
                # Append new data to price history
                added_count = self.data_manager.append_ohlc_data(recent_ohlc)
                
                if added_count > 0:
                    logger.info(f"Updated price history with {added_count} new candles")
                    
                    # Reload the updated price history
                    prices, volumes = self.data_manager.load_price_history()
                    
                    # Update our cached history with recent data
                    if len(prices) > len(self.price_history):
                        # Clear and repopulate with fresh data
                        self.price_history.clear()
                        self.volume_history.clear()
                        
                        # Take the most recent 1000 points
                        for price, volume in zip(prices[-1000:], volumes[-1000:]):
                            self.price_history.append(price)
                            self.volume_history.append(volume)
                        
                        logger.info(f"Price history updated: {len(self.price_history)} points")
                
        except Exception as e:
            logger.warning(f"Failed to update price history: {e}")



    def _calculate_volatility_safe(self, prices, window=20):
        """
        Safely calculate volatility from price data

        Args:
            prices: List or array of prices
            window: Number of periods to use for calculation

        Returns:
            float: Volatility as standard deviation of returns
        """
        try:
            if len(prices) < window + 1:
                return 0.02  # Default volatility

            # Get the required number of prices
            recent_prices = np.array(
                prices[-window - 1 :]
            )  # Need window+1 prices to get window returns

            # Calculate price changes
            price_changes = np.diff(recent_prices)  # This gives us 'window' changes

            # Calculate returns (avoiding division by zero)
            previous_prices = recent_prices[:-1]  # First 'window' prices

            # Create mask for non-zero prices
            valid_mask = previous_prices != 0

            if not np.any(valid_mask):
                return 0.02  # Default if all prices are zero

            # Calculate returns only for valid prices
            returns = np.zeros_like(price_changes)
            returns[valid_mask] = (
                price_changes[valid_mask] / previous_prices[valid_mask]
            )

            # Calculate volatility
            volatility = float(np.std(returns))

            # Sanity check - cap volatility at reasonable bounds
            volatility = max(0.001, min(1.0, volatility))

            return volatility

        except Exception as e:
            logger.warning(f"Volatility calculation failed: {e}")
            return 0.02  # Safe default

    

    def _determine_market_regime(
        self, prices: List[float], ma_short: float, ma_long: float
    ) -> str:
        """Determine current market regime"""
        if len(prices) < 50:
            return "unknown"

        # Simple regime detection based on moving averages
        if ma_short > ma_long * 1.02:
            return "uptrend"
        elif ma_short < ma_long * 0.98:
            return "downtrend"
        else:
            return "ranging"

    def _add_sentiment_analysis(self, indicators: MarketIndicators):
        """Add news sentiment analysis to indicators"""
        try:
            from core.indicators import (
                fetch_enhanced_news,
                calculate_enhanced_sentiment,
            )

            articles = fetch_enhanced_news(top_n=10)
            if articles:
                analysis = calculate_enhanced_sentiment(articles)
                indicators.sentiment = analysis.get("sentiment", 0.0)
                indicators.risk_off_probability = analysis.get(
                    "risk_off_probability", 0.0
                )
                logger.debug(
                    f"Sentiment: {indicators.sentiment:.3f}, Risk-off: {indicators.risk_off_probability:.3f}"
                )
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            

    def _add_ml_predictions(self, indicators: MarketIndicators):
        """Add ML predictions to indicators"""
        try:
            indicators_dict = {
                "rsi": indicators.rsi,
                "macd": indicators.macd,
                "signal": indicators.signal,
                "current_price": indicators.current_price,
                "upper_band": indicators.bollinger_upper,
                "lower_band": indicators.bollinger_lower,
                "vwap": indicators.vwap,
                "sentiment": indicators.sentiment,
                "volatility": indicators.volatility,
                "news_analysis": {
                    "risk_off_probability": indicators.risk_off_probability
                },
            }

            fail_prob, success_prob = self._ml_engine.predict_trade_success(
                indicators_dict
            )
            indicators.ml_success_probability = success_prob
            logger.debug(f"ML prediction: {success_prob:.3f} success probability")
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")

    def _add_peak_analysis(
        self, indicators: MarketIndicators, prices: List[float], volumes: List[float]
    ):
        """Add peak detection analysis to indicators"""
        try:
            indicators_dict = {
                "rsi": indicators.rsi,
                "macd": indicators.macd,
                "volume_ratio": (
                    indicators.current_volume / np.mean(volumes[-10:])
                    if len(volumes) >= 10
                    else 1.0
                ),
            }

            analysis = self._peak_system.predict_peak_probability(
                indicators.current_price, indicators_dict, prices[-100:], volumes[-100:]
            )

            indicators.peak_probability = analysis.get("peak_probability", 0.0)
            indicators.peak_recommendation = analysis.get(
                "recommended_action", "neutral"
            )
            logger.debug(f"Peak probability: {indicators.peak_probability:.3f}")
        except Exception as e:
            logger.warning(f"Peak analysis failed: {e}")

    def _add_onchain_analysis(self, indicators: MarketIndicators):
        """Add on-chain analysis to indicators"""
        try:
            signals = self.onchain_analyzer.get_onchain_signals()
            indicators.netflow = signals.get("netflow", 0.0)
            indicators.fee_rate = signals.get("fee_rate", 0.0)
            logger.debug(
                f"On-chain: Netflow={indicators.netflow:.2f}, FeeRate={indicators.fee_rate:.2f}"
            )
        except Exception as e:
            logger.warning(f"On-chain analysis failed: {e}")

    def generate_signal(self, indicators: MarketIndicators) -> TradingSignal:
        """
        Generate trading signal from market indicators.

        Args:
            indicators: Market indicators object

        Returns:
            TradingSignal with action, confidence, and details
        """
        # Initialize default signal
        action = TradingAction.HOLD
        confidence = 0.0
        reasoning = []
        risk_level = self._assess_risk_level(indicators)

        # Emergency conditions check
        emergency_action = self._check_emergency_conditions(indicators)
        if emergency_action:
            action = emergency_action
            confidence = 0.9
            reasoning.append("EMERGENCY: Extreme market conditions")

        # Normal trading logic
        else:
            # Calculate signal scores
            buy_score, buy_reasons = self._calculate_buy_score(indicators)
            sell_score, sell_reasons = self._calculate_sell_score(indicators)

            # Determine action based on scores
            if buy_score >= 4 and risk_level != RiskLevel.EXTREME:
                action = TradingAction.BUY
                confidence = min(0.9, buy_score / 7.0)
                reasoning = buy_reasons

            elif sell_score >= 4 or (sell_score >= 3 and risk_level == RiskLevel.HIGH):
                action = TradingAction.SELL
                confidence = min(0.9, sell_score / 7.0)
                reasoning = sell_reasons

            else:
                action = TradingAction.HOLD
                confidence = 0.5
                reasoning = [
                    f"Insufficient signals: Buy={buy_score}/7, Sell={sell_score}/7"
                ]

        # Check for overrides
        if self.config.enable_peak_detection and indicators.peak_probability:
            action, confidence, reasoning = self._apply_peak_override(
                action, confidence, reasoning, indicators
            )

        # Calculate position size
        volume = self._calculate_position_size(
            action, indicators, confidence, risk_level
        )

        # Calculate stop loss and take profit
        stop_loss, take_profit = self._calculate_risk_levels(action, indicators)

        # Create signal
        signal = TradingSignal(
            action=action,
            confidence=confidence,
            volume=volume,
            price=indicators.current_price,
            reasoning=reasoning[:3],  # Top 3 reasons
            risk_level=risk_level,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=volume,  # Will be converted to actual volume later
        )

        logger.info(
            f"Signal generated: {action.value.upper()} with {confidence:.1%} confidence"
        )
        logger.debug(f"Reasoning: {', '.join(reasoning[:3])}")

        return signal

    def _assess_risk_level(self, indicators: MarketIndicators) -> RiskLevel:
        """Assess current market risk level"""
        risk_score = 0

        # Risk factors
        if indicators.risk_off_probability > 0.7:
            risk_score += 3
        elif indicators.risk_off_probability > 0.5:
            risk_score += 2
        elif indicators.risk_off_probability > 0.3:
            risk_score += 1

        if indicators.volatility > 0.08:
            risk_score += 2
        elif indicators.volatility > 0.05:
            risk_score += 1

        if indicators.sentiment < -0.2:
            risk_score += 1

        # Determine risk level
        if risk_score >= 5:
            return RiskLevel.EXTREME
        elif risk_score >= 3:
            return RiskLevel.HIGH
        elif risk_score >= 1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _check_emergency_conditions(
        self, indicators: MarketIndicators
    ) -> Optional[TradingAction]:
        """Check for emergency trading conditions"""
        # Extreme risk-off
        if indicators.risk_off_probability > 0.8:
            logger.warning(
                f"EMERGENCY: Extreme risk-off probability {indicators.risk_off_probability:.1%}"
            )
            return TradingAction.SELL

        # Liquidation cascade detection
        if (
            indicators.volatility > 0.10
            and indicators.sentiment < -0.3
            and indicators.current_price < indicators.vwap * 0.95
        ):
            logger.warning("EMERGENCY: Potential liquidation cascade detected")
            return TradingAction.SELL

        return None

    def _calculate_buy_score(
        self, indicators: MarketIndicators
    ) -> Tuple[float, List[str]]:
        """Calculate buy signal score and reasons"""
        score = 0
        reasons = []
        logger.info(f"Calculating buy score for indicators: {indicators}")

        # Technical signals
        logger.info(f"Indicators RSI: {indicators.rsi} config RSI: {self.config.rsi_oversold}")
        if indicators.rsi < self.config.rsi_oversold:
            score += 1.5
            reasons.append(f"Oversold RSI: {indicators.rsi:.1f}")
        elif indicators.rsi < 45:
            score += 0.5
            reasons.append(f"Low RSI: {indicators.rsi:.1f}")

        print(f"Indicators VWAP: {indicators.vwap} Price: {indicators.current_price}")
        if indicators.current_price < indicators.vwap * 0.98:
            score += 1
            reasons.append("Price below VWAP")

        print(f"Indicators Bollinger Lower: {indicators.bollinger_lower} Price: {indicators.current_price}")
        if indicators.current_price < indicators.bollinger_lower:
            score += 1
            reasons.append("Price below Bollinger lower band")

        print(f"Indicators MACD: {indicators.macd} Signal: {indicators.signal}")
        if indicators.macd > indicators.signal:
            score += 0.5
            reasons.append("MACD bullish crossover")

        # Sentiment and ML
        print(f"Indicators Sentiment: {indicators.sentiment} Risk-off: {indicators.risk_off_probability}")
        if indicators.sentiment > 0.1:
            score += 1
            reasons.append(f"Positive sentiment: {indicators.sentiment:.3f}")

        if (
            indicators.ml_success_probability
            and indicators.ml_success_probability > 0.7
        ):
            score += 1.5
            reasons.append(
                f"High ML confidence: {indicators.ml_success_probability:.1%}"
            )

        # OnChain factors
        if hasattr(indicators, 'netflow'):
            # Negative netflow = accumulation (good for buying)
            if indicators.netflow < -5000:  # 5000+ BTC accumulation
                score += 1.5
                reasons.append(f"Strong accumulation: {indicators.netflow:.0f} BTC")
            elif indicators.netflow < -1000:  # 1000+ BTC accumulation
                score += 1
                reasons.append(f"Moderate accumulation: {indicators.netflow:.0f} BTC")
            elif indicators.netflow > 5000:  # Distribution
                score -= 1
                reasons.append(f"Exchange distribution: {indicators.netflow:.0f} BTC")
        
        if hasattr(indicators, 'old_utxos'):
            # Old UTXO movement can indicate long-term holders selling
            if indicators.old_utxos > 50:
                score -= 0.5
                reasons.append(f"Old UTXO movement: {indicators.old_utxos}")
        
        if hasattr(indicators, 'fee_rate'):
            # Low fees = less network congestion
            if indicators.fee_rate < 10:
                score += 0.5
                reasons.append(f"Low network fees: {indicators.fee_rate:.1f} sat/vB")
            elif indicators.fee_rate > 50:
                score -= 0.5
                reasons.append(f"High network fees: {indicators.fee_rate:.1f} sat/vB")

        print(f"DEBUG: Buy Score: {score}, Reasons: {reasons}")
        return score, reasons

    def _calculate_sell_score(
        self, indicators: MarketIndicators
    ) -> Tuple[float, List[str]]:
        """Calculate sell signal score and reasons"""
        score = 0
        reasons = []

        # Technical signals
        if indicators.rsi > self.config.rsi_overbought:
            score += 1.5
            reasons.append(f"Overbought RSI: {indicators.rsi:.1f}")
        elif indicators.rsi > 65:
            score += 0.5
            reasons.append(f"High RSI: {indicators.rsi:.1f}")

        if indicators.current_price > indicators.vwap * 1.02:
            score += 1
            reasons.append("Price above VWAP")

        if indicators.current_price > indicators.bollinger_upper:
            score += 1
            reasons.append("Price above Bollinger upper band")

        if indicators.macd < indicators.signal:
            score += 0.5
            reasons.append("MACD bearish crossover")

        # Risk and sentiment
        if indicators.risk_off_probability > 0.6:
            score += 2
            reasons.append(f"High risk-off: {indicators.risk_off_probability:.1%}")

        if indicators.sentiment < -0.2:
            score += 1
            reasons.append(f"Negative sentiment: {indicators.sentiment:.3f}")

        # ML and peak detection
        if (
            indicators.ml_success_probability
            and indicators.ml_success_probability < 0.3
        ):
            score += 1
            reasons.append(
                f"Low ML confidence: {indicators.ml_success_probability:.1%}"
            )

        if (
            indicators.peak_probability
            and indicators.peak_probability > self.config.peak_avoid_threshold
        ):
            score += 1.5
            reasons.append(f"High peak probability: {indicators.peak_probability:.1%}")

        # Market regime
        if indicators.market_regime == "downtrend":
            score += 1
            reasons.append("Downtrend regime")

         # OnChain factors for selling
        if hasattr(indicators, 'netflow'):
            # Positive netflow = distribution (bearish)
            if indicators.netflow > 5000:  # Major exchange inflow
                score += 1.5
                reasons.append(f"Major exchange inflow: {indicators.netflow:.0f} BTC")
            elif indicators.netflow > 1000:
                score += 1
                reasons.append(f"Exchange inflow: {indicators.netflow:.0f} BTC")
        
        if hasattr(indicators, 'old_utxos'):
            # Old coins moving often precedes selloffs
            if indicators.old_utxos > 100:
                score += 1
                reasons.append(f"Significant old UTXO movement: {indicators.old_utxos}")
        
        if hasattr(indicators, 'fee_rate'):
            # Very high fees can indicate panic selling
            if indicators.fee_rate > 100:
                score += 0.5
                reasons.append(f"Network congestion: {indicators.fee_rate:.1f} sat/vB")

        return score, reasons

    def _apply_peak_override(
        self,
        action: TradingAction,
        confidence: float,
        reasoning: List[str],
        indicators: MarketIndicators,
    ) -> Tuple[TradingAction, float, List[str]]:
        """Apply peak detection override to trading signal"""
        if not indicators.peak_probability:
            return action, confidence, reasoning

        # Override buy if peak probability is high
        if (
            action == TradingAction.BUY
            and indicators.peak_probability > self.config.peak_avoid_threshold
        ):

            logger.warning(
                f"PEAK OVERRIDE: Changing BUY to HOLD (peak prob: {indicators.peak_probability:.1%})"
            )
            return (
                TradingAction.HOLD,
                confidence * 0.5,
                ["Peak detection override"] + reasoning,
            )

        # Enhance sell signal if at peak
        if (
            action == TradingAction.SELL
            and indicators.peak_probability > self.config.peak_avoid_threshold
        ):

            confidence = min(0.95, confidence * 1.2)
            reasoning = [
                f"Peak detected: {indicators.peak_probability:.1%}"
            ] + reasoning

        return action, confidence, reasoning

    def _calculate_position_size(
        self,
        action: TradingAction,
        indicators: MarketIndicators,
        confidence: float,
        risk_level: RiskLevel,
    ) -> float:
        """Calculate position size based on risk and confidence"""
        if action == TradingAction.HOLD:
            return 0.0

        # Get current balances
        btc_balance = self.trade_executor.get_total_btc_balance() or 0
        eur_balance = self.trade_executor.get_available_balance("EUR") or 0

        # Base position size
        base_size = self.config.base_position_size_pct

        # Confidence adjustment (0.5x to 1.5x)
        confidence_mult = 0.5 + confidence

        # Risk adjustment
        risk_mult = {
            RiskLevel.LOW: 1.2,
            RiskLevel.MEDIUM: 1.0,
            RiskLevel.HIGH: 0.6,
            RiskLevel.EXTREME: 0.3,
        }[risk_level]

        # Volatility adjustment
        vol_mult = max(0.5, min(1.5, 1.0 - (indicators.volatility - 0.02) * 5))

        # ML confidence adjustment
        ml_mult = 1.0
        if indicators.ml_success_probability:
            if indicators.ml_success_probability > 0.7:
                ml_mult = 1.2
            elif indicators.ml_success_probability < 0.3:
                ml_mult = 0.6

        # Calculate final position size
        position_pct = base_size * confidence_mult * risk_mult * vol_mult * ml_mult

        # Apply bounds
        position_pct = max(
            self.config.min_position_size_pct,
            min(self.config.max_position_size_pct, position_pct),
        )

        # Convert to actual volume
        if action == TradingAction.BUY:
            position_eur = eur_balance * position_pct
            position_btc = position_eur / indicators.current_price
        else:  # SELL
            position_btc = btc_balance * position_pct

        logger.debug(f"Position size: {position_pct:.1%} ({position_btc:.8f} BTC)")

        return position_btc

    def _calculate_risk_levels(
        self, action: TradingAction, indicators: MarketIndicators
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels"""
        if action == TradingAction.HOLD:
            return None, None

        current_price = indicators.current_price

        # Dynamic stop loss based on volatility
        stop_mult = self.config.stop_loss_pct
        if indicators.volatility > 0.05:
            stop_mult *= 1.5  # Wider stop in high volatility

        # Dynamic take profit based on trend
        profit_mult = self.config.take_profit_pct
        if indicators.market_regime == "uptrend" and action == TradingAction.BUY:
            profit_mult *= 1.5  # Higher target in uptrend

        if action == TradingAction.BUY:
            stop_loss = current_price * (1 - stop_mult)
            take_profit = current_price * (1 + profit_mult)
        else:  # SELL
            stop_loss = current_price * (1 + stop_mult)
            take_profit = current_price * (1 - profit_mult)

        return stop_loss, take_profit

    def execute_strategy(self):
        """
        Main strategy execution method.
        Orchestrates the complete trading cycle.
        """
        try:
            # Check daily reset
            self._check_daily_reset()

            # Check if we can trade
            if not self._can_trade():
                logger.info("Trading conditions not met, skipping cycle")
                return

            # Update pending orders
            self._update_pending_orders()

            # Analyze market
            indicators = self.analyze_market()

            # Generate trading signal
            signal = self.generate_signal(indicators)

            # Log decision
            self._log_trading_decision(signal, indicators)

            # Execute trade if not holding
            if signal.action != TradingAction.HOLD:
                success = self._execute_trade(signal)

                if success:
                    self.daily_trades += 1
                    self.total_trades += 1
                    self.last_trade_time = time.time()

                    # Update performance tracker
                    if self._performance_tracker:
                        self._performance_tracker.record_trade(
                            order_id=f"trade_{int(time.time())}",
                            side=signal.action.value,
                            volume=signal.volume,
                            price=signal.price,
                            fee=signal.price * signal.volume * 0.0025,
                            timestamp=time.time(),
                        )

            # Update equity tracking
            self._update_equity_tracking(indicators.current_price)

            # Retrain ML if needed
            if (
                self.config.enable_ml
                and self.total_trades % self.config.ml_retrain_interval_trades == 0
            ):
                self._retrain_ml_model()

            logger.info(f"Strategy cycle complete: {signal.action.value.upper()}")

        except Exception as e:
            logger.error(f"Strategy execution failed: {e}", exc_info=True)

    def _check_daily_reset(self):
        """Check and perform daily reset if needed"""
        current_date = datetime.now().date()
        if current_date > self.last_daily_reset:
            logger.info("Performing daily reset")
            self.daily_trades = 0
            self.last_daily_reset = current_date

    def _can_trade(self) -> bool:
        """Check if trading conditions are met"""
        # Check daily limit
        if self.daily_trades >= self.config.max_daily_trades:
            logger.warning(
                f"Daily trade limit reached ({self.daily_trades}/{self.config.max_daily_trades})"
            )
            return False

        # Check cooldown
        if time.time() - self.last_trade_time < self.config.trade_cooldown_seconds:
            remaining = self.config.trade_cooldown_seconds - (
                time.time() - self.last_trade_time
            )
            logger.info(f"Trade cooldown active: {remaining:.0f}s remaining")
            return False

        return True

    def _update_pending_orders(self):
        """Update status of pending orders"""
        if not self.order_manager:
            return

        try:
            results = self.order_manager.check_and_update_orders()

            if results["filled"]:
                logger.info(f"Orders filled: {results['filled']}")
                for order_id in results["filled"]:
                    self._process_filled_order(order_id)

            if results["cancelled"]:
                logger.info(f"Orders cancelled: {results['cancelled']}")
                # FIXED: Reset strategies when orders are cancelled
                for order_id in results["cancelled"]:
                    self._process_cancelled_order(order_id)

        except Exception as e:
            logger.error(f"Failed to update pending orders: {e}")

    def _process_cancelled_order(self, order_id: str):
        """ADDED: Process a cancelled order and reset strategies"""
        try:
            logger.info(f"Processing cancelled order: {order_id}")
            
            # Reset DCA strategy so it can try again
            for name, strategy in self.strategies.items():
                if hasattr(strategy, 'reset_after_cancellation'):
                    strategy.reset_after_cancellation()
                    logger.info(f"Reset {name} strategy after order cancellation")
                elif name == 'dca' and hasattr(strategy, 'last_buy_time'):
                    # Manual reset for DCA strategy
                    strategy.last_buy_time = None  # Reset so it can buy again
                    logger.info(f"Reset DCA timing after cancelled order")
                    
        except Exception as e:
            logger.error(f"Failed to process cancelled order {order_id}: {e}")

    def _process_filled_order(self, order_id: str):
        """Process a filled order"""
        try:
            order_info = self.order_manager.filled_orders.get(order_id)
            if not order_info:
                return

            # Update win/loss tracking
            if order_info["side"] == "sell":
                # Calculate profit/loss
                # (This would need access to average buy price)
                pass

            logger.info(f"Processed filled order: {order_id}")

        except Exception as e:
            logger.error(f"Failed to process filled order {order_id}: {e}")

    

    def _execute_trade(self, signal: TradingSignal) -> bool:
        """Execute trade based on signal - FIXED ORDER BOOK VERSION"""
        try:
            # FIXED: More intelligent minimum volume handling
            min_volume = 0.0001  # Bitvavo minimum
            
            if signal.volume < min_volume:
                # Try to increase volume to minimum if we have enough balance
                if signal.action == TradingAction.BUY:
                    eur_balance = self.trade_executor.get_available_balance("EUR") or 0
                    max_affordable = (eur_balance * 0.95) / signal.price  # Use 95% of balance
                    
                    if max_affordable >= min_volume:
                        logger.info(f"🔧 Increasing volume from {signal.volume:.8f} to minimum {min_volume:.8f}")
                        signal.volume = min_volume
                    else:
                        logger.warning(f"❌ Insufficient balance for minimum trade: need €{min_volume * signal.price:.2f}, have €{eur_balance:.2f}")
                        return False
                else:  # SELL
                    btc_balance = self.trade_executor.get_total_btc_balance() or 0
                    if btc_balance >= min_volume:
                        logger.info(f"🔧 Increasing sell volume to minimum {min_volume:.8f}")
                        signal.volume = min_volume
                    else:
                        logger.warning(f"❌ Insufficient BTC for minimum sell: need {min_volume:.8f}, have {btc_balance:.8f}")
                        return False

            # Get order book
            order_book = self.trade_executor.get_btc_order_book()
            if not order_book:
                logger.error("Cannot fetch order book")
                return False

            # FIXED: Handle different order book structures safely
            is_dca_order = signal.reasoning and any('DCA' in str(reason) for reason in signal.reasoning)
            
            if is_dca_order:
                # For DCA orders, use market price for immediate execution
                logger.info(f"🎯 DCA order detected - using market price for immediate fill")
                
                try:
                    if signal.action == TradingAction.BUY:
                        # Try different order book structures
                        if isinstance(order_book, dict) and 'asks' in order_book:
                            asks = order_book['asks']
                            if asks and len(asks) > 0:
                                # Handle both dict and list structures
                                if isinstance(asks[0], dict):
                                    optimal_price = asks[0]['price']
                                elif isinstance(asks[0], list) and len(asks[0]) >= 2:
                                    optimal_price = asks[0][0]  # First element is usually price
                                else:
                                    optimal_price = float(asks[0])
                            else:
                                raise ValueError("Empty asks")
                        else:
                            raise ValueError("Invalid order book structure")
                            
                    else:  # SELL
                        if isinstance(order_book, dict) and 'bids' in order_book:
                            bids = order_book['bids']
                            if bids and len(bids) > 0:
                                if isinstance(bids[0], dict):
                                    optimal_price = bids[0]['price']
                                elif isinstance(bids[0], list) and len(bids[0]) >= 2:
                                    optimal_price = bids[0][0]
                                else:
                                    optimal_price = float(bids[0])
                            else:
                                raise ValueError("Empty bids")
                        else:
                            raise ValueError("Invalid order book structure")
                            
                    logger.info(f"🎯 Market price for DCA {signal.action.value.upper()}: €{optimal_price:.2f}")
                    
                except (KeyError, IndexError, ValueError, TypeError) as e:
                    logger.warning(f"⚠️ Could not get market price: {e}, falling back to optimal price")
                    optimal_price = self.trade_executor.get_optimal_price(order_book, signal.action.value)
            else:
                # For regular trades, use optimal price calculation
                optimal_price = self.trade_executor.get_optimal_price(order_book, signal.action.value)

            if not optimal_price:
                logger.error("Cannot determine optimal price")
                return False

            # Place order through order manager
            order_id = self.order_manager.place_limit_order_with_timeout(
                volume=signal.volume,
                side=signal.action.value,
                price=optimal_price,
                timeout=self.config.order_timeout_seconds,
            )

            if order_id:
                order_type = "DCA" if is_dca_order else "REGULAR"
                logger.info(
                    f"✅ {order_type} Order placed: {order_id} - "
                    f"{signal.action.value.upper()} {signal.volume:.8f} BTC @ €{optimal_price:.2f}"
                )
                return True
            else:
                logger.error("Failed to place order")
                return False

        except Exception as e:
            logger.error(f"Trade execution failed: {e}", exc_info=True)
            return False

    def _log_trading_decision(
        self, signal: TradingSignal, indicators: MarketIndicators
    ):
        """Log trading decision for analysis"""
        try:
            self.data_manager.log_strategy(
                timestamp=signal.timestamp.isoformat(),
                price=indicators.current_price,
                trade_volume=(
                    signal.volume if signal.action != TradingAction.HOLD else 0
                ),
                side=signal.action.value if signal.action != TradingAction.HOLD else "",
                reason=" | ".join(signal.reasoning),
                rsi=indicators.rsi,
                macd=indicators.macd,
                signal=indicators.signal,
                ma_short=indicators.ma_short,
                ma_long=indicators.ma_long,
                upper_band=indicators.bollinger_upper,
                lower_band=indicators.bollinger_lower,
                sentiment=indicators.sentiment,
                volatility=indicators.volatility,
                buy_decision=signal.action == TradingAction.BUY,
                sell_decision=signal.action == TradingAction.SELL,
            )
        except Exception as e:
            logger.error(f"Failed to log decision: {e}")

    def _update_equity_tracking(self, current_price: float):
        """Update equity and performance tracking"""
        if not self._performance_tracker:
            return

        try:
            btc_balance = self.trade_executor.get_total_btc_balance() or 0
            eur_balance = self.trade_executor.get_available_balance("EUR") or 0

            self._performance_tracker.update_equity(
                btc_balance, eur_balance, current_price
            )
        except Exception as e:
            logger.error(f"Failed to update equity tracking: {e}")

    def _retrain_ml_model(self):
        """Retrain ML model with recent data"""
        if not self._ml_engine:
            return

        try:
            logger.info("Retraining ML model...")

            # Load recent trade data
            df = pd.read_csv(self.data_manager.bot_logs_file)
            if len(df) >= self.config.ml_min_training_samples:
                self._ml_engine.train_on_historical_data(df)
                logger.info(f"ML model retrained on {len(df)} samples")

        except Exception as e:
            logger.error(f"ML retraining failed: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive bot status"""
        try:
            btc_balance = self.trade_executor.get_total_btc_balance() or 0
            eur_balance = self.trade_executor.get_available_balance("EUR") or 0
            current_price, _ = self.trade_executor.fetch_current_price()

            status = {
                "timestamp": datetime.now().isoformat(),
                "session": {
                    "start_time": self.session_start.isoformat(),
                    "duration_hours": (
                        datetime.now() - self.session_start
                    ).total_seconds()
                    / 3600,
                    "daily_trades": self.daily_trades,
                    "total_trades": self.total_trades,
                },
                "balances": {
                    "btc": btc_balance,
                    "eur": eur_balance,
                    "total_value_eur": eur_balance + btc_balance * (current_price or 0),
                },
                "market": {
                    "current_price": current_price,
                    "price_history_size": len(self.price_history),
                },
                "configuration": {
                    "ml_enabled": self.config.enable_ml,
                    "peak_detection_enabled": self.config.enable_peak_detection,
                    "max_daily_trades": self.config.max_daily_trades,
                },
            }

            # Add performance metrics if available
            if self._performance_tracker:
                status["performance"] = (
                    self._performance_tracker.generate_performance_report()
                )

            return status

        except Exception as e:
            logger.error(f"Failed to generate status: {e}")
            return {"error": str(e)}

    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down trading bot...")

        try:
            # Cancel pending orders
            if self.order_manager:
                pending = self.order_manager.get_pending_orders()
                for order_id in pending:
                    try:
                        self.order_manager._cancel_order(order_id)
                        logger.info(f"Cancelled pending order: {order_id}")
                    except Exception as e:
                        logger.error(f"Failed to cancel order {order_id}: {e}")

            # Save ML model
            if self._ml_engine:
                self._ml_engine._save_model()

            # Save peak patterns
            if self._peak_system:
                self._peak_system._save_patterns()

            # Shutdown thread pool
            self.executor.shutdown(wait=True)

            logger.info("Bot shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
