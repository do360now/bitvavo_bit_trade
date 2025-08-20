# bitcoin_bot/unified_bot.py
"""
Unified Bitcoin Trading Bot
Consolidates all functionality into a single, comprehensive bot
"""

import time
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# from dataclasses import dataclass, field
# from enum import Enum
import numpy as np
import pandas as pd
import logging
import utils.logger as logger

# Import your existing components
try:
    # Try to import from your actual file structure
    from core.bot import (
        TradingBot as CoreBot,
        BotConfiguration,
        MarketIndicators,
        TradingSignal,
        TradingAction,
        RiskLevel,
    )
except ImportError:
    # Fallback for different structure
    from core.bot import (
        TradingBot as CoreBot,
        BotConfiguration,
        MarketIndicators,
        TradingSignal,
        TradingAction,
        RiskLevel,
    )

from core.data_manager import DataManager
from trading.executor import TradeExecutor
from trading.order_manager import OrderManager

# Optional advanced components with fallbacks
try:
    from analysis.ml_engine import MLEngine, MLConfig
except ImportError:
    logger.warning("ML engine not available - using mock")
    MLEngine = None
    MLConfig = None

try:
    from analysis.peak_detection import PeakAvoidanceSystem, PeakDetectionConfig
except ImportError:
    logger.warning("Peak detection not available - using mock")
    PeakAvoidanceSystem = None
    PeakDetectionConfig = None

try:
    from trading.strategies import StrategyFactory, StrategyType, StrategyConfig
except ImportError:
    logger.warning("Advanced strategies not available - using basic logic")
    StrategyFactory = None
    StrategyType = None
    StrategyConfig = None

logger = logging.getLogger(__name__)


class UnifiedTradingBot:
    """
    Unified trading bot that consolidates all functionality
    """

    def __init__(
        self,
        bitvavo_api,
        config: Optional[BotConfiguration] = None,
        enable_ml: bool = True,
        enable_peak_detection: bool = True,
        enable_advanced_strategies: bool = True,
        enable_onchain_analysis: bool = True,
    ):
        """
        Initialize unified trading bot

        Args:
            bitvavo_api: Authenticated Bitvavo API instance
            config: Bot configuration
            enable_ml: Enable machine learning features
            enable_peak_detection: Enable peak detection
            enable_advanced_strategies: Enable advanced trading strategies
        """
        self.bitvavo_api = bitvavo_api
        self.config = config or BotConfiguration()

        # Initialize core components
        self.data_manager = DataManager("./price_history.json", "./bot_logs.csv")
        self.trade_executor = TradeExecutor(bitvavo_api)
        self.order_manager = OrderManager(bitvavo_api)

        # Initialize the core bot
        self.core_bot = CoreBot(
            data_manager=self.data_manager,
            trade_executor=self.trade_executor,
            order_manager=self.order_manager,
            config=self.config,
        )

        # Optional components
        self.ml_engine = None
        self.peak_system = None
        self.strategies = {}

        # Initialize optional features
        if enable_ml:
            self._initialize_ml_engine()

        if enable_peak_detection:
            self._initialize_peak_detection()

        if enable_advanced_strategies:
            self._initialize_strategies()

         # Initialize onchain analyzer if enabled
        self.onchain_analyzer = None
        if enable_onchain_analysis:
            self._initialize_onchain_analyzer()

        # Performance tracking
        self.session_start = datetime.now()
        self.total_trades = 0
        self.successful_trades = 0
        self.total_pnl = 0.0

        # State management
        self.running = False
        self.last_update = time.time()

        self._performance_tracker = None
        self._init_performance_tracker()

        logger.info("Unified trading bot initialized successfully")


    def _initialize_onchain_analyzer(self):
        """Initialize on-chain analyzer"""
        try:
            from onchain_analyzer import OnChainAnalyzer
            
            self.onchain_analyzer = OnChainAnalyzer()
            
            # Test the connection
            signals = self.onchain_analyzer.get_onchain_signals()
            if signals:
                logger.info("OnChain analyzer initialized successfully")
                logger.info(f"Network status: Fee rate={signals.get('fee_rate', 0):.1f} sat/vB, "
                        f"Netflow={signals.get('netflow', 0):.1f} BTC")
            else:
                logger.warning("OnChain analyzer initialized but no data received")
                
        except ImportError:
            logger.warning("OnChain analyzer module not available")
            self.onchain_analyzer = None
        except Exception as e:
            logger.error(f"Failed to initialize OnChain analyzer: {e}")
        self.onchain_analyzer = None

    def _init_performance_tracker(self):
        """Initialize performance tracker with fallback"""
        try:
            # Try to import the real performance tracker
            from performance_tracker import PerformanceTracker

            btc_balance = self.trade_executor.get_total_btc_balance() or 0
            eur_balance = self.trade_executor.get_available_balance("EUR") or 0

            self._performance_tracker = PerformanceTracker(btc_balance, eur_balance)
            logger.info("Performance tracker initialized")
        except ImportError:
            # Use the simple fallback above
            btc_balance = self.trade_executor.get_total_btc_balance() or 0
            eur_balance = self.trade_executor.get_available_balance("EUR") or 0

            # Create simple inline tracker
            class SimplePerformanceTracker:
                def __init__(self, btc, eur):
                    self.trades = []
                    self.initial_btc = btc
                    self.initial_eur = eur

                def record_trade(self, order_id, side, volume, price, fee, timestamp):
                    self.trades.append({"side": side, "volume": volume, "price": price})

                def update_equity(self, btc_balance, eur_balance, btc_price):
                    pass

                def generate_performance_report(self):
                    return {"total_trades": len(self.trades), "equity_points": 0}

            self._performance_tracker = SimplePerformanceTracker(
                btc_balance, eur_balance
            )
            logger.info("Simple performance tracker initialized (fallback)")
        except Exception as e:
            logger.warning(f"Performance tracker initialization failed: {e}")
            self._performance_tracker = None

    def _initialize_ml_engine(self):
        """Initialize machine learning engine - FIXED VERSION"""
        if MLEngine is None:
            logger.warning("ML engine module not available")
            return

        try:
            # Import MLConfig here to access ModelType
            from analysis.ml_engine import MLConfig, ModelType

            # Use the enum properly
            ml_config = MLConfig(
                model_type=ModelType.ENSEMBLE,  # Use the enum directly
                prediction_task="trade_success",
                use_technical_features=True,
                use_market_features=True,
                use_sentiment_features=True,
                online_learning_enabled=True,
            )

            self.ml_engine = MLEngine(ml_config)

            # Try to load existing model or train on historical data
            if os.path.exists("./ml_model.pkl"):
                self.ml_engine.load_model("./ml_model.pkl")
                logger.info("Loaded existing ML model")
            else:
                # Train on historical data if available
                if os.path.exists(self.data_manager.bot_logs_file):
                    df = pd.read_csv(self.data_manager.bot_logs_file)
                    if len(df) >= 100:
                        self.ml_engine.train(df)
                        logger.info(f"Trained ML model on {len(df)} historical samples")

            logger.info("ML engine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ML engine: {e}")
            self.ml_engine = None

    def _initialize_peak_detection(self):
        """Initialize peak detection system"""
        if PeakAvoidanceSystem is None:
            logger.warning("Peak detection module not available")
            return

        try:
            peak_config = PeakDetectionConfig(
                lookback_periods=100,
                min_prominence_pct=0.02,
                enable_pattern_recognition=True,
                peak_avoidance_threshold=0.7,
            )

            self.peak_system = PeakAvoidanceSystem(peak_config)
            logger.info("Peak detection system initialized")

        except Exception as e:
            logger.error(f"Failed to initialize peak detection: {e}")
            self.peak_system = None

    def _initialize_strategies(self):
        """Initialize advanced trading strategies"""
        if StrategyFactory is None:
            logger.warning("Advanced strategies module not available")
            return

        try:
            # Create different strategies
            strategy_configs = {
                "dca": StrategyConfig(
                    name="DCA Strategy",
                    enabled=True,
                    params={"buy_interval_hours": 24, "buy_amount_percentage": 0.05},
                ),
                "momentum": StrategyConfig(
                    name="Momentum Strategy",
                    enabled=True,
                    params={"momentum_period": 20, "momentum_threshold": 0.02},
                ),
                "mean_reversion": StrategyConfig(
                    name="Mean Reversion Strategy",
                    enabled=True,
                    params={"lookback_period": 50, "deviation_threshold": 2.0},
                ),
                "hybrid": StrategyConfig(name="Hybrid Strategy", enabled=True),
            }

            for name, config in strategy_configs.items():
                try:
                    strategy_type = getattr(
                        StrategyType, name.upper() if name != "hybrid" else "HYBRID"
                    )
                    self.strategies[name] = StrategyFactory.create_strategy(
                        strategy_type, config
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize {name} strategy: {e}")

            logger.info(f"Initialized {len(self.strategies)} advanced strategies")

        except Exception as e:
            logger.error(f"Failed to initialize strategies: {e}")

    def analyze_market(self) -> MarketIndicators:
        """
        Enhanced market analysis combining all available data sources
        """
        # Get base market analysis from core bot
        indicators = self.core_bot.analyze_market()

        if self.onchain_analyzer:
            self._add_onchain_analysis(indicators)

        # Enhance with ML predictions if available
        if self.ml_engine and self.ml_engine.is_trained:
            try:
                market_data = {
                    "rsi": indicators.rsi,
                    "macd": indicators.macd,
                    "signal": indicators.signal,
                    "current_price": indicators.current_price,
                    "upper_band": indicators.bollinger_upper,
                    "lower_band": indicators.bollinger_lower,
                    "vwap": indicators.vwap,
                    "sentiment": indicators.sentiment,
                    "volatility": indicators.volatility,
                    "timestamp": datetime.now(),
                }

                prediction, confidence = self.ml_engine.predict(market_data)
                indicators.ml_success_probability = confidence

                logger.debug(f"ML prediction: {confidence:.3f} confidence")

            except Exception as e:
                logger.warning(f"ML prediction failed: {e}")

        # Enhance with peak detection if available
        if self.peak_system:
            try:
                # Get price history
                prices, volumes = self.data_manager.load_price_history()

                if len(prices) >= 50:
                    analysis = self.peak_system.analyze(
                        current_price=indicators.current_price,
                        prices=prices[-200:],  # Last 200 data points
                        volumes=volumes[-200:] if len(volumes) >= 200 else volumes,
                        indicators={
                            "rsi": indicators.rsi,
                            "macd": indicators.macd,
                            "volume_ratio": (
                                indicators.current_volume / np.mean(volumes[-10:])
                                if len(volumes) >= 10
                                else 1.0
                            ),
                        },
                    )

                    indicators.peak_probability = analysis.get("peak_probability", 0.0)
                    indicators.peak_recommendation = analysis.get(
                        "recommended_action", "normal_entry"
                    )

                    logger.debug(f"Peak probability: {indicators.peak_probability:.3f}")

            except Exception as e:
                logger.warning(f"Peak analysis failed: {e}")

        return indicators

    def _add_onchain_analysis(self, indicators: MarketIndicators):
        """Add on-chain analysis to indicators"""
        try:
            signals = self.onchain_analyzer.get_onchain_signals()
            
            indicators.netflow = signals.get('netflow', 0.0)
            indicators.fee_rate = signals.get('fee_rate', 0.0)
            indicators.onchain_volume = signals.get('volume', 0.0)
            indicators.old_utxos = signals.get('old_utxos', 0.0)
            
            logger.debug(f"OnChain data: Netflow={indicators.netflow:.1f}, "
                        f"Fees={indicators.fee_rate:.1f}, Volume={indicators.onchain_volume:.1f}")
            
        except Exception as e:
            logger.warning(f"OnChain analysis failed: {e}")
    
    def generate_unified_signal(self, indicators: MarketIndicators) -> TradingSignal:
        """
        Generate trading signal using multiple approaches and combine them
        """
        signals = []

        # Get signal from core bot
        core_signal = self.core_bot.generate_signal(indicators)
        signals.append(("core", core_signal, 1.0))  # Base weight of 1.0

        # Get signals from advanced strategies
        for name, strategy in self.strategies.items():
            try:
                should_trade, reason = strategy.should_trade(indicators)
                if should_trade:
                    analysis = strategy.analyze(indicators)
                    signal = strategy.generate_signal(indicators, analysis)

                    # Weight strategies based on their historical performance
                    performance = strategy.get_performance_metrics()
                    weight = max(0.5, performance.get("win_rate", 0.5))

                    signals.append((name, signal, weight))
                    logger.debug(
                        f"{name} strategy signal: {signal.action.value} (conf: {signal.confidence:.2f})"
                    )

            except Exception as e:
                logger.warning(f"Strategy {name} failed: {e}")
                continue

        # Combine signals using weighted voting
        return self._combine_signals(signals, indicators)

    def _combine_signals(
        self,
        signals: List[Tuple[str, TradingSignal, float]],
        indicators: MarketIndicators,
    ) -> TradingSignal:
        """
        Combine multiple trading signals using weighted voting
        """
        if not signals:
            return TradingSignal(
                action=TradingAction.HOLD,
                confidence=0.5,
                volume=0.0,
                price=indicators.current_price,
                reasoning=["No signals available"],
            )

        # Weighted voting
        buy_score = 0.0
        sell_score = 0.0
        hold_score = 0.0
        total_weight = 0.0

        all_reasoning = []

        for name, signal, weight in signals:
            total_weight += weight

            if signal.action == TradingAction.BUY:
                buy_score += weight * signal.confidence
                all_reasoning.extend([f"[{name}] {r}" for r in signal.reasoning[:1]])
            elif signal.action == TradingAction.SELL:
                sell_score += weight * signal.confidence
                all_reasoning.extend([f"[{name}] {r}" for r in signal.reasoning[:1]])
            else:
                hold_score += weight * 0.5

        # Normalize scores
        if total_weight > 0:
            buy_score /= total_weight
            sell_score /= total_weight
            hold_score /= total_weight

        # Determine final action
        max_score = max(buy_score, sell_score, hold_score)

        if max_score == buy_score and buy_score > 0.6:
            action = TradingAction.BUY
            confidence = min(0.95, buy_score)
        elif max_score == sell_score and sell_score > 0.6:
            action = TradingAction.SELL
            confidence = min(0.95, sell_score)
        else:
            action = TradingAction.HOLD
            confidence = 0.5
            all_reasoning = ["Insufficient consensus among strategies"]

        # Apply peak detection override
        if (
            self.peak_system
            and indicators.peak_probability
            and indicators.peak_probability > 0.7
            and action == TradingAction.BUY
        ):

            logger.warning(
                f"Peak override: Changing BUY to HOLD (peak prob: {indicators.peak_probability:.1%})"
            )
            action = TradingAction.HOLD
            confidence *= 0.7
            all_reasoning.insert(
                0, f"Peak detection override ({indicators.peak_probability:.1%})"
            )

        # Calculate position size using core bot logic
        volume = self.core_bot._calculate_position_size(
            action, indicators, confidence, self._assess_risk_level(indicators)
        )

        return TradingSignal(
            action=action,
            confidence=confidence,
            volume=volume,
            price=indicators.current_price,
            reasoning=all_reasoning[:3],  # Top 3 reasons
            risk_level=self._assess_risk_level(indicators),
        )

    def _assess_risk_level(self, indicators: MarketIndicators) -> RiskLevel:
        """Assess current risk level"""
        risk_score = 0

        # High volatility increases risk
        if indicators.volatility > 0.08:
            risk_score += 3
        elif indicators.volatility > 0.05:
            risk_score += 2
        elif indicators.volatility > 0.03:
            risk_score += 1

        # Negative sentiment increases risk
        if indicators.sentiment < -0.3:
            risk_score += 2
        elif indicators.sentiment < -0.1:
            risk_score += 1

        # Risk-off conditions
        if indicators.risk_off_probability > 0.7:
            risk_score += 3
        elif indicators.risk_off_probability > 0.5:
            risk_score += 2

        # Peak probability increases risk for buying
        if indicators.peak_probability and indicators.peak_probability > 0.7:
            risk_score += 2

        # Determine risk level
        if risk_score >= 6:
            return RiskLevel.EXTREME
        elif risk_score >= 4:
            return RiskLevel.HIGH
        elif risk_score >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def diagnose_data_freshness(self):
        """Diagnose data freshness issue"""
        try:
            # Load current price history
            prices, volumes = self.data_manager.load_price_history()
            
            if prices:
                # Get timestamps from file
                with open(self.data_manager.price_history_file, 'r') as f:
                    data = json.load(f)
                
                if data:
                    latest_timestamp = max(int(candle[0]) for candle in data)
                    earliest_timestamp = min(int(candle[0]) for candle in data)
                    
                    latest_date = datetime.fromtimestamp(latest_timestamp)
                    earliest_date = datetime.fromtimestamp(earliest_timestamp)
                    current_date = datetime.now()
                    
                    data_age_hours = (current_date - latest_date).total_seconds() / 3600
                    
                    print(f"\n📊 PRICE HISTORY DIAGNOSIS:")
                    print(f"   Earliest data: {earliest_date}")
                    print(f"   Latest data: {latest_date}")
                    print(f"   Current time: {current_date}")
                    print(f"   Data age: {data_age_hours:.1f} hours")
                    print(f"   Total candles: {len(data)}")
                    
                    if data_age_hours > 1:
                        print(f"   ⚠️  WARNING: Data is {data_age_hours:.1f} hours old!")
                        return False
                    else:
                        print(f"   ✅ Data is fresh")
                        return True
                
        except Exception as e:
            print(f"❌ Diagnosis failed: {e}")
            return False
    
    
    
    
    def _force_price_history_update(self):
        """Force update of price history before trading decisions"""
        try:
            logger.debug("Forcing price history update...")
            
            # Get fresh OHLC data
            recent_data = self.trade_executor.get_ohlc_data(
                pair="BTC/EUR",
                interval='15m',
                since=int(time.time()) - (7 * 24 * 3600),  # Last 7 days
                limit=672  # 7 days * 24 hours * 4 (15-min intervals)
            )
            
            if recent_data:
                added = self.data_manager.append_ohlc_data(recent_data)
                if added > 0:
                    logger.info(f"🔄 Price history updated with {added} fresh candles")
                    
                    # Reload price history
                    prices, volumes = self.data_manager.load_price_history()
                    
                    # Update cache
                    self.price_history.clear()
                    self.volume_history.clear()
                    
                    for price, volume in zip(prices[-1000:], volumes[-1000:]):
                        self.price_history.append(price)
                        self.volume_history.append(volume)
                    
                    logger.info(f"📊 Using {len(self.price_history)} data points for analysis")
                else:
                    logger.debug("No new data to add")
            else:
                logger.warning("Failed to fetch recent OHLC data")
                
        except Exception as e:
            logger.error(f"Failed to force price history update: {e}")
    
    
    def execute_unified_strategy(self):
        """Execute unified trading strategy with better error handling"""
        indicators = None  # Initialize to avoid UnboundLocalError
        
        try:
            logger.info("Executing unified trading strategy...")
            
            # Update pending orders first
            self.core_bot._update_pending_orders()
            
            # Check if we can trade
            if not self.core_bot._can_trade():
                logger.info("Trading conditions not met")
                return
            
            # Try to analyze market
            try:
                indicators = self.analyze_market()
            except ValueError as e:
                if "Insufficient price history" in str(e):
                    logger.warning(f"⚠️  Price history issue: {e}")
                    logger.info("🔄 Attempting to rebuild price history...")
                    
                    # Try to rebuild price history
                    success = self._emergency_rebuild_price_history()
                    if success:
                        logger.info("✅ Price history rebuilt, retrying analysis...")
                        indicators = self.analyze_market()
                    else:
                        logger.error("❌ Failed to rebuild price history")
                        return
                else:
                    raise  # Re-raise other ValueError types
            
            # Generate unified signal
            signal = self.generate_unified_signal(indicators)
            
            # Log decision
            self.core_bot._log_trading_decision(signal, indicators)
            
            # Execute trade if signal is actionable
            if signal.action != TradingAction.HOLD and signal.confidence > self.config.min_confidence_threshold:
                success = self.core_bot._execute_trade(signal)
                
                if success:
                    # Update performance tracking
                    self.total_trades += 1
                    
                    # Record trade in performance tracker
                    if self._performance_tracker:
                        try:
                            order_id = f"trade_{int(time.time())}"
                            fee = signal.price * signal.volume * 0.0025
                            
                            self._performance_tracker.record_trade(
                                order_id=order_id,
                                side=signal.action.value,
                                volume=signal.volume,
                                price=signal.price,
                                fee=fee,
                                timestamp=time.time()
                            )
                            
                            logger.info(f"📊 Trade recorded in performance tracker")
                            
                        except Exception as e:
                            logger.warning(f"Failed to record trade in performance tracker: {e}")
                    
                    # Update strategy states
                    for strategy in self.strategies.values():
                        strategy.update_state(signal, True)
                    
                    logger.info(f"✅ Trade executed: {signal.action.value.upper()}")
                else:
                    logger.warning("Trade execution failed")
            else:
                logger.info(f"Holding: {signal.reasoning[0] if signal.reasoning else 'Low confidence'}")
            
            # Update equity tracking (with safety check)
            if indicators is not None:
                if self._performance_tracker:
                    try:
                        btc_balance = self.trade_executor.get_total_btc_balance() or 0
                        eur_balance = self.trade_executor.get_available_balance("EUR") or 0
                        
                        self._performance_tracker.update_equity(btc_balance, eur_balance, indicators.current_price)
                        
                    except Exception as e:
                        logger.warning(f"Failed to update equity tracking: {e}")
            
            self.last_update = time.time()
            
        except Exception as e:
            logger.error(f"Unified strategy execution failed: {e}", exc_info=True)

    def _emergency_rebuild_price_history(self) -> bool:
        """Emergency rebuild of price history"""
        try:
            logger.info("🚨 Emergency: Rebuilding price history from scratch...")
            
            # Get comprehensive OHLC data
            fresh_data = self.trade_executor.get_ohlc_data(
                pair="BTC/EUR",
                interval='15m',
                since=int(time.time()) - (14 * 24 * 3600),  # Last 14 days
                limit=1344  # 14 days * 24 hours * 4 (15-min intervals)
            )
            
            if fresh_data:
                logger.info(f"📥 Fetched {len(fresh_data)} fresh OHLC candles")
                
                # Clear existing data
                with open(self.data_manager.price_history_file, 'w') as f:
                    json.dump([], f)
                
                # Add fresh data
                added = self.data_manager.append_ohlc_data(fresh_data)
                
                if added > 0:
                    # Reload price history
                    prices, volumes = self.data_manager.load_price_history()
                    
                    # Update cache
                    self.price_history.clear()
                    self.volume_history.clear()
                    
                    for price, volume in zip(prices[-1000:], volumes[-1000:]):
                        self.price_history.append(price)
                        self.volume_history.append(volume)
                    
                    logger.info(f"✅ Emergency rebuild successful: {len(self.price_history)} points")
                    return True
                else:
                    logger.error("❌ No data added during emergency rebuild")
                    return False
            else:
                logger.error("❌ Failed to fetch fresh OHLC data")
                return False
                
        except Exception as e:
            logger.error(f"Emergency rebuild failed: {e}")
            return False


    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status including all components"""
        status = self.core_bot.get_status()

        # Add unified bot specific information
        unified_status = {
            "unified_bot": {
                "session_start": self.session_start.isoformat(),
                "total_trades": self.total_trades,
                "successful_trades": self.successful_trades,
                "win_rate": self.successful_trades / max(1, self.total_trades),
                "total_pnl": self.total_pnl,
                "last_update": datetime.fromtimestamp(self.last_update).isoformat(),
            },
            "components": {
                "ml_engine_enabled": self.ml_engine is not None,
                "ml_engine_trained": (
                    self.ml_engine.is_trained if self.ml_engine else False
                ),
                "peak_detection_enabled": self.peak_system is not None,
                "active_strategies": list(self.strategies.keys()),
                "strategy_count": len(self.strategies),
            },
        }

        if self._performance_tracker:
            try:
                performance_report = self._performance_tracker.generate_performance_report()
                unified_status['performance'] = performance_report
            except Exception as e:
                logger.warning(f"Failed to generate performance report: {e}")
                unified_status['performance'] = {'error': str(e)}

        # Add ML performance if available
        if self.ml_engine:
            ml_summary = self.ml_engine.get_performance_summary()
            unified_status["ml_performance"] = ml_summary

        # Add strategy performance
        strategy_performance = {}
        for name, strategy in self.strategies.items():
            strategy_performance[name] = strategy.get_performance_metrics()

        unified_status["strategy_performance"] = strategy_performance

        # Merge with core status
        status.update(unified_status)

        return status

    def print_comprehensive_status(self):
        """Print comprehensive status report"""
        try:
            status = self.get_comprehensive_status()

            print(f"\n{'='*80}")
            print("🤖 UNIFIED BITCOIN TRADING BOT STATUS")
            print(f"{'='*80}")

            # Session info
            session = status["unified_bot"]
            uptime = (
                datetime.now() - datetime.fromisoformat(session["session_start"])
            ).total_seconds() / 3600
            print(f"⏱️  Uptime: {uptime:.1f} hours")
            print(f"📊 Total Trades: {session['total_trades']}")
            print(f"🎯 Win Rate: {session['win_rate']:.1%}")
            print(f"💰 Total P&L: €{session['total_pnl']:.2f}")

            # Balances
            balances = status["balances"]
            print(f"💎 BTC Balance: {balances['btc']:.8f}")
            print(f"💶 EUR Balance: €{balances['eur']:.2f}")
            print(f"🏦 Total Value: €{balances['total_value_eur']:.2f}")

            # Market info
            market = status["market"]
            print(f"📈 Current BTC Price: €{market['current_price']:.2f}")

            # Components status
            components = status["components"]
            print("\n🔧 COMPONENTS:")
            print(
                f"   🧠 ML Engine: {'✅ Trained' if components['ml_engine_trained'] else '❌ Not trained' if components['ml_engine_enabled'] else '⏸️  Disabled'}"
            )
            print(
                f"   📊 Peak Detection: {'✅ Active' if components['peak_detection_enabled'] else '⏸️  Disabled'}"
            )
            print(
                f"   🎯 Active Strategies: {components['strategy_count']} ({', '.join(components['active_strategies'])})"
            )

            # ML Performance
            if "ml_performance" in status:
                ml_perf = status["ml_performance"]
                print("\n🧠 ML PERFORMANCE:")
                print(f"   Accuracy: {ml_perf.get('accuracy', 0):.1%}")
                print(f"   Predictions: {ml_perf.get('prediction_count', 0)}")
                print(f"   Model: {ml_perf.get('model_type', 'Unknown')}")

            # Strategy Performance
            if status["strategy_performance"]:
                print("\n🎯 STRATEGY PERFORMANCE:")
                for name, perf in status["strategy_performance"].items():
                    win_rate = perf["win_rate"]
                    trades = perf["total_trades"]
                    print(
                        f"   {name.title()}: {win_rate:.1%} win rate ({trades} trades)"
                    )

            # Add onchain section
            if self.onchain_analyzer:
                try:
                    signals = self.onchain_analyzer.get_onchain_signals()
                    print(f"\n🔗 ONCHAIN METRICS:")
                    print(f"   📊 Network Volume: {signals.get('onchain_volume', 0):.0f} BTC")
                    print(f"   💸 Net Flow: {signals.get('netflow', 0):+.0f} BTC ({'📈 Accumulation' if signals.get('netflow', 0) < 0 else '📉 Distribution' if signals.get('netflow', 0) > 0 else 'Neutral'})")
                    print(f"   ⛽ Fee Rate: {signals.get('fee_rate', 0):.1f} sat/vB")
                    print(f"   🕰️ Old UTXOs: {signals.get('old_utxos', 0)} moving")
                except Exception as e:
                    print(f"   ❌ OnChain data error: {e}")

             # Add performance section
            if self._performance_tracker and 'performance' in status:
                perf = status['performance']
                print(f"\n📊 PERFORMANCE METRICS:")
                
                if 'error' not in perf:
                    print(f"   💰 Current Equity: {perf['equity']['current']}")
                    print(f"   📈 24h Return: {perf['returns']['24h']}")
                    print(f"   📅 7d Return: {perf['returns']['7d']}")
                    print(f"   🎯 Total Return: {perf['returns']['total']}")
                    print(f"   📊 Sharpe Ratio: {perf['risk_metrics']['sharpe_ratio']}")
                    print(f"   📉 Max Drawdown: {perf['risk_metrics']['max_drawdown']}")
                    print(f"   🎲 Win Rate: {perf['risk_metrics']['win_rate']}")
                    print(f"   💸 Total Fees: €{perf['trade_stats']['total_fees']:.2f}")
                else:
                    print(f"   ❌ Performance data error: {perf['error']}")

            print(f"{'='*80}\n")

        except Exception as e:
            logger.error(f"Failed to print status: {e}")
            print(f"Status error: {e}")

    def save_state(self):
        """Save current state to disk"""
        try:
            # Save ML model
            if self.ml_engine:
                self.ml_engine.save_model("./ml_model.pkl")

            # Save peak patterns
            if self.peak_system:
                # Peak system should have its own save method
                pass

            # Save unified bot state
            state = {
                "session_start": self.session_start.isoformat(),
                "total_trades": self.total_trades,
                "successful_trades": self.successful_trades,
                "total_pnl": self.total_pnl,
                "last_update": self.last_update,
            }

            with open("./unified_bot_state.json", "w") as f:
                json.dump(state, f, indent=2)

            logger.info("Unified bot state saved")

        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def load_state(self):
        """Load state from disk"""
        try:
            if os.path.exists("./unified_bot_state.json"):
                with open("./unified_bot_state.json", "r") as f:
                    state = json.load(f)

                self.session_start = datetime.fromisoformat(
                    state.get("session_start", datetime.now().isoformat())
                )
                self.total_trades = state.get("total_trades", 0)
                self.successful_trades = state.get("successful_trades", 0)
                self.total_pnl = state.get("total_pnl", 0.0)
                self.last_update = state.get("last_update", time.time())

                logger.info("Unified bot state loaded")

        except Exception as e:
            logger.error(f"Failed to load state: {e}")

    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down unified trading bot...")

        try:
            # Save state
            self.save_state()

            # Shutdown core bot
            self.core_bot.shutdown()

            self.running = False

            logger.info("Unified bot shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
