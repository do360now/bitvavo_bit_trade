# bitcoin_bot/analysis/ml_engine.py
"""
Machine Learning Engine for Bitcoin Trading Bot
Provides advanced ML capabilities for trade prediction and strategy optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import logging
import pickle
import warnings

warnings.filterwarnings("ignore")

# Scikit-learn imports
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import (
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    confusion_matrix,
)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Additional ML libraries
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Available model types"""

    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    NEURAL_NETWORK = "neural_network"
    SVM = "svm"
    ENSEMBLE = "ensemble"


class PredictionTask(Enum):
    """Prediction task types"""

    PRICE_DIRECTION = "price_direction"  # Up/Down/Neutral
    TRADE_SUCCESS = "trade_success"  # Success/Failure
    PRICE_REGRESSION = "price_regression"  # Actual price
    VOLATILITY = "volatility"  # Volatility prediction
    OPTIMAL_ACTION = "optimal_action"  # Buy/Sell/Hold


@dataclass
class MLConfig:
    """Machine Learning configuration"""

    # Model settings
    model_type: ModelType = ModelType.ENSEMBLE
    prediction_task: PredictionTask = PredictionTask.TRADE_SUCCESS

    # Feature engineering
    use_technical_features: bool = True
    use_market_features: bool = True
    use_sentiment_features: bool = True
    use_time_features: bool = True
    use_onchain_features: bool = False

    # Feature selection
    feature_selection_enabled: bool = True
    max_features: int = 30
    feature_selection_method: str = "mutual_info"

    # Scaling
    scaler_type: str = "robust"  # "standard", "robust", "minmax"

    # Training parameters
    test_size: float = 0.2
    validation_splits: int = 5
    retraining_interval: int = 100  # Retrain every N predictions
    min_training_samples: int = 500

    # Model specific parameters
    model_params: Dict[str, Any] = field(default_factory=dict)

    # Online learning
    online_learning_enabled: bool = True
    online_batch_size: int = 50

    # Performance thresholds
    min_accuracy_threshold: float = 0.55
    min_f1_threshold: float = 0.50

    # Ensemble settings
    ensemble_models: List[str] = field(
        default_factory=lambda: ["random_forest", "gradient_boosting", "neural_network"]
    )


@dataclass
class FeatureSet:
    """Container for feature data"""

    features: np.ndarray
    feature_names: List[str]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPerformance:
    """Model performance metrics"""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    training_time: float = 0.0
    prediction_time: float = 0.0
    sample_count: int = 0


class FeatureEngineer:
    """
    Feature engineering for ML models
    """

    def __init__(self, config: MLConfig):
        self.config = config
        self.feature_names = []
        self.scaler = self._create_scaler()
        self.is_fitted = False

    def _create_scaler(self):
        """Create appropriate scaler"""
        if self.config.scaler_type == "standard":
            return StandardScaler()
        elif self.config.scaler_type == "robust":
            return RobustScaler()
        elif self.config.scaler_type == "minmax":
            return MinMaxScaler()
        else:
            return StandardScaler()

    def extract_features(self, market_data: Dict[str, Any]) -> FeatureSet:
        """
        Extract features from market data

        Args:
            market_data: Dictionary containing market indicators and data

        Returns:
            FeatureSet with extracted features
        """
        features = []
        feature_names = []

        # Technical features
        if self.config.use_technical_features:
            tech_features, tech_names = self._extract_technical_features(market_data)
            features.extend(tech_features)
            feature_names.extend(tech_names)

        # Market structure features
        if self.config.use_market_features:
            market_features, market_names = self._extract_market_features(market_data)
            features.extend(market_features)
            feature_names.extend(market_names)

        # Sentiment features
        if self.config.use_sentiment_features:
            sent_features, sent_names = self._extract_sentiment_features(market_data)
            features.extend(sent_features)
            feature_names.extend(sent_names)

        # Time-based features
        if self.config.use_time_features:
            time_features, time_names = self._extract_time_features(market_data)
            features.extend(time_features)
            feature_names.extend(time_names)

        # On-chain features
        if self.config.use_onchain_features:
            chain_features, chain_names = self._extract_onchain_features(market_data)
            features.extend(chain_features)
            feature_names.extend(chain_names)

        # Store feature names
        self.feature_names = feature_names

        # Create feature array
        feature_array = np.array(features).reshape(1, -1)

        # Scale features if scaler is fitted
        if self.is_fitted:
            feature_array = self.scaler.transform(feature_array)

        return FeatureSet(
            features=feature_array,
            feature_names=feature_names,
            timestamp=datetime.now(),
            metadata={"market_data": market_data},
        )

    def _extract_technical_features(self, data: Dict) -> Tuple[List[float], List[str]]:
        """Extract technical indicator features"""
        features = []
        names = []

        # RSI
        rsi = data.get("rsi", 50)
        features.append(rsi / 100.0)  # Normalize to 0-1
        names.append("rsi_normalized")

        # RSI categories
        features.append(1.0 if rsi < 30 else 0.0)  # Oversold
        features.append(1.0 if rsi > 70 else 0.0)  # Overbought
        names.extend(["rsi_oversold", "rsi_overbought"])

        # MACD
        macd = data.get("macd", 0)
        signal = data.get("signal", 0)
        features.append(np.tanh(macd / 1000))  # Bounded
        features.append(np.tanh(signal / 1000))
        features.append(1.0 if macd > signal else 0.0)  # Bullish cross
        names.extend(["macd_normalized", "signal_normalized", "macd_bullish"])

        # Bollinger Bands
        current_price = data.get("current_price", 0)
        upper_band = data.get("upper_band", current_price)
        lower_band = data.get("lower_band", current_price)

        if upper_band > lower_band:
            bb_position = (current_price - lower_band) / (upper_band - lower_band)
        else:
            bb_position = 0.5

        features.append(bb_position)
        features.append(1.0 if current_price < lower_band else 0.0)  # Below lower
        features.append(1.0 if current_price > upper_band else 0.0)  # Above upper
        names.extend(["bb_position", "bb_below_lower", "bb_above_upper"])

        # Moving averages
        ma_short = data.get("ma_short", current_price)
        ma_long = data.get("ma_long", current_price)

        features.append((current_price - ma_short) / ma_short if ma_short > 0 else 0)
        features.append((current_price - ma_long) / ma_long if ma_long > 0 else 0)
        features.append(1.0 if ma_short > ma_long else 0.0)  # Golden cross
        names.extend(["price_to_ma_short", "price_to_ma_long", "golden_cross"])

        # VWAP
        vwap = data.get("vwap", current_price)
        features.append((current_price - vwap) / vwap if vwap > 0 else 0)
        names.append("price_to_vwap")

        return features, names

    def _extract_market_features(self, data: Dict) -> Tuple[List[float], List[str]]:
        """Extract market structure features"""
        features = []
        names = []

        # Volatility
        volatility = data.get("volatility", 0.02)
        features.append(min(1.0, volatility / 0.1))  # Normalize
        features.append(1.0 if volatility > 0.05 else 0.0)  # High volatility
        names.extend(["volatility_normalized", "high_volatility"])

        # Volume
        volume_ratio = data.get("volume_ratio", 1.0)
        features.append(min(2.0, volume_ratio) / 2.0)  # Normalize to 0-1
        features.append(1.0 if volume_ratio > 1.5 else 0.0)  # Volume surge
        names.extend(["volume_ratio", "volume_surge"])

        # Price momentum
        price_momentum = data.get("price_momentum", 0)
        features.append(np.tanh(price_momentum / 0.1))  # Bounded
        features.append(1.0 if price_momentum > 0.02 else 0.0)  # Strong momentum
        names.extend(["price_momentum", "strong_momentum"])

        # Market regime
        market_regime = data.get("market_regime", "ranging")
        features.append(1.0 if market_regime == "uptrend" else 0.0)
        features.append(1.0 if market_regime == "downtrend" else 0.0)
        features.append(1.0 if market_regime == "ranging" else 0.0)
        names.extend(["regime_uptrend", "regime_downtrend", "regime_ranging"])

        # Price levels
        prices = data.get("price_history", [])
        if len(prices) >= 20:
            current_price = prices[-1]
            min_price = min(prices[-20:])
            max_price = max(prices[-20:])

            if max_price > min_price:
                price_position = (current_price - min_price) / (max_price - min_price)
            else:
                price_position = 0.5

            features.append(price_position)
            features.append(1.0 if price_position > 0.8 else 0.0)  # Near high
            features.append(1.0 if price_position < 0.2 else 0.0)  # Near low
            names.extend(["price_position", "near_high", "near_low"])
        else:
            features.extend([0.5, 0.0, 0.0])
            names.extend(["price_position", "near_high", "near_low"])

        return features, names

    def _extract_sentiment_features(self, data: Dict) -> Tuple[List[float], List[str]]:
        """Extract sentiment and news features"""
        features = []
        names = []

        # Sentiment score
        sentiment = data.get("sentiment", 0)
        features.append((sentiment + 1) / 2)  # Convert -1,1 to 0,1
        features.append(1.0 if sentiment > 0.1 else 0.0)  # Positive
        features.append(1.0 if sentiment < -0.1 else 0.0)  # Negative
        names.extend(
            ["sentiment_normalized", "sentiment_positive", "sentiment_negative"]
        )

        # Risk-off probability
        risk_off_prob = data.get("risk_off_probability", 0)
        features.append(risk_off_prob)
        features.append(1.0 if risk_off_prob > 0.6 else 0.0)  # High risk
        features.append(1.0 if risk_off_prob > 0.8 else 0.0)  # Extreme risk
        names.extend(["risk_off_prob", "high_risk", "extreme_risk"])

        # News volume
        news_volume = data.get("news_volume", 0)
        features.append(min(1.0, news_volume / 50))  # Normalize
        names.append("news_volume_normalized")

        return features, names

    def _extract_time_features(self, data: Dict) -> Tuple[List[float], List[str]]:
        """Extract time-based features"""
        features = []
        names = []

        timestamp = data.get("timestamp", datetime.now())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        # Hour of day (cyclic encoding)
        hour = timestamp.hour
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        names.extend(["hour_sin", "hour_cos"])

        # Day of week (cyclic encoding)
        day = timestamp.weekday()
        features.append(np.sin(2 * np.pi * day / 7))
        features.append(np.cos(2 * np.pi * day / 7))
        names.extend(["day_sin", "day_cos"])

        # Month (cyclic encoding)
        month = timestamp.month
        features.append(np.sin(2 * np.pi * month / 12))
        features.append(np.cos(2 * np.pi * month / 12))
        names.extend(["month_sin", "month_cos"])

        # Trading session
        features.append(1.0 if 8 <= hour < 16 else 0.0)  # European session
        features.append(1.0 if 13 <= hour < 22 else 0.0)  # US session
        features.append(1.0 if 0 <= hour < 8 else 0.0)  # Asian session
        names.extend(["european_session", "us_session", "asian_session"])

        # Weekend
        features.append(1.0 if day >= 5 else 0.0)
        names.append("is_weekend")

        return features, names

    def _extract_onchain_features(self, data: Dict) -> Tuple[List[float], List[str]]:
        """Extract on-chain features"""
        features = []
        names = []

        # Network fees
        fee_rate = data.get("fee_rate", 10)
        features.append(min(1.0, fee_rate / 100))  # Normalize
        features.append(1.0 if fee_rate > 50 else 0.0)  # High fees
        names.extend(["fee_rate_normalized", "high_fees"])

        # Netflow
        netflow = data.get("netflow", 0)
        features.append(np.tanh(netflow / 10000))  # Bounded
        features.append(1.0 if netflow < -5000 else 0.0)  # Accumulation
        features.append(1.0 if netflow > 5000 else 0.0)  # Distribution
        names.extend(["netflow_normalized", "accumulation", "distribution"])

        # Old UTXOs
        old_utxos = data.get("old_utxos", 0)
        features.append(min(1.0, old_utxos / 100))  # Normalize
        names.append("old_utxos_normalized")

        return features, names

    def fit(self, feature_matrix: np.ndarray):
        """Fit the scaler on training data"""
        self.scaler.fit(feature_matrix)
        self.is_fitted = True

    def transform(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler"""
        if not self.is_fitted:
            logger.warning("Scaler not fitted, fitting now")
            self.fit(feature_matrix)
        return self.scaler.transform(feature_matrix)


class ModelFactory:
    """
    Factory for creating ML models
    """

    @staticmethod
    def create_model(model_type: ModelType, config: MLConfig, task: PredictionTask):
        """Create a model based on type and task"""

        is_classification = task in [
            PredictionTask.PRICE_DIRECTION,
            PredictionTask.TRADE_SUCCESS,
            PredictionTask.OPTIMAL_ACTION,
        ]

        if model_type == ModelType.RANDOM_FOREST:
            if is_classification:
                return RandomForestClassifier(
                    n_estimators=config.model_params.get("n_estimators", 100),
                    max_depth=config.model_params.get("max_depth", 10),
                    min_samples_split=config.model_params.get("min_samples_split", 5),
                    min_samples_leaf=config.model_params.get("min_samples_leaf", 2),
                    random_state=42,
                    n_jobs=-1,
                )
            else:
                return RandomForestRegressor(
                    n_estimators=config.model_params.get("n_estimators", 100),
                    max_depth=config.model_params.get("max_depth", 10),
                    random_state=42,
                    n_jobs=-1,
                )

        elif model_type == ModelType.GRADIENT_BOOSTING:
            if is_classification:
                return GradientBoostingClassifier(
                    n_estimators=config.model_params.get("n_estimators", 100),
                    learning_rate=config.model_params.get("learning_rate", 0.1),
                    max_depth=config.model_params.get("max_depth", 5),
                    random_state=42,
                )
            else:
                return GradientBoostingRegressor(
                    n_estimators=config.model_params.get("n_estimators", 100),
                    learning_rate=config.model_params.get("learning_rate", 0.1),
                    max_depth=config.model_params.get("max_depth", 5),
                    random_state=42,
                )

        elif model_type == ModelType.XGBOOST and XGBOOST_AVAILABLE:
            if is_classification:
                return xgb.XGBClassifier(
                    n_estimators=config.model_params.get("n_estimators", 100),
                    learning_rate=config.model_params.get("learning_rate", 0.1),
                    max_depth=config.model_params.get("max_depth", 6),
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric="logloss",
                )
            else:
                return xgb.XGBRegressor(
                    n_estimators=config.model_params.get("n_estimators", 100),
                    learning_rate=config.model_params.get("learning_rate", 0.1),
                    max_depth=config.model_params.get("max_depth", 6),
                    random_state=42,
                )

        elif model_type == ModelType.LIGHTGBM and LIGHTGBM_AVAILABLE:
            if is_classification:
                return lgb.LGBMClassifier(
                    n_estimators=config.model_params.get("n_estimators", 100),
                    learning_rate=config.model_params.get("learning_rate", 0.1),
                    max_depth=config.model_params.get("max_depth", -1),
                    random_state=42,
                    verbose=-1,
                )
            else:
                return lgb.LGBMRegressor(
                    n_estimators=config.model_params.get("n_estimators", 100),
                    learning_rate=config.model_params.get("learning_rate", 0.1),
                    max_depth=config.model_params.get("max_depth", -1),
                    random_state=42,
                    verbose=-1,
                )

        elif model_type == ModelType.NEURAL_NETWORK:
            if is_classification:
                return MLPClassifier(
                    hidden_layer_sizes=config.model_params.get(
                        "hidden_layers", (100, 50)
                    ),
                    activation=config.model_params.get("activation", "relu"),
                    learning_rate_init=config.model_params.get("learning_rate", 0.001),
                    max_iter=config.model_params.get("max_iter", 500),
                    early_stopping=True,
                    random_state=42,
                )
            else:
                return MLPRegressor(
                    hidden_layer_sizes=config.model_params.get(
                        "hidden_layers", (100, 50)
                    ),
                    activation=config.model_params.get("activation", "relu"),
                    learning_rate_init=config.model_params.get("learning_rate", 0.001),
                    max_iter=config.model_params.get("max_iter", 500),
                    early_stopping=True,
                    random_state=42,
                )

        elif model_type == ModelType.SVM:
            if is_classification:
                return SVC(
                    kernel=config.model_params.get("kernel", "rbf"),
                    C=config.model_params.get("C", 1.0),
                    gamma=config.model_params.get("gamma", "scale"),
                    probability=True,
                    random_state=42,
                )
            else:
                return SVR(
                    kernel=config.model_params.get("kernel", "rbf"),
                    C=config.model_params.get("C", 1.0),
                    gamma=config.model_params.get("gamma", "scale"),
                )

        else:
            # Default to Random Forest
            if is_classification:
                return RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                return RandomForestRegressor(n_estimators=100, random_state=42)


class MLEngine:
    """
    Main Machine Learning Engine
    """

    def __init__(self, config: Optional[MLConfig] = None):
        """
        Initialize ML Engine

        Args:
            config: ML configuration
        """
        self.config = config or MLConfig()
        self.feature_engineer = FeatureEngineer(self.config)

        # Models
        self.model = None
        self.ensemble_models = {}

        # Training data
        self.training_features = []
        self.training_labels = []

        # Performance tracking
        self.performance_history = []
        self.prediction_count = 0
        self.last_retrain_count = 0

        # Model state
        self.is_trained = False
        self.model_file = "ml_model.pkl"

        # Feature importance
        self.feature_importance = {}

        # Online learning buffer
        self.online_buffer_X = []
        self.online_buffer_y = []

        # Initialize model
        self._initialize_model()

        logger.info(f"ML Engine initialized with {self.config.model_type.value} model")

    def _initialize_model(self):
        """Initialize the ML model"""
        if self.config.model_type == ModelType.ENSEMBLE:
            # Create ensemble of models
            for model_name in self.config.ensemble_models:
                model_type = ModelType(model_name)
                self.ensemble_models[model_name] = ModelFactory.create_model(
                    model_type, self.config, self.config.prediction_task
                )

            # Create voting classifier/regressor
            is_classification = self.config.prediction_task in [
                PredictionTask.PRICE_DIRECTION,
                PredictionTask.TRADE_SUCCESS,
                PredictionTask.OPTIMAL_ACTION,
            ]

            if is_classification:
                self.model = VotingClassifier(
                    estimators=list(self.ensemble_models.items()), voting="soft"
                )
            else:
                # For regression, we'll average predictions manually
                self.model = self.ensemble_models
        else:
            # Single model
            self.model = ModelFactory.create_model(
                self.config.model_type, self.config, self.config.prediction_task
            )

    def prepare_training_data(
        self, historical_data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from historical trades

        Args:
            historical_data: DataFrame with historical trade data

        Returns:
            Tuple of (features, labels)
        """
        features_list = []
        labels_list = []

        for i in range(len(historical_data) - 1):
            row = historical_data.iloc[i]
            next_row = historical_data.iloc[i + 1]

            # Create market data dictionary
            market_data = {
                "rsi": row.get("rsi", 50),
                "macd": row.get("macd", 0),
                "signal": row.get("signal", 0),
                "current_price": row.get("price", 0),
                "upper_band": row.get("upper_band", row.get("price", 0)),
                "lower_band": row.get("lower_band", row.get("price", 0)),
                "ma_short": row.get("ma_short", row.get("price", 0)),
                "ma_long": row.get("ma_long", row.get("price", 0)),
                "vwap": row.get("vwap", row.get("price", 0)),
                "volatility": row.get("volatility", 0.02),
                "sentiment": row.get("sentiment", 0),
                "timestamp": row.get("timestamp", datetime.now()),
            }

            # Extract features
            feature_set = self.feature_engineer.extract_features(market_data)
            features_list.append(feature_set.features[0])

            # Create label based on task
            label = self._create_label(row, next_row, self.config.prediction_task)
            if label is not None:
                labels_list.append(label)
            else:
                features_list.pop()  # Remove features if no valid label

        if not features_list:
            return np.array([]), np.array([])

        return np.array(features_list), np.array(labels_list)

    def _create_label(
        self, current_row: pd.Series, next_row: pd.Series, task: PredictionTask
    ):
        """Create label based on prediction task"""
        current_price = current_row.get("price", 0)
        next_price = next_row.get("price", 0)

        if task == PredictionTask.PRICE_DIRECTION:
            # 0: Down, 1: Neutral, 2: Up
            threshold = 0.002  # 0.2% threshold
            if next_price > current_price * (1 + threshold):
                return 2  # Up
            elif next_price < current_price * (1 - threshold):
                return 0  # Down
            else:
                return 1  # Neutral

        elif task == PredictionTask.TRADE_SUCCESS:
            # For trades only
            if current_row.get("buy_decision") == "True":
                # Buy trade - successful if price went up
                return 1 if next_price > current_price * 1.002 else 0
            elif current_row.get("sell_decision") == "True":
                # Sell trade - successful if price went down
                return 1 if next_price < current_price * 0.998 else 0
            else:
                return None  # No trade

        elif task == PredictionTask.PRICE_REGRESSION:
            # Return actual price change
            return (next_price - current_price) / current_price

        elif task == PredictionTask.VOLATILITY:
            # Return volatility measure
            return abs(next_price - current_price) / current_price

        elif task == PredictionTask.OPTIMAL_ACTION:
            # 0: Hold, 1: Buy, 2: Sell
            threshold = 0.005  # 0.5% threshold for action
            if next_price > current_price * (1 + threshold):
                return 1  # Should have bought
            elif next_price < current_price * (1 - threshold):
                return 2  # Should have sold
            else:
                return 0  # Should have held

        return None

    def train(self, training_data: Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]):
        """
        Train the ML model

        Args:
            training_data: Either DataFrame or tuple of (features, labels)
        """
        start_time = datetime.now()

        # Prepare data
        if isinstance(training_data, pd.DataFrame):
            X, y = self.prepare_training_data(training_data)
        else:
            X, y = training_data

        if len(X) < self.config.min_training_samples:
            logger.warning(
                f"Insufficient training samples: {len(X)} < {self.config.min_training_samples}"
            )
            return

        # Fit feature scaler
        self.feature_engineer.fit(X)
        X_scaled = self.feature_engineer.transform(X)

        # Feature selection if enabled
        if self.config.feature_selection_enabled:
            X_scaled = self._select_features(X_scaled, y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=self.config.test_size, random_state=42
        )

        # Train model
        logger.info(f"Training model on {len(X_train)} samples...")

        if self.config.model_type == ModelType.ENSEMBLE and isinstance(
            self.model, dict
        ):
            # Train ensemble models separately
            for name, model in self.ensemble_models.items():
                model.fit(X_train, y_train)
                logger.info(f"Trained {name} model")
        else:
            self.model.fit(X_train, y_train)

        # Evaluate performance
        performance = self._evaluate_model(X_test, y_test)

        # Calculate feature importance
        self._calculate_feature_importance(X_train, y_train)

        # Store training data for online learning
        self.training_features = X_scaled
        self.training_labels = y

        # Update state
        self.is_trained = True
        training_time = (datetime.now() - start_time).total_seconds()
        performance.training_time = training_time
        performance.sample_count = len(X)

        self.performance_history.append(performance)

        logger.info(
            f"Model trained in {training_time:.2f}s - Accuracy: {performance.accuracy:.3f}, F1: {performance.f1_score:.3f}"
        )

        # Save model
        self.save_model()

    def _select_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Select best features"""
        if self.config.feature_selection_method == "mutual_info":
            selector = SelectKBest(
                mutual_info_classif, k=min(self.config.max_features, X.shape[1])
            )
        elif self.config.feature_selection_method == "f_classif":
            selector = SelectKBest(
                f_classif, k=min(self.config.max_features, X.shape[1])
            )
        else:
            return X

        X_selected = selector.fit_transform(X, y)

        # Store selected feature indices
        selected_features = selector.get_support(indices=True)
        selected_names = [
            self.feature_engineer.feature_names[i] for i in selected_features
        ]
        logger.info(
            f"Selected {len(selected_names)} features: {selected_names[:10]}..."
        )

        return X_selected

    def _evaluate_model(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> ModelPerformance:
        """Evaluate model performance"""
        is_classification = self.config.prediction_task in [
            PredictionTask.PRICE_DIRECTION,
            PredictionTask.TRADE_SUCCESS,
            PredictionTask.OPTIMAL_ACTION,
        ]

        if self.config.model_type == ModelType.ENSEMBLE and isinstance(
            self.model, dict
        ):
            # Ensemble prediction
            predictions = self._ensemble_predict(X_test)
        else:
            predictions = self.model.predict(X_test)

        if is_classification:
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(
                y_test, predictions, average="weighted", zero_division=0
            )
            recall = recall_score(
                y_test, predictions, average="weighted", zero_division=0
            )
            f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

            # ROC AUC for binary classification
            if len(np.unique(y_test)) == 2:
                if hasattr(self.model, "predict_proba"):
                    y_proba = self.model.predict_proba(X_test)[:, 1]
                    roc_auc = roc_auc_score(y_test, y_proba)
                else:
                    roc_auc = None
            else:
                roc_auc = None

            cm = confusion_matrix(y_test, predictions)

        else:
            # Regression metrics
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)

            # Convert to pseudo-accuracy for consistency
            accuracy = max(0, 1 - mae)  # Simple accuracy proxy
            precision = accuracy
            recall = accuracy
            f1 = accuracy
            roc_auc = None
            cm = None

        return ModelPerformance(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            confusion_matrix=cm,
        )

    def _calculate_feature_importance(self, X: np.ndarray, y: np.ndarray):
        """Calculate feature importance"""
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            feature_names = self.feature_engineer.feature_names[: len(importances)]

            self.feature_importance = dict(zip(feature_names, importances))

            # Sort and log top features
            sorted_features = sorted(
                self.feature_importance.items(), key=lambda x: x[1], reverse=True
            )
            logger.info("Top 10 important features:")
            for name, importance in sorted_features[:10]:
                logger.info(f"  {name}: {importance:.4f}")

    def predict(self, market_data: Dict[str, Any]) -> Tuple[Any, float]:
        """
        Make prediction on new market data

        Args:
            market_data: Current market indicators

        Returns:
            Tuple of (prediction, confidence)
        """
        if not self.is_trained:
            logger.warning("Model not trained, returning default prediction")
            return self._default_prediction()

        # Extract features
        feature_set = self.feature_engineer.extract_features(market_data)
        X = feature_set.features

        # Scale features
        X_scaled = self.feature_engineer.transform(X)

        # Make prediction
        start_time = datetime.now()

        if self.config.model_type == ModelType.ENSEMBLE and isinstance(
            self.model, dict
        ):
            prediction = self._ensemble_predict(X_scaled)[0]
            confidence = self._ensemble_confidence(X_scaled)[0]
        else:
            prediction = self.model.predict(X_scaled)[0]

            # Get confidence (probability for classification)
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(X_scaled)[0]
                confidence = np.max(probabilities)
            else:
                confidence = 0.5  # Default confidence

        prediction_time = (datetime.now() - start_time).total_seconds()

        # Update prediction count
        self.prediction_count += 1

        # Check if retraining needed
        if (
            self.config.online_learning_enabled
            and self.prediction_count - self.last_retrain_count
            >= self.config.retraining_interval
        ):
            self._trigger_retrain()

        logger.debug(
            f"Prediction: {prediction}, Confidence: {confidence:.3f}, Time: {prediction_time:.4f}s"
        )

        return prediction, confidence

    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble prediction"""
        predictions = []

        for name, model in self.ensemble_models.items():
            pred = model.predict(X)
            predictions.append(pred)

        # Majority voting for classification, average for regression
        predictions = np.array(predictions)

        if self.config.prediction_task in [
            PredictionTask.PRICE_DIRECTION,
            PredictionTask.TRADE_SUCCESS,
            PredictionTask.OPTIMAL_ACTION,
        ]:
            # Majority voting
            from scipy.stats import mode

            final_predictions = mode(predictions, axis=0)[0][0]
        else:
            # Average
            final_predictions = np.mean(predictions, axis=0)

        return final_predictions

    def _ensemble_confidence(self, X: np.ndarray) -> np.ndarray:
        """Calculate ensemble confidence"""
        confidences = []

        for name, model in self.ensemble_models.items():
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                confidence = np.max(proba, axis=1)
                confidences.append(confidence)

        if confidences:
            return np.mean(confidences, axis=0)
        else:
            return np.array([0.5] * X.shape[0])

    def _default_prediction(self) -> Tuple[Any, float]:
        """Return default prediction when model not trained"""
        if self.config.prediction_task == PredictionTask.PRICE_DIRECTION:
            return 1, 0.33  # Neutral
        elif self.config.prediction_task == PredictionTask.TRADE_SUCCESS:
            return 0, 0.5  # Don't trade
        elif self.config.prediction_task == PredictionTask.OPTIMAL_ACTION:
            return 0, 0.33  # Hold
        else:
            return 0.0, 0.0

    def update_online(self, market_data: Dict[str, Any], actual_outcome: Any):
        """
        Update model with new data point (online learning)

        Args:
            market_data: Market data used for prediction
            actual_outcome: Actual outcome that occurred
        """
        if not self.config.online_learning_enabled:
            return

        # Extract features
        feature_set = self.feature_engineer.extract_features(market_data)
        X = feature_set.features[0]

        # Add to buffer
        self.online_buffer_X.append(X)
        self.online_buffer_y.append(actual_outcome)

        # Trigger batch update if buffer is full
        if len(self.online_buffer_X) >= self.config.online_batch_size:
            self._batch_update()

    def _batch_update(self):
        """Perform batch update with buffered data"""
        if not self.online_buffer_X:
            return

        X_new = np.array(self.online_buffer_X)
        y_new = np.array(self.online_buffer_y)

        # Scale features
        X_scaled = self.feature_engineer.transform(X_new)

        # Partial fit if supported
        if hasattr(self.model, "partial_fit"):
            self.model.partial_fit(X_scaled, y_new)
            logger.info(f"Performed online update with {len(X_new)} samples")
        else:
            # Add to training data for next retrain
            self.training_features = np.vstack([self.training_features, X_scaled])
            self.training_labels = np.concatenate([self.training_labels, y_new])

        # Clear buffer
        self.online_buffer_X = []
        self.online_buffer_y = []

    def _trigger_retrain(self):
        """Trigger model retraining"""
        if len(self.training_features) < self.config.min_training_samples:
            return

        logger.info("Triggering model retrain...")
        self.train((self.training_features, self.training_labels))
        self.last_retrain_count = self.prediction_count

    def save_model(self, filepath: Optional[str] = None):
        """Save model to file"""
        filepath = filepath or self.model_file

        try:
            model_data = {
                "model": self.model,
                "ensemble_models": self.ensemble_models,
                "feature_engineer": self.feature_engineer,
                "config": self.config,
                "is_trained": self.is_trained,
                "feature_importance": self.feature_importance,
                "performance_history": self.performance_history[
                    -10:
                ],  # Last 10 performances
            }

            with open(filepath, "wb") as f:
                pickle.dump(model_data, f)

            logger.info(f"Model saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def load_model(self, filepath: Optional[str] = None):
        """Load model from file"""
        filepath = filepath or self.model_file

        try:
            with open(filepath, "rb") as f:
                model_data = pickle.load(f)

            self.model = model_data["model"]
            self.ensemble_models = model_data.get("ensemble_models", {})
            self.feature_engineer = model_data["feature_engineer"]
            self.config = model_data["config"]
            self.is_trained = model_data["is_trained"]
            self.feature_importance = model_data.get("feature_importance", {})
            self.performance_history = model_data.get("performance_history", [])

            logger.info(f"Model loaded from {filepath}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.performance_history:
            return {}

        recent_performance = self.performance_history[-1]

        return {
            "accuracy": recent_performance.accuracy,
            "precision": recent_performance.precision,
            "recall": recent_performance.recall,
            "f1_score": recent_performance.f1_score,
            "roc_auc": recent_performance.roc_auc,
            "training_time": recent_performance.training_time,
            "sample_count": recent_performance.sample_count,
            "prediction_count": self.prediction_count,
            "model_type": self.config.model_type.value,
            "is_trained": self.is_trained,
            "top_features": (
                list(self.feature_importance.items())[:5]
                if self.feature_importance
                else []
            ),
        }


# Example usage
if __name__ == "__main__":
    # Create ML engine with custom config
    config = MLConfig(
        model_type=ModelType.ENSEMBLE,
        prediction_task=PredictionTask.TRADE_SUCCESS,
        use_technical_features=True,
        use_market_features=True,
        use_sentiment_features=True,
        online_learning_enabled=True,
    )

    ml_engine = MLEngine(config)

    # Load sample data
    sample_data = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2024-01-01", periods=1000, freq="15min"),
            "price": np.random.randn(1000).cumsum() + 50000,
            "rsi": np.random.uniform(20, 80, 1000),
            "macd": np.random.randn(1000) * 100,
            "signal": np.random.randn(1000) * 100,
            "sentiment": np.random.uniform(-1, 1, 1000),
            "buy_decision": np.random.choice(["True", "False"], 1000),
            "sell_decision": np.random.choice(["True", "False"], 1000),
        }
    )

    # Train model
    ml_engine.train(sample_data)

    # Make prediction
    market_data = {
        "rsi": 45,
        "macd": 50,
        "signal": 30,
        "current_price": 50000,
        "sentiment": 0.1,
        "timestamp": datetime.now(),
    }

    prediction, confidence = ml_engine.predict(market_data)
    print(f"Prediction: {prediction}, Confidence: {confidence:.3f}")

    # Get performance summary
    summary = ml_engine.get_performance_summary()
    print(f"Performance Summary: {summary}")
