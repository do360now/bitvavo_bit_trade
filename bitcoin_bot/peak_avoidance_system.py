import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime
from scipy.signal import find_peaks
from scipy.stats import zscore
import json
import os
from dataclasses import dataclass


@dataclass
class PeakPattern:
    """Represents a detected peak pattern"""

    peak_price: float
    peak_time: datetime
    drop_percentage: float
    time_to_drop: int  # minutes
    volume_profile: List[float]
    leading_indicators: Dict[str, float]  # RSI, MACD, etc at peak
    pattern_type: str  # 'local_peak', 'major_peak', 'blow_off_top'


class PeakAvoidanceSystem:
    """
    Advanced system to detect and avoid buying at peaks before sharp drops.
    Learns from historical patterns to predict future peak behavior.
    """

    def __init__(self, lookback_days: int = 90):
        self.lookback_days = lookback_days
        self.historical_peaks = []
        self.pattern_database = []
        self.peak_indicators = {}
        self.pattern_file = "peak_patterns.json"
        self._load_historical_patterns()

    def _load_historical_patterns(self):
        """Load previously identified peak patterns"""
        try:
            if os.path.exists(self.pattern_file):
                with open(self.pattern_file, "r") as f:
                    data = json.load(f)
                    self.pattern_database = data.get("patterns", [])
                    self.peak_indicators = data.get("indicators", {})
                print(
                    f"✅ Loaded {len(self.pattern_database)} historical peak patterns"
                )
        except Exception as e:
            print(f"⚠️ Could not load peak patterns: {e}")

    def _save_patterns(self):
        """Save detected patterns for future learning"""
        try:
            data = {
                "patterns": self.pattern_database[-500:],  # Keep last 500 patterns
                "indicators": self.peak_indicators,
                "last_updated": datetime.now().isoformat(),
            }
            with open(self.pattern_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ Could not save peak patterns: {e}")

    def analyze_price_history(
        self,
        prices: List[float],
        volumes: List[float],
        timestamps: List[datetime],
        indicators: List[Dict],
    ) -> List[PeakPattern]:
        """
        Analyze historical price data to identify peak patterns and their outcomes.
        This builds the learning database for future peak detection.
        """
        if len(prices) < 50:
            return []

        prices_array = np.array(prices)
        volumes_array = np.array(volumes)

        # Detect peaks using multiple methods
        peaks = self._detect_peaks_multi_method(prices_array)

        detected_patterns = []

        for peak_idx in peaks:
            if peak_idx < 10 or peak_idx > len(prices) - 20:  # Need context around peak
                continue

            # Analyze what happened after this peak
            peak_price = prices[peak_idx]
            peak_time = (
                timestamps[peak_idx] if peak_idx < len(timestamps) else datetime.now()
            )

            # Look for significant drops after peak (>3%)
            future_prices = prices[peak_idx : peak_idx + 20]  # Next 20 periods
            if not future_prices:
                continue

            min_future_price = min(future_prices)
            drop_percentage = (peak_price - min_future_price) / peak_price

            # Only consider significant drops
            if drop_percentage > 0.03:  # 3% or more drop
                # Find when the drop occurred
                min_idx = future_prices.index(min_future_price)
                time_to_drop = min_idx * 15  # Assuming 15-minute intervals

                # Get volume profile around peak
                volume_start = max(0, peak_idx - 5)
                volume_end = min(len(volumes), peak_idx + 10)
                volume_profile = volumes[volume_start:volume_end]

                # Get leading indicators at peak
                leading_indicators = {}
                if peak_idx < len(indicators):
                    indicator_data = indicators[peak_idx]
                    leading_indicators = {
                        "rsi": indicator_data.get("rsi", 50),
                        "macd": indicator_data.get("macd", 0),
                        "volume_ratio": (
                            volumes[peak_idx]
                            / np.mean(volumes[max(0, peak_idx - 10) : peak_idx])
                            if peak_idx > 10
                            else 1.0
                        ),
                        "price_momentum": (
                            (prices[peak_idx] - prices[peak_idx - 5])
                            / prices[peak_idx - 5]
                            if peak_idx > 5
                            else 0
                        ),
                    }

                # Classify peak type
                pattern_type = self._classify_peak_type(
                    drop_percentage, time_to_drop, leading_indicators
                )

                pattern = PeakPattern(
                    peak_price=peak_price,
                    peak_time=peak_time,
                    drop_percentage=drop_percentage,
                    time_to_drop=time_to_drop,
                    volume_profile=volume_profile,
                    leading_indicators=leading_indicators,
                    pattern_type=pattern_type,
                )

                detected_patterns.append(pattern)

        # Add to our learning database
        self.pattern_database.extend(
            [self._pattern_to_dict(p) for p in detected_patterns]
        )
        self._update_peak_indicators()
        self._save_patterns()

        print(f"🔍 Detected {len(detected_patterns)} new peak patterns")
        return detected_patterns

    def _detect_peaks_multi_method(self, prices: np.array) -> List[int]:
        """Use multiple methods to detect peaks for robustness"""
        peaks_all = []

        # Method 1: Simple peak detection with prominence
        peaks1, properties = find_peaks(
            prices, prominence=prices.std() * 0.5, distance=5
        )
        peaks_all.extend(peaks1)

        # Method 2: Z-score based peak detection
        z_scores = np.abs(zscore(prices))
        peaks2 = np.where(z_scores > 2.0)[0]  # 2 standard deviations
        peaks_all.extend(peaks2)

        # Method 3: Local maxima in rolling windows
        window = 10
        for i in range(window, len(prices) - window):
            window_prices = prices[i - window : i + window + 1]
            if prices[i] == max(window_prices):
                peaks_all.append(i)

        # Remove duplicates and sort
        peaks_unique = sorted(list(set(peaks_all)))

        # Filter peaks that are too close together
        filtered_peaks = []
        for peak in peaks_unique:
            if (
                not filtered_peaks or peak - filtered_peaks[-1] > 5
            ):  # At least 5 periods apart
                filtered_peaks.append(peak)

        return filtered_peaks

    def _classify_peak_type(
        self, drop_percentage: float, time_to_drop: int, indicators: Dict[str, float]
    ) -> str:
        """Classify the type of peak based on characteristics"""
        rsi = indicators.get("rsi", 50)
        volume_ratio = indicators.get("volume_ratio", 1.0)
        momentum = indicators.get("price_momentum", 0)

        # Blow-off top: Large drop (>8%), high RSI (>75), high volume, fast drop
        if (
            drop_percentage > 0.08
            and rsi > 75
            and volume_ratio > 2.0
            and time_to_drop < 60
        ):
            return "blow_off_top"

        # Major peak: Significant drop (>5%), overbought conditions
        elif drop_percentage > 0.05 and rsi > 65:
            return "major_peak"

        # Local peak: Smaller drop but still significant
        else:
            return "local_peak"

    def _pattern_to_dict(self, pattern: PeakPattern) -> Dict:
        """Convert PeakPattern to dictionary for JSON storage"""
        return {
            "peak_price": pattern.peak_price,
            "peak_time": (
                pattern.peak_time.isoformat()
                if isinstance(pattern.peak_time, datetime)
                else str(pattern.peak_time)
            ),
            "drop_percentage": pattern.drop_percentage,
            "time_to_drop": pattern.time_to_drop,
            "volume_profile": pattern.volume_profile,
            "leading_indicators": pattern.leading_indicators,
            "pattern_type": pattern.pattern_type,
        }

    def _update_peak_indicators(self):
        """Update statistical indicators about peak patterns"""
        if not self.pattern_database:
            return

        # Calculate statistics for different peak types
        by_type = {}
        for pattern in self.pattern_database:
            ptype = pattern.get("pattern_type", "unknown")
            if ptype not in by_type:
                by_type[ptype] = []
            by_type[ptype].append(pattern)

        self.peak_indicators = {}
        for ptype, patterns in by_type.items():
            self.peak_indicators[ptype] = {
                "count": len(patterns),
                "avg_drop": np.mean([p["drop_percentage"] for p in patterns]),
                "avg_time_to_drop": np.mean([p["time_to_drop"] for p in patterns]),
                "avg_rsi_at_peak": np.mean(
                    [p["leading_indicators"].get("rsi", 50) for p in patterns]
                ),
                "avg_volume_ratio": np.mean(
                    [p["leading_indicators"].get("volume_ratio", 1.0) for p in patterns]
                ),
            }

    def predict_peak_probability(
        self,
        current_price: float,
        current_indicators: Dict[str, float],
        recent_prices: List[float],
        recent_volumes: List[float],
    ) -> Dict[str, float]:
        """
        Predict the probability that current conditions represent a peak before a drop.
        Returns probabilities for different peak types and recommended actions.
        """
        if len(recent_prices) < 20 or not self.pattern_database:
            return {
                "peak_probability": 0.3,
                "confidence": 0.0,
                "recommended_action": "neutral",
            }

        # Calculate current market characteristics
        current_rsi = current_indicators.get("rsi", 50)
        current_macd = current_indicators.get("macd", 0)
        current_volume_ratio = (
            (recent_volumes[-1] / np.mean(recent_volumes[-10:]))
            if len(recent_volumes) >= 10
            else 1.0
        )
        current_momentum = (
            (current_price - recent_prices[-5]) / recent_prices[-5]
            if len(recent_prices) >= 5
            else 0
        )

        # Check if we're at a local high
        is_local_high = (
            current_price >= max(recent_prices[-10:])
            if len(recent_prices) >= 10
            else False
        )

        current_characteristics = {
            "rsi": current_rsi,
            "macd": current_macd,
            "volume_ratio": current_volume_ratio,
            "momentum": current_momentum,
            "is_local_high": is_local_high,
        }

        # Compare against historical peak patterns
        probabilities = {}
        for pattern_type in ["blow_off_top", "major_peak", "local_peak"]:
            prob = self._calculate_pattern_similarity(
                current_characteristics, pattern_type
            )
            probabilities[pattern_type] = prob

        # Overall peak probability (weighted by severity)
        overall_probability = (
            probabilities.get("blow_off_top", 0) * 0.5
            + probabilities.get("major_peak", 0) * 0.3
            + probabilities.get("local_peak", 0) * 0.2
        )

        # Calculate confidence based on number of similar patterns
        confidence = min(
            1.0, len(self.pattern_database) / 50
        )  # Full confidence with 50+ patterns

        # Generate recommendation
        recommendation = self._generate_peak_recommendation(
            overall_probability, probabilities, current_characteristics
        )

        return {
            "peak_probability": overall_probability,
            "pattern_probabilities": probabilities,
            "confidence": confidence,
            "recommended_action": recommendation["action"],
            "reasoning": recommendation["reasoning"],
            "risk_level": recommendation["risk_level"],
            "current_characteristics": current_characteristics,
        }

    def _calculate_pattern_similarity(
        self, current_characteristics: Dict, pattern_type: str
    ) -> float:
        """Calculate similarity between current conditions and historical peak patterns"""
        if pattern_type not in self.peak_indicators:
            return 0.0

        type_patterns = [
            p for p in self.pattern_database if p.get("pattern_type") == pattern_type
        ]
        if not type_patterns:
            return 0.0

        similarities = []
        for pattern in type_patterns:
            pattern_indicators = pattern.get("leading_indicators", {})

            # Calculate similarity scores for each indicator
            rsi_similarity = (
                1.0
                - abs(
                    current_characteristics["rsi"] - pattern_indicators.get("rsi", 50)
                )
                / 100.0
            )
            volume_similarity = 1.0 - min(
                1.0,
                abs(
                    current_characteristics["volume_ratio"]
                    - pattern_indicators.get("volume_ratio", 1.0)
                )
                / 2.0,
            )
            momentum_similarity = 1.0 - min(
                1.0,
                abs(
                    current_characteristics["momentum"]
                    - pattern_indicators.get("price_momentum", 0)
                )
                / 0.1,
            )

            # Weight the similarities
            overall_similarity = (
                rsi_similarity * 0.4
                + volume_similarity * 0.3
                + momentum_similarity * 0.3
            )
            similarities.append(max(0.0, overall_similarity))

        # Return average similarity to this pattern type
        return np.mean(similarities) if similarities else 0.0

    def _generate_peak_recommendation(
        self, overall_prob: float, pattern_probs: Dict, current_chars: Dict
    ) -> Dict[str, str]:
        """Generate trading recommendation based on peak analysis"""

        # High probability of blow-off top
        if pattern_probs.get("blow_off_top", 0) > 0.7:
            return {
                "action": "avoid_buying",
                "reasoning": f"High blow-off top probability ({pattern_probs['blow_off_top']:.1%}). RSI: {current_chars['rsi']:.1f}, Volume spike: {current_chars['volume_ratio']:.1f}x",
                "risk_level": "very_high",
            }

        # High probability of major peak
        elif pattern_probs.get("major_peak", 0) > 0.6 or overall_prob > 0.6:
            return {
                "action": "reduce_buying",
                "reasoning": f"High peak probability ({overall_prob:.1%}). Overbought conditions detected",
                "risk_level": "high",
            }

        # Moderate peak probability
        elif overall_prob > 0.4:
            return {
                "action": "cautious",
                "reasoning": f"Moderate peak risk ({overall_prob:.1%}). Wait for better entry",
                "risk_level": "medium",
            }

        # Low peak probability - good for buying
        elif overall_prob < 0.2 and current_chars["rsi"] < 60:
            return {
                "action": "favorable_buying",
                "reasoning": f"Low peak risk ({overall_prob:.1%}). Good entry conditions",
                "risk_level": "low",
            }

        # Neutral conditions
        else:
            return {
                "action": "neutral",
                "reasoning": f"Neutral peak risk ({overall_prob:.1%}). Standard caution advised",
                "risk_level": "medium",
            }

    def get_adaptive_entry_strategy(
        self,
        current_price: float,
        current_indicators: Dict,
        recent_prices: List[float],
        recent_volumes: List[float],
    ) -> Dict:
        """
        Get adaptive entry strategy that learns from peak patterns.
        Returns specific entry recommendations and position sizing.
        """
        peak_analysis = self.predict_peak_probability(
            current_price, current_indicators, recent_prices, recent_volumes
        )

        # Base strategy adjustments
        base_position_multiplier = 1.0
        entry_patience = 0  # Minutes to wait
        max_chase_percentage = 0.02  # How much above current price to chase

        action = peak_analysis["recommended_action"]
        peak_prob = peak_analysis["peak_probability"]

        if action == "avoid_buying":
            # Completely avoid buying
            base_position_multiplier = 0.0
            entry_patience = 120  # Wait 2 hours
            max_chase_percentage = 0.0

        elif action == "reduce_buying":
            # Reduce position size significantly
            base_position_multiplier = 0.3
            entry_patience = 60  # Wait 1 hour
            max_chase_percentage = 0.005  # Only chase 0.5%

        elif action == "cautious":
            # Modest position size with patience
            base_position_multiplier = 0.6
            entry_patience = 30  # Wait 30 minutes
            max_chase_percentage = 0.01  # Chase 1%

        elif action == "favorable_buying":
            # Increase position size for good opportunities
            base_position_multiplier = 1.3
            entry_patience = 0  # No waiting
            max_chase_percentage = 0.03  # Chase up to 3%

        else:  # neutral
            # Standard approach
            base_position_multiplier = 1.0
            entry_patience = 15  # Brief wait
            max_chase_percentage = 0.015  # Chase 1.5%

        # Additional adjustments based on learned patterns
        if peak_prob > 0.5 and current_indicators.get("rsi", 50) > 70:
            # Very dangerous conditions based on historical patterns
            base_position_multiplier *= 0.5
            entry_patience = max(entry_patience, 90)

        # Calculate smart entry price
        smart_entry_price = self._calculate_smart_entry_price(
            current_price, recent_prices, peak_analysis, max_chase_percentage
        )

        return {
            "position_multiplier": base_position_multiplier,
            "entry_patience_minutes": entry_patience,
            "smart_entry_price": smart_entry_price,
            "max_chase_percentage": max_chase_percentage,
            "peak_analysis": peak_analysis,
            "strategy_reasoning": self._generate_strategy_reasoning(
                peak_analysis, base_position_multiplier
            ),
        }

    def _calculate_smart_entry_price(
        self,
        current_price: float,
        recent_prices: List[float],
        peak_analysis: Dict,
        max_chase_pct: float,
    ) -> float:
        """Calculate optimal entry price based on peak analysis and support levels"""

        # Find recent support levels
        if len(recent_prices) >= 20:
            recent_lows = []
            for i in range(2, len(recent_prices) - 2):
                if (
                    recent_prices[i] < recent_prices[i - 1]
                    and recent_prices[i] < recent_prices[i - 2]
                    and recent_prices[i] < recent_prices[i + 1]
                    and recent_prices[i] < recent_prices[i + 2]
                ):
                    recent_lows.append(recent_prices[i])

            if recent_lows:
                # Target entry near recent support
                support_level = max(recent_lows[-3:])  # Most recent strong support
                target_entry = max(
                    support_level, current_price * (1 - 0.02)
                )  # Don't go more than 2% below current
            else:
                target_entry = current_price * 0.998  # Slightly below current price
        else:
            target_entry = current_price * 0.999

        # Adjust based on peak probability
        peak_prob = peak_analysis.get("peak_probability", 0.3)
        if peak_prob > 0.6:
            # High peak probability - be very patient
            target_entry = min(target_entry, current_price * 0.985)  # Wait for 1.5% dip
        elif peak_prob < 0.2:
            # Low peak probability - can be more aggressive
            target_entry = min(
                current_price * (1 + max_chase_pct), current_price * 1.01
            )

        return target_entry

    def _generate_strategy_reasoning(
        self, peak_analysis: Dict, position_multiplier: float
    ) -> List[str]:
        """Generate human-readable reasoning for the strategy"""
        reasoning = []

        action = peak_analysis.get("recommended_action", "neutral")
        peak_prob = peak_analysis.get("peak_probability", 0.3)
        confidence = peak_analysis.get("confidence", 0.0)

        reasoning.append(
            f"Peak probability: {peak_prob:.1%} (confidence: {confidence:.1%})"
        )
        reasoning.append(f"Action: {action.replace('_', ' ').title()}")
        reasoning.append(f"Position adjustment: {position_multiplier:.1f}x normal size")

        # Add specific pattern warnings
        pattern_probs = peak_analysis.get("pattern_probabilities", {})
        for pattern_type, prob in pattern_probs.items():
            if prob > 0.5:
                reasoning.append(
                    f"Warning: {pattern_type.replace('_', ' ').title()} pattern detected ({prob:.1%})"
                )

        # Add current market characteristics
        current_chars = peak_analysis.get("current_characteristics", {})
        if current_chars.get("rsi", 50) > 70:
            reasoning.append(f"RSI overbought: {current_chars['rsi']:.1f}")
        if current_chars.get("volume_ratio", 1.0) > 2.0:
            reasoning.append(
                f"Volume spike: {current_chars['volume_ratio']:.1f}x normal"
            )

        return reasoning

    def update_patterns_from_trades(self, trade_history: List[Dict]):
        """Update pattern database based on actual trade outcomes"""
        print("🔄 Updating peak patterns based on trade outcomes...")

        for trade in trade_history[-50:]:  # Last 50 trades
            if trade.get("action") == "buy":
                entry_price = trade.get("price", 0)
                entry_time = trade.get("timestamp")

                # Look for trades that resulted in immediate losses (bought at peaks)
                # This would need integration with your actual trade tracking
                # For now, we'll use a simplified version

                # If this was a losing trade that dropped quickly, it might have been a peak
                if "loss_percentage" in trade and trade["loss_percentage"] > 0.03:
                    # This suggests we bought at a peak - learn from it
                    self._add_failed_entry_pattern(trade)

        self._save_patterns()

    def _add_failed_entry_pattern(self, failed_trade: Dict):
        """Add a pattern for a failed entry (bought at peak)"""
        # Create a pattern from the failed trade
        pattern_dict = {
            "peak_price": failed_trade.get("price", 0),
            "peak_time": failed_trade.get("timestamp", datetime.now().isoformat()),
            "drop_percentage": failed_trade.get("loss_percentage", 0),
            "time_to_drop": 30,  # Assume quick drop
            "volume_profile": [],
            "leading_indicators": {
                "rsi": failed_trade.get("rsi_at_entry", 50),
                "volume_ratio": 1.0,
                "price_momentum": 0.01,
            },
            "pattern_type": "failed_entry",
        }

        self.pattern_database.append(pattern_dict)
        print(
            f"📚 Added failed entry pattern: {failed_trade.get('price', 0):.2f} → {failed_trade.get('loss_percentage', 0):.1%} loss"
        )


# Integration class for the main trading bot
class PeakAwareTrader:
    """Integration wrapper for the peak avoidance system"""

    def __init__(self, lookback_days: int = 90):
        self.peak_system = PeakAvoidanceSystem(lookback_days)
        self.last_analysis_time = None
        self.current_entry_strategy = None

    def should_avoid_buying(
        self,
        current_price: float,
        indicators: Dict[str, float],
        recent_prices: List[float],
        recent_volumes: List[float],
    ) -> Tuple[bool, str]:
        """
        Main function to check if buying should be avoided due to peak risk.
        Returns (should_avoid, reasoning)
        """
        try:
            peak_analysis = self.peak_system.predict_peak_probability(
                current_price, indicators, recent_prices, recent_volumes
            )

            action = peak_analysis.get("recommended_action", "neutral")
            peak_prob = peak_analysis.get("peak_probability", 0.3)
            reasoning = peak_analysis.get("reasoning", "Standard analysis")

            # Determine if we should avoid buying
            should_avoid = action in ["avoid_buying", "reduce_buying"]

            if should_avoid:
                return True, f"PEAK RISK: {reasoning} (Probability: {peak_prob:.1%})"
            elif action == "cautious":
                return False, f"MODERATE RISK: {reasoning} - Consider smaller position"
            else:
                return False, f"LOW RISK: {reasoning} - Normal trading"

        except Exception as e:
            print(f"⚠️ Peak analysis error: {e}")
            return False, "Peak analysis unavailable - proceeding with caution"

    def get_position_adjustment(
        self,
        base_position_size: float,
        current_price: float,
        indicators: Dict[str, float],
        recent_prices: List[float],
        recent_volumes: List[float],
    ) -> Tuple[float, str]:
        """
        Get position size adjustment based on peak analysis.
        Returns (adjusted_position_size, reasoning)
        """
        try:
            entry_strategy = self.peak_system.get_adaptive_entry_strategy(
                current_price, indicators, recent_prices, recent_volumes
            )

            multiplier = entry_strategy.get("position_multiplier", 1.0)
            adjusted_size = base_position_size * multiplier

            reasoning_list = entry_strategy.get(
                "strategy_reasoning", ["Standard sizing"]
            )
            reasoning = " | ".join(reasoning_list[:2])  # First 2 reasons

            return adjusted_size, f"Position: {multiplier:.1f}x base - {reasoning}"

        except Exception as e:
            print(f"⚠️ Position adjustment error: {e}")
            return base_position_size, "Using standard position size"

    def analyze_historical_data(self, price_data: List[Dict]):
        """
        Analyze historical price data to build peak pattern database.
        price_data should be list of dicts with: price, volume, timestamp, indicators
        """
        if len(price_data) < 100:
            print("⚠️ Need at least 100 data points for peak analysis")
            return

        prices = [d["price"] for d in price_data]
        volumes = [d["volume"] for d in price_data]
        timestamps = [d["timestamp"] for d in price_data]
        indicators = [d.get("indicators", {}) for d in price_data]

        patterns = self.peak_system.analyze_price_history(
            prices, volumes, timestamps, indicators
        )
        print(f"🎯 Peak analysis complete: {len(patterns)} patterns identified")

        return patterns
