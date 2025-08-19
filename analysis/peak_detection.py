# bitcoin_bot/analysis/peak_detection.py
"""
Advanced Peak Detection System for Bitcoin Trading
Identifies and predicts price peaks to avoid buying at local/global maxima
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import os
from collections import deque
from scipy import signal, stats
from scipy.signal import find_peaks, argrelextrema, savgol_filter
from scipy.stats import zscore, linregress
import warnings
warnings.filterwarnings('ignore')

# Optional advanced libraries
try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

class PeakType(Enum):
    """Types of price peaks"""
    LOCAL_MAX = "local_maximum"
    GLOBAL_MAX = "global_maximum"
    RESISTANCE_LEVEL = "resistance_level"
    BLOW_OFF_TOP = "blow_off_top"
    DOUBLE_TOP = "double_top"
    HEAD_SHOULDERS = "head_and_shoulders"
    RISING_WEDGE = "rising_wedge"
    EXHAUSTION_PEAK = "exhaustion_peak"

class PeakConfidence(Enum):
    """Confidence levels for peak detection"""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95

@dataclass
class PeakPattern:
    """Represents a detected peak pattern"""
    peak_type: PeakType
    peak_index: int
    peak_price: float
    peak_time: datetime
    confidence: float
    drop_after_peak: Optional[float] = None
    time_to_drop: Optional[int] = None  # In periods
    volume_at_peak: Optional[float] = None
    volume_profile: List[float] = field(default_factory=list)
    leading_indicators: Dict[str, float] = field(default_factory=dict)
    pattern_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PeakDetectionConfig:
    """Configuration for peak detection"""
    # Detection parameters
    lookback_periods: int = 100
    min_prominence_pct: float = 0.02  # 2% minimum prominence
    min_distance_periods: int = 10
    
    # Pattern recognition
    enable_pattern_recognition: bool = True
    pattern_similarity_threshold: float = 0.75
    
    # Volume analysis
    volume_spike_threshold: float = 2.0  # 2x average volume
    volume_confirmation_required: bool = True
    
    # Statistical parameters
    zscore_threshold: float = 2.0
    confidence_threshold: float = 0.6
    
    # Machine learning
    use_ml_enhancement: bool = False
    ml_model_path: Optional[str] = None
    
    # Risk parameters
    peak_avoidance_threshold: float = 0.7
    cooldown_periods_after_peak: int = 20
    
    # Smoothing
    use_smoothing: bool = True
    smoothing_window: int = 5
    
    # Pattern database
    save_patterns: bool = True
    pattern_file: str = "peak_patterns.json"
    max_stored_patterns: int = 1000

class PeakDetector:
    """
    Advanced peak detection using multiple algorithms
    """
    
    def __init__(self, config: Optional[PeakDetectionConfig] = None):
        """
        Initialize peak detector
        
        Args:
            config: Peak detection configuration
        """
        self.config = config or PeakDetectionConfig()
        self.detected_patterns = []
        self.pattern_database = self._load_pattern_database()
        self.resistance_levels = []
        self.support_levels = []
        
        logger.info(f"Peak detector initialized with lookback={self.config.lookback_periods}")
    
    def detect_peaks(
        self,
        prices: np.ndarray,
        volumes: Optional[np.ndarray] = None,
        timestamps: Optional[List[datetime]] = None
    ) -> List[PeakPattern]:
        """
        Detect peaks using multiple methods
        
        Args:
            prices: Price array
            volumes: Optional volume array
            timestamps: Optional timestamp array
            
        Returns:
            List of detected peak patterns
        """
        if len(prices) < self.config.min_distance_periods:
            return []
        
        # Smooth prices if configured
        if self.config.use_smoothing and len(prices) > self.config.smoothing_window:
            smoothed_prices = savgol_filter(prices, self.config.smoothing_window, 3)
        else:
            smoothed_prices = prices
        
        peaks = []
        
        # Method 1: Scipy find_peaks
        scipy_peaks = self._detect_scipy_peaks(smoothed_prices)
        peaks.extend(scipy_peaks)
        
        # Method 2: Local extrema
        extrema_peaks = self._detect_local_extrema(smoothed_prices)
        peaks.extend(extrema_peaks)
        
        # Method 3: Statistical peaks
        stat_peaks = self._detect_statistical_peaks(smoothed_prices)
        peaks.extend(stat_peaks)
        
        # Method 4: Pattern-based peaks
        if self.config.enable_pattern_recognition:
            pattern_peaks = self._detect_pattern_peaks(prices, volumes)
            peaks.extend(pattern_peaks)
        
        # Merge and deduplicate peaks
        merged_peaks = self._merge_peaks(peaks)
        
        # Add metadata
        for peak in merged_peaks:
            if timestamps and peak.peak_index < len(timestamps):
                peak.peak_time = timestamps[peak.peak_index]
            
            if volumes is not None and peak.peak_index < len(volumes):
                peak.volume_at_peak = volumes[peak.peak_index]
                
                # Volume profile around peak
                start_idx = max(0, peak.peak_index - 5)
                end_idx = min(len(volumes), peak.peak_index + 5)
                peak.volume_profile = list(volumes[start_idx:end_idx])
            
            # Calculate drop after peak if possible
            if peak.peak_index < len(prices) - 10:
                future_prices = prices[peak.peak_index:peak.peak_index + 20]
                min_future = np.min(future_prices)
                peak.drop_after_peak = (peak.peak_price - min_future) / peak.peak_price
                peak.time_to_drop = np.argmin(future_prices)
        
        # Store detected patterns
        self.detected_patterns = merged_peaks
        
        return merged_peaks
    
    def _detect_scipy_peaks(self, prices: np.ndarray) -> List[PeakPattern]:
        """Detect peaks using scipy find_peaks"""
        peaks = []
        
        # Calculate prominence threshold
        price_range = np.max(prices) - np.min(prices)
        min_prominence = price_range * self.config.min_prominence_pct
        
        # Find peaks
        peak_indices, properties = find_peaks(
            prices,
            prominence=min_prominence,
            distance=self.config.min_distance_periods,
            width=3
        )
        
        for idx in peak_indices:
            confidence = self._calculate_peak_confidence(
                prices, idx, properties.get('prominences', [0])[0] if 'prominences' in properties else 0
            )
            
            pattern = PeakPattern(
                peak_type=PeakType.LOCAL_MAX,
                peak_index=idx,
                peak_price=prices[idx],
                peak_time=datetime.now(),
                confidence=confidence,
                pattern_metadata={'method': 'scipy_find_peaks'}
            )
            peaks.append(pattern)
        
        return peaks
    
    def _detect_local_extrema(self, prices: np.ndarray) -> List[PeakPattern]:
        """Detect peaks using local extrema"""
        peaks = []
        
        # Find local maxima
        local_max = argrelextrema(prices, np.greater, order=self.config.min_distance_periods // 2)[0]
        
        for idx in local_max:
            # Verify it's a significant peak
            window = self.config.min_distance_periods
            start = max(0, idx - window)
            end = min(len(prices), idx + window)
            
            if prices[idx] == np.max(prices[start:end]):
                confidence = self._calculate_peak_confidence(prices, idx)
                
                pattern = PeakPattern(
                    peak_type=PeakType.LOCAL_MAX,
                    peak_index=idx,
                    peak_price=prices[idx],
                    peak_time=datetime.now(),
                    confidence=confidence,
                    pattern_metadata={'method': 'local_extrema'}
                )
                peaks.append(pattern)
        
        return peaks
    
    def _detect_statistical_peaks(self, prices: np.ndarray) -> List[PeakPattern]:
        """Detect peaks using statistical methods"""
        peaks = []
        
        if len(prices) < 20:
            return peaks
        
        # Calculate z-scores
        z_scores = zscore(prices)
        
        # Find statistical outliers
        outlier_indices = np.where(z_scores > self.config.zscore_threshold)[0]
        
        # Filter for local maxima among outliers
        for idx in outlier_indices:
            if idx > 0 and idx < len(prices) - 1:
                if prices[idx] > prices[idx - 1] and prices[idx] > prices[idx + 1]:
                    confidence = min(0.95, (z_scores[idx] - self.config.zscore_threshold) / 2 + 0.5)
                    
                    pattern = PeakPattern(
                        peak_type=PeakType.LOCAL_MAX,
                        peak_index=idx,
                        peak_price=prices[idx],
                        peak_time=datetime.now(),
                        confidence=confidence,
                        pattern_metadata={'method': 'statistical', 'z_score': float(z_scores[idx])}
                    )
                    peaks.append(pattern)
        
        return peaks
    
    def _detect_pattern_peaks(self, prices: np.ndarray, volumes: Optional[np.ndarray]) -> List[PeakPattern]:
        """Detect specific peak patterns"""
        patterns = []
        
        # Double Top
        double_tops = self._find_double_tops(prices)
        patterns.extend(double_tops)
        
        # Head and Shoulders
        head_shoulders = self._find_head_and_shoulders(prices)
        patterns.extend(head_shoulders)
        
        # Blow-off Top
        if volumes is not None:
            blow_offs = self._find_blow_off_tops(prices, volumes)
            patterns.extend(blow_offs)
        
        # Rising Wedge
        wedges = self._find_rising_wedges(prices)
        patterns.extend(wedges)
        
        return patterns
    
    def _find_double_tops(self, prices: np.ndarray) -> List[PeakPattern]:
        """Find double top patterns"""
        patterns = []
        
        if len(prices) < 30:
            return patterns
        
        # Find all local peaks
        peaks, _ = find_peaks(prices, distance=5)
        
        if len(peaks) < 2:
            return patterns
        
        # Check for double tops
        for i in range(len(peaks) - 1):
            first_peak = peaks[i]
            
            for j in range(i + 1, min(i + 3, len(peaks))):  # Look for second peak within reasonable distance
                second_peak = peaks[j]
                
                # Check if peaks are similar height (within 2%)
                height_diff = abs(prices[first_peak] - prices[second_peak]) / prices[first_peak]
                
                if height_diff < 0.02:
                    # Find valley between peaks
                    valley_idx = np.argmin(prices[first_peak:second_peak]) + first_peak
                    valley_depth = (prices[first_peak] - prices[valley_idx]) / prices[first_peak]
                    
                    if valley_depth > 0.03:  # Valley at least 3% below peaks
                        pattern = PeakPattern(
                            peak_type=PeakType.DOUBLE_TOP,
                            peak_index=second_peak,
                            peak_price=prices[second_peak],
                            peak_time=datetime.now(),
                            confidence=0.7 + min(0.25, valley_depth * 2),
                            pattern_metadata={
                                'first_peak': int(first_peak),
                                'second_peak': int(second_peak),
                                'valley_index': int(valley_idx),
                                'valley_depth': float(valley_depth)
                            }
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def _find_head_and_shoulders(self, prices: np.ndarray) -> List[PeakPattern]:
        """Find head and shoulders patterns"""
        patterns = []
        
        if len(prices) < 40:
            return patterns
        
        # Find all local peaks
        peaks, _ = find_peaks(prices, distance=5)
        
        if len(peaks) < 3:
            return patterns
        
        # Look for head and shoulders
        for i in range(len(peaks) - 2):
            left_shoulder = peaks[i]
            head = peaks[i + 1]
            right_shoulder = peaks[i + 2]
            
            # Check if head is higher than shoulders
            if (prices[head] > prices[left_shoulder] and 
                prices[head] > prices[right_shoulder]):
                
                # Check if shoulders are similar height
                shoulder_diff = abs(prices[left_shoulder] - prices[right_shoulder]) / prices[left_shoulder]
                
                if shoulder_diff < 0.03:  # Within 3%
                    # Calculate neckline
                    left_valley = np.argmin(prices[left_shoulder:head]) + left_shoulder
                    right_valley = np.argmin(prices[head:right_shoulder]) + head
                    
                    neckline = (prices[left_valley] + prices[right_valley]) / 2
                    
                    pattern = PeakPattern(
                        peak_type=PeakType.HEAD_SHOULDERS,
                        peak_index=head,
                        peak_price=prices[head],
                        peak_time=datetime.now(),
                        confidence=0.75,
                        pattern_metadata={
                            'left_shoulder': int(left_shoulder),
                            'head': int(head),
                            'right_shoulder': int(right_shoulder),
                            'neckline': float(neckline)
                        }
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _find_blow_off_tops(self, prices: np.ndarray, volumes: np.ndarray) -> List[PeakPattern]:
        """Find blow-off top patterns (parabolic rise with volume spike)"""
        patterns = []
        
        if len(prices) < 20 or len(volumes) < 20:
            return patterns
        
        # Calculate price acceleration
        price_changes = np.diff(prices)
        price_acceleration = np.diff(price_changes)
        
        # Calculate volume ratio
        volume_ma = pd.Series(volumes).rolling(10).mean().values
        volume_ratio = volumes / np.where(volume_ma > 0, volume_ma, 1)
        
        # Find peaks with high acceleration and volume
        peaks, _ = find_peaks(prices, distance=10)
        
        for peak_idx in peaks:
            if peak_idx < 10 or peak_idx >= len(prices) - 2:
                continue
            
            # Check for parabolic rise
            recent_prices = prices[peak_idx - 10:peak_idx]
            price_gain = (prices[peak_idx] - recent_prices[0]) / recent_prices[0]
            
            # Check acceleration
            if peak_idx - 2 >= 0 and peak_idx - 2 < len(price_acceleration):
                acceleration = price_acceleration[peak_idx - 2:peak_idx].mean()
            else:
                acceleration = 0
            
            # Check volume spike
            if peak_idx < len(volume_ratio):
                vol_spike = volume_ratio[peak_idx]
            else:
                vol_spike = 1
            
            if (price_gain > 0.10 and  # 10% gain in 10 periods
                acceleration > 0 and
                vol_spike > self.config.volume_spike_threshold):
                
                confidence = min(0.9, 0.5 + price_gain + (vol_spike - 2) * 0.1)
                
                pattern = PeakPattern(
                    peak_type=PeakType.BLOW_OFF_TOP,
                    peak_index=peak_idx,
                    peak_price=prices[peak_idx],
                    peak_time=datetime.now(),
                    confidence=confidence,
                    volume_at_peak=volumes[peak_idx],
                    pattern_metadata={
                        'price_gain': float(price_gain),
                        'volume_spike': float(vol_spike),
                        'acceleration': float(acceleration)
                    }
                )
                patterns.append(pattern)
        
        return patterns
    
    def _find_rising_wedges(self, prices: np.ndarray) -> List[PeakPattern]:
        """Find rising wedge patterns"""
        patterns = []
        
        if len(prices) < 30:
            return patterns
        
        # Use sliding window to detect wedges
        window_size = 20
        
        for i in range(len(prices) - window_size):
            window = prices[i:i + window_size]
            
            # Find highs and lows
            highs = []
            lows = []
            
            for j in range(1, len(window) - 1):
                if window[j] > window[j-1] and window[j] > window[j+1]:
                    highs.append((j, window[j]))
                elif window[j] < window[j-1] and window[j] < window[j+1]:
                    lows.append((j, window[j]))
            
            if len(highs) >= 2 and len(lows) >= 2:
                # Calculate trendlines
                high_indices = [h[0] for h in highs]
                high_prices = [h[1] for h in highs]
                low_indices = [l[0] for l in lows]
                low_prices = [l[1] for l in lows]
                
                # Linear regression for trendlines
                high_slope, high_intercept, _, _, _ = linregress(high_indices, high_prices)
                low_slope, low_intercept, _, _, _ = linregress(low_indices, low_prices)
                
                # Check for converging lines (wedge)
                if high_slope > 0 and low_slope > 0 and high_slope < low_slope:
                    # Rising wedge detected
                    peak_idx = i + window_size - 1
                    
                    pattern = PeakPattern(
                        peak_type=PeakType.RISING_WEDGE,
                        peak_index=peak_idx,
                        peak_price=prices[peak_idx],
                        peak_time=datetime.now(),
                        confidence=0.65,
                        pattern_metadata={
                            'high_slope': float(high_slope),
                            'low_slope': float(low_slope),
                            'window_start': i
                        }
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _calculate_peak_confidence(self, prices: np.ndarray, peak_idx: int, prominence: float = 0) -> float:
        """Calculate confidence score for a detected peak"""
        confidence = 0.5  # Base confidence
        
        # Factor 1: Prominence
        if prominence > 0:
            price_range = np.max(prices) - np.min(prices)
            prominence_ratio = prominence / price_range
            confidence += prominence_ratio * 0.2
        
        # Factor 2: Relative height
        window = min(20, len(prices) // 4)
        start = max(0, peak_idx - window)
        end = min(len(prices), peak_idx + window)
        
        local_prices = prices[start:end]
        if len(local_prices) > 0:
            height_ratio = (prices[peak_idx] - np.mean(local_prices)) / np.std(local_prices) if np.std(local_prices) > 0 else 0
            confidence += min(0.2, height_ratio * 0.1)
        
        # Factor 3: Sharpness
        if peak_idx > 0 and peak_idx < len(prices) - 1:
            left_diff = prices[peak_idx] - prices[peak_idx - 1]
            right_diff = prices[peak_idx] - prices[peak_idx + 1]
            sharpness = (left_diff + right_diff) / (2 * prices[peak_idx])
            confidence += min(0.1, sharpness * 10)
        
        return min(0.95, confidence)
    
    def _merge_peaks(self, peaks: List[PeakPattern]) -> List[PeakPattern]:
        """Merge duplicate peaks detected by different methods"""
        if not peaks:
            return []
        
        # Sort by peak index
        sorted_peaks = sorted(peaks, key=lambda x: x.peak_index)
        
        merged = []
        current_group = [sorted_peaks[0]]
        
        for peak in sorted_peaks[1:]:
            # If peaks are close together, group them
            if peak.peak_index - current_group[-1].peak_index <= 3:
                current_group.append(peak)
            else:
                # Merge the current group and start a new one
                merged.append(self._merge_peak_group(current_group))
                current_group = [peak]
        
        # Don't forget the last group
        merged.append(self._merge_peak_group(current_group))
        
        return merged
    
    def _merge_peak_group(self, group: List[PeakPattern]) -> PeakPattern:
        """Merge a group of similar peaks into one"""
        if len(group) == 1:
            return group[0]
        
        # Use the peak with highest confidence
        best_peak = max(group, key=lambda x: x.confidence)
        
        # Combine confidence scores
        combined_confidence = 1 - np.prod([1 - p.confidence for p in group])
        best_peak.confidence = min(0.95, combined_confidence)
        
        # Combine metadata
        best_peak.pattern_metadata['merged_from'] = [p.peak_type.value for p in group]
        best_peak.pattern_metadata['merge_count'] = len(group)
        
        return best_peak
    
    def _load_pattern_database(self) -> List[Dict]:
        """Load historical patterns from file"""
        if not self.config.save_patterns:
            return []
        
        try:
            if os.path.exists(self.config.pattern_file):
                with open(self.config.pattern_file, 'r') as f:
                    data = json.load(f)
                    return data.get('patterns', [])
        except Exception as e:
            logger.warning(f"Failed to load pattern database: {e}")
        
        return []
    
    def save_patterns(self, patterns: List[PeakPattern]):
        """Save detected patterns to database"""
        if not self.config.save_patterns:
            return
        
        try:
            # Convert patterns to dict
            pattern_dicts = []
            for pattern in patterns:
                pattern_dict = {
                    'peak_type': pattern.peak_type.value,
                    'peak_price': pattern.peak_price,
                    'confidence': pattern.confidence,
                    'drop_after_peak': pattern.drop_after_peak,
                    'time_to_drop': pattern.time_to_drop,
                    'volume_at_peak': pattern.volume_at_peak,
                    'leading_indicators': pattern.leading_indicators,
                    'pattern_metadata': pattern.pattern_metadata,
                    'timestamp': pattern.peak_time.isoformat() if pattern.peak_time else None
                }
                pattern_dicts.append(pattern_dict)
            
            # Add to database
            self.pattern_database.extend(pattern_dicts)
            
            # Keep only recent patterns
            if len(self.pattern_database) > self.config.max_stored_patterns:
                self.pattern_database = self.pattern_database[-self.config.max_stored_patterns:]
            
            # Save to file
            with open(self.config.pattern_file, 'w') as f:
                json.dump({'patterns': self.pattern_database}, f, indent=2)
            
            logger.info(f"Saved {len(pattern_dicts)} patterns to database")
            
        except Exception as e:
            logger.error(f"Failed to save patterns: {e}")

class ResistanceSupportAnalyzer:
    """
    Analyzes resistance and support levels
    """
    
    def __init__(self, min_touches: int = 2, tolerance_pct: float = 0.02):
        """
        Initialize analyzer
        
        Args:
            min_touches: Minimum touches to confirm level
            tolerance_pct: Price tolerance for clustering
        """
        self.min_touches = min_touches
        self.tolerance_pct = tolerance_pct
        self.resistance_levels = []
        self.support_levels = []
    
    def find_levels(self, prices: np.ndarray) -> Tuple[List[float], List[float]]:
        """
        Find resistance and support levels
        
        Args:
            prices: Price array
            
        Returns:
            Tuple of (resistance_levels, support_levels)
        """
        if len(prices) < 20:
            return [], []
        
        # Find local peaks and troughs
        peaks, _ = find_peaks(prices, distance=5)
        troughs, _ = find_peaks(-prices, distance=5)
        
        # Get peak and trough prices
        peak_prices = prices[peaks] if len(peaks) > 0 else []
        trough_prices = prices[troughs] if len(troughs) > 0 else []
        
        # Cluster similar levels
        self.resistance_levels = self._cluster_levels(peak_prices)
        self.support_levels = self._cluster_levels(trough_prices)
        
        return self.resistance_levels, self.support_levels
    
    def _cluster_levels(self, levels: np.ndarray) -> List[float]:
        """Cluster similar price levels"""
        if len(levels) < self.min_touches:
            return []
        
        if not SKLEARN_AVAILABLE:
            # Simple clustering without sklearn
            return self._simple_cluster(levels)
        
        # Use DBSCAN for clustering
        levels_reshaped = levels.reshape(-1, 1)
        
        # Calculate epsilon based on tolerance
        eps = np.mean(levels) * self.tolerance_pct
        
        clustering = DBSCAN(eps=eps, min_samples=self.min_touches).fit(levels_reshaped)
        
        # Extract cluster centers
        clustered_levels = []
        for label in set(clustering.labels_):
            if label != -1:  # Ignore noise
                cluster_points = levels[clustering.labels_ == label]
                clustered_levels.append(np.mean(cluster_points))
        
        return sorted(clustered_levels)
    
    def _simple_cluster(self, levels: np.ndarray) -> List[float]:
        """Simple clustering without sklearn"""
        if len(levels) == 0:
            return []
        
        sorted_levels = np.sort(levels)
        clusters = []
        current_cluster = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            if (level - current_cluster[-1]) / current_cluster[-1] < self.tolerance_pct:
                current_cluster.append(level)
            else:
                if len(current_cluster) >= self.min_touches:
                    clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        
        if len(current_cluster) >= self.min_touches:
            clusters.append(np.mean(current_cluster))
        
        return clusters
    
    def get_nearest_levels(self, current_price: float) -> Dict[str, Optional[float]]:
        """Get nearest resistance and support levels"""
        result = {
            'nearest_resistance': None,
            'nearest_support': None,
            'distance_to_resistance': None,
            'distance_to_support': None
        }
        
        # Find nearest resistance
        resistances_above = [r for r in self.resistance_levels if r > current_price]
        if resistances_above:
            result['nearest_resistance'] = min(resistances_above)
            result['distance_to_resistance'] = (result['nearest_resistance'] - current_price) / current_price
        
        # Find nearest support
        supports_below = [s for s in self.support_levels if s < current_price]
        if supports_below:
            result['nearest_support'] = max(supports_below)
            result['distance_to_support'] = (current_price - result['nearest_support']) / current_price
        
        return result

class PeakProbabilityCalculator:
    """
    Calculates probability of current price being a peak
    """
    
    def __init__(self, detector: PeakDetector, analyzer: ResistanceSupportAnalyzer):
        """
        Initialize calculator
        
        Args:
            detector: Peak detector instance
            analyzer: Resistance/support analyzer instance
        """
        self.detector = detector
        self.analyzer = analyzer
    
    def calculate_peak_probability(
        self,
        current_price: float,
        prices: np.ndarray,
        volumes: Optional[np.ndarray] = None,
        indicators: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculate probability that current price is a peak
        
        Args:
            current_price: Current price
            prices: Historical prices
            volumes: Historical volumes
            indicators: Technical indicators
            
        Returns:
            Dictionary with peak probability and analysis
        """
        probabilities = []
        reasons = []
        
        # Check if at recent high
        if len(prices) >= 20:
            recent_high = np.max(prices[-20:])
            if current_price >= recent_high * 0.99:
                prob = min(0.8, (current_price - recent_high) / recent_high * 10 + 0.5)
                probabilities.append(prob)
                reasons.append(f"At recent high (prob: {prob:.2f})")
        
        # Check distance from resistance
        levels = self.analyzer.get_nearest_levels(current_price)
        if levels['nearest_resistance'] and levels['distance_to_resistance']:
            if levels['distance_to_resistance'] < 0.02:  # Within 2% of resistance
                prob = 0.7 - levels['distance_to_resistance'] * 10
                probabilities.append(prob)
                reasons.append(f"Near resistance (prob: {prob:.2f})")
        
        # Check technical indicators
        if indicators:
            indicator_prob = self._analyze_indicators(indicators)
            if indicator_prob > 0.5:
                probabilities.append(indicator_prob)
                reasons.append(f"Overbought indicators (prob: {indicator_prob:.2f})")
        
        # Check volume divergence
        if volumes is not None and len(volumes) >= 10:
            volume_prob = self._analyze_volume_divergence(prices, volumes)
            if volume_prob > 0.5:
                probabilities.append(volume_prob)
                reasons.append(f"Volume divergence (prob: {volume_prob:.2f})")
        
        # Check for recent peaks
        recent_peaks = self.detector.detect_peaks(prices[-50:] if len(prices) >= 50 else prices)
        if recent_peaks:
            latest_peak = recent_peaks[-1]
            if latest_peak.peak_index >= len(prices) - 5:  # Peak in last 5 periods
                probabilities.append(latest_peak.confidence)
                reasons.append(f"Recent peak detected (conf: {latest_peak.confidence:.2f})")
        
        # Calculate overall probability
        if probabilities:
            # Use weighted average with higher weights for stronger signals
            weights = [p ** 2 for p in probabilities]  # Square to emphasize strong signals
            overall_prob = np.average(probabilities, weights=weights)
        else:
            overall_prob = 0.3  # Default low probability
        
        # Determine recommendation
        if overall_prob > 0.7:
            recommendation = "AVOID_BUYING"
            risk_level = "HIGH"
        elif overall_prob > 0.5:
            recommendation = "REDUCE_POSITION"
            risk_level = "MEDIUM"
        else:
            recommendation = "NORMAL_TRADING"
            risk_level = "LOW"
        
        return {
            'peak_probability': overall_prob,
            'component_probabilities': probabilities,
            'reasons': reasons,
            'recommendation': recommendation,
            'risk_level': risk_level,
            'nearest_resistance': levels.get('nearest_resistance'),
            'distance_to_resistance': levels.get('distance_to_resistance')
        }
    
    def _analyze_indicators(self, indicators: Dict[str, float]) -> float:
        """Analyze technical indicators for peak probability"""
        prob_components = []
        
        # RSI
        rsi = indicators.get('rsi', 50)
        if rsi > 70:
            prob_components.append(0.6 + (rsi - 70) / 60)  # 0.6 to 0.8
        elif rsi > 60:
            prob_components.append(0.4 + (rsi - 60) / 50)  # 0.4 to 0.6
        
        # MACD divergence
        macd = indicators.get('macd', 0)
        signal = indicators.get('signal', 0)
        if macd < signal and macd > 0:  # Bearish divergence
            prob_components.append(0.6)
        
        # Bollinger Bands
        price = indicators.get('current_price', 0)
        upper_band = indicators.get('upper_band', price)
        if upper_band > 0 and price > upper_band:
            prob_components.append(0.7)
        
        return np.mean(prob_components) if prob_components else 0.3
    
    def _analyze_volume_divergence(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Analyze volume for divergence"""
        if len(prices) < 10 or len(volumes) < 10:
            return 0.3
        
        # Check if price rising but volume falling (bearish divergence)
        recent_prices = prices[-10:]
        recent_volumes = volumes[-10:]
        
        price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        volume_trend = (recent_volumes[-1] - recent_volumes[0]) / recent_volumes[0] if recent_volumes[0] > 0 else 0
        
        if price_trend > 0.02 and volume_trend < -0.1:  # Price up, volume down
            return 0.7
        elif price_trend > 0.01 and volume_trend < 0:
            return 0.6
        else:
            return 0.3

class PeakAvoidanceSystem:
    """
    Complete peak avoidance system for trading
    """
    
    def __init__(self, config: Optional[PeakDetectionConfig] = None):
        """
        Initialize peak avoidance system
        
        Args:
            config: Configuration for peak detection
        """
        self.config = config or PeakDetectionConfig()
        self.detector = PeakDetector(self.config)
        self.analyzer = ResistanceSupportAnalyzer()
        self.calculator = PeakProbabilityCalculator(self.detector, self.analyzer)
        
        # Tracking
        self.last_peak_time = None
        self.consecutive_peak_signals = 0
        
        logger.info("Peak avoidance system initialized")
    
    def analyze(
        self,
        current_price: float,
        prices: List[float],
        volumes: Optional[List[float]] = None,
        indicators: Optional[Dict[str, float]] = None,
        timestamps: Optional[List[datetime]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive peak analysis
        
        Args:
            current_price: Current price
            prices: Historical prices
            volumes: Historical volumes
            indicators: Technical indicators
            timestamps: Timestamps for prices
            
        Returns:
            Complete analysis with recommendations
        """
        # Convert to numpy arrays
        prices_array = np.array(prices)
        volumes_array = np.array(volumes) if volumes else None
        
        # Detect peaks
        detected_peaks = self.detector.detect_peaks(prices_array, volumes_array, timestamps)
        
        # Find resistance/support levels
        resistance_levels, support_levels = self.analyzer.find_levels(prices_array)
        
        # Calculate peak probability
        peak_analysis = self.calculator.calculate_peak_probability(
            current_price, prices_array, volumes_array, indicators
        )
        
        # Check for recent peaks
        recent_peak = None
        if detected_peaks:
            for peak in reversed(detected_peaks):
                if peak.peak_index >= len(prices) - 10:  # Peak in last 10 periods
                    recent_peak = peak
                    break
        
        # Update tracking
        if peak_analysis['peak_probability'] > self.config.peak_avoidance_threshold:
            self.consecutive_peak_signals += 1
            self.last_peak_time = datetime.now()
        else:
            self.consecutive_peak_signals = 0
        
        # Check cooldown
        in_cooldown = False
        if self.last_peak_time:
            time_since_peak = (datetime.now() - self.last_peak_time).total_seconds() / 60  # Minutes
            if time_since_peak < self.config.cooldown_periods_after_peak * 15:  # Assuming 15-min periods
                in_cooldown = True
        
        # Generate final recommendation
        should_avoid = (
            peak_analysis['peak_probability'] > self.config.peak_avoidance_threshold or
            self.consecutive_peak_signals >= 3 or
            in_cooldown
        )
        
        # Position sizing recommendation
        if should_avoid:
            position_multiplier = 0.0
            action = "AVOID_ENTRY"
        elif peak_analysis['peak_probability'] > 0.5:
            position_multiplier = 0.5
            action = "REDUCE_SIZE"
        else:
            position_multiplier = 1.0
            action = "NORMAL_ENTRY"
        
        return {
            'current_price': current_price,
            'detected_peaks': len(detected_peaks),
            'recent_peak': recent_peak,
            'resistance_levels': resistance_levels,
            'support_levels': support_levels,
            'peak_probability': peak_analysis['peak_probability'],
            'peak_reasons': peak_analysis['reasons'],
            'should_avoid_buying': should_avoid,
            'position_multiplier': position_multiplier,
            'recommended_action': action,
            'consecutive_peak_signals': self.consecutive_peak_signals,
            'in_cooldown': in_cooldown,
            'risk_level': peak_analysis['risk_level']
        }
    
    def save_analysis(self, analysis: Dict[str, Any]):
        """Save analysis results for learning"""
        if self.config.save_patterns and self.detector.detected_patterns:
            self.detector.save_patterns(self.detector.detected_patterns)

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    periods = 200
    t = np.linspace(0, 4 * np.pi, periods)
    
    # Create price data with peaks
    base_price = 50000
    trend = t * 100
    seasonal = 2000 * np.sin(t)
    noise = np.random.randn(periods) * 500
    
    prices = base_price + trend + seasonal + noise
    
    # Add some artificial peaks
    prices[50] *= 1.05  # 5% spike
    prices[100] *= 1.08  # 8% spike
    prices[150] *= 1.03  # 3% spike
    
    # Generate volume data
    volumes = np.random.exponential(1000, periods)
    volumes[50] *= 3  # Volume spike at peak
    volumes[100] *= 4
    
    # Generate timestamps
    timestamps = [datetime.now() - timedelta(minutes=15*i) for i in range(periods-1, -1, -1)]
    
    # Test the system
    config = PeakDetectionConfig(
        lookback_periods=50,
        min_prominence_pct=0.02,
        enable_pattern_recognition=True,
        save_patterns=True
    )
    
    system = PeakAvoidanceSystem(config)
    
    # Analyze current situation
    current_price = prices[-1]
    indicators = {
        'rsi': 72,
        'macd': 100,
        'signal': 80,
        'current_price': current_price,
        'upper_band': current_price * 0.98
    }
    
    analysis = system.analyze(
        current_price=current_price,
        prices=list(prices),
        volumes=list(volumes),
        indicators=indicators,
        timestamps=timestamps
    )
    
    print("Peak Analysis Results:")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Peak Probability: {analysis['peak_probability']:.2%}")
    print(f"Should Avoid Buying: {analysis['should_avoid_buying']}")
    print(f"Recommended Action: {analysis['recommended_action']}")
    print(f"Risk Level: {analysis['risk_level']}")
    print(f"Position Multiplier: {analysis['position_multiplier']:.2f}x")
    print(f"\nReasons:")
    for reason in analysis['peak_reasons']:
        print(f"  - {reason}")
    print(f"\nResistance Levels: {analysis['resistance_levels']}")
    print(f"Support Levels: {analysis['support_levels']}")

