import numpy as np
import pandas as pd
from typing import Optional, List, Dict
import os
import logging
from dotenv import load_dotenv
from datetime import datetime, timedelta
import requests
from tenacity import retry, wait_exponential, stop_after_attempt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import yfinance as yf

# Load environment variables
load_dotenv()

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# News API key
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
if not NEWS_API_KEY:
    logger.error("NEWS_API_KEY not found in environment variables")

# Ensure NLTK data
nltk.download("vader_lexicon", quiet=True)
sid = SentimentIntensityAnalyzer()

# Enhanced news keywords for macro-economic events
NEWS_KEYWORDS = [
    "bitcoin",
    "btc",
    "cryptocurrency",
    "crypto",
    "trump tariffs",
    "trade war",
    "economic policy",
    "federal reserve",
    "fed",
    "interest rates",
    "inflation",
    "recession",
    "market crash",
    "liquidation",
    "risk-off",
    "risk-on",
    "dollar strength",
    "dxy",
    "stock market",
    "s&p 500",
    "nasdaq",
    "equity markets",
    "china trade",
    "geopolitical",
    "regulatory",
]

# Risk-off signal keywords
RISK_OFF_KEYWORDS = [
    "tariff",
    "trade war",
    "recession",
    "crash",
    "liquidation",
    "sell-off",
    "emergency",
    "crisis",
    "sanctions",
    "conflict",
    "volatility",
    "panic",
    "margin call",
    "deleveraging",
    "risk-off",
]

# News cache
news_cache = {"timestamp": None, "articles": None}
NEWS_CACHE_DURATION = timedelta(minutes=60)  # Reduced cache time for faster updates

# Market data cache for correlations
market_data_cache = {"timestamp": None, "data": None}
MARKET_CACHE_DURATION = timedelta(minutes=45)


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(5))
def fetch_enhanced_news(top_n: int = 20) -> Optional[List[dict]]:
    """
    Fetch latest news using enhanced keywords for macro-economic events.
    Returns up to top_n articles with enhanced filtering.
    """
    current_time = datetime.now()
    if (
        news_cache["timestamp"]
        and (current_time - news_cache["timestamp"]) < NEWS_CACHE_DURATION
    ):
        logger.debug("Using cached news articles")
        return news_cache["articles"][:top_n]

    if not NEWS_API_KEY:
        logger.error("No News API key available")
        return None

    all_articles = []

    # Search for different keyword combinations
    search_queries = [
        "bitcoin cryptocurrency",
        "trump tariffs trade war",
        "federal reserve interest rates",
        "market crash recession",
        "crypto regulation policy",
    ]

    for query in search_queries:
        try:
            logger.info(f"Fetching news for: {query}")
            url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&apiKey={NEWS_API_KEY}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            articles = response.json().get("articles", [])
            all_articles.extend(articles)

        except Exception as e:
            logger.warning(f"Failed to fetch news for query '{query}': {e}")
            continue

    # Remove duplicates and sort by publish date
    unique_articles = {}
    for article in all_articles:
        url = article.get("url", "")
        if url and url not in unique_articles:
            unique_articles[url] = article

    sorted_articles = sorted(
        unique_articles.values(), key=lambda x: x.get("publishedAt", ""), reverse=True
    )

    news_cache["timestamp"] = current_time
    news_cache["articles"] = sorted_articles
    logger.info(f"Fetched {len(sorted_articles)} unique articles")

    return sorted_articles[:top_n]


def calculate_enhanced_sentiment(articles: Optional[List[dict]]) -> Dict[str, float]:
    """
    Calculate enhanced sentiment analysis with risk-off probability.
    Returns sentiment score and risk-off probability.
    """
    if not articles:
        logger.warning("No articles for sentiment analysis; returning neutral")
        return {"sentiment": 0.0, "risk_off_probability": 0.0, "macro_weight": 1.0}

    total_sentiment = 0.0
    weighted_sentiment = 0.0
    total_weight = 0.0
    risk_off_signals = 0
    macro_articles = 0

    for article in articles:
        title = article.get("title", "").lower()
        description = article.get("description", "") or ""
        text = f"{title}. {description}"

        # Calculate base sentiment
        score = sid.polarity_scores(text)["compound"]

        # Determine article weight based on content
        weight = 1.0
        is_macro = False

        # Higher weight for macro-economic articles
        macro_keywords = [
            "tariff",
            "fed",
            "federal reserve",
            "trade war",
            "recession",
            "policy",
        ]
        if any(keyword in title for keyword in macro_keywords):
            weight = 2.0
            is_macro = True
            macro_articles += 1

        # Higher weight for crypto-specific articles
        crypto_keywords = ["bitcoin", "btc", "cryptocurrency", "crypto"]
        if any(keyword in title for keyword in crypto_keywords) and not is_macro:
            weight = 1.5

        # Check for risk-off signals
        if any(keyword in title for keyword in RISK_OFF_KEYWORDS):
            risk_off_signals += 1
            # Amplify negative sentiment for risk-off articles
            if score < 0:
                score *= 1.5

        total_sentiment += score
        weighted_sentiment += score * weight
        total_weight += weight

    avg_sentiment = total_sentiment / len(articles) if articles else 0.0
    weighted_avg_sentiment = (
        weighted_sentiment / total_weight if total_weight > 0 else 0.0
    )
    risk_off_probability = min(
        1.0, risk_off_signals / max(1, len(articles)) * 2
    )  # Scale to 0-1
    macro_weight = 1.0 + (
        macro_articles / max(1, len(articles))
    )  # Boost when more macro news

    logger.info(
        f"Sentiment Analysis - Avg: {avg_sentiment:.3f}, Weighted: {weighted_avg_sentiment:.3f}, "
        f"Risk-off prob: {risk_off_probability:.3f}, Macro articles: {macro_articles}"
    )

    return {
        "sentiment": weighted_avg_sentiment,
        "risk_off_probability": risk_off_probability,
        "macro_weight": macro_weight,
        "total_articles": len(articles),
        "macro_articles": macro_articles,
    }


def get_market_correlations() -> Dict[str, float]:
    """
    Calculate Bitcoin correlations with traditional markets.
    Fixed version with better error handling and alternative tickers.
    """
    current_time = datetime.now()
    if (
        market_data_cache["timestamp"]
        and (current_time - market_data_cache["timestamp"]) < MARKET_CACHE_DURATION
    ):
        return market_data_cache["data"]

    try:
        # Fetch 30 days of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        # Updated tickers - using EUR-based assets where possible to match your EUR trading
        tickers = {
            "BTC-EUR": "Bitcoin EUR",  # Match your actual trading pair
            "SPY": "S&P 500",
            "UUP": "Dollar Index ETF",  # Alternative to DXY=X (more reliable)
            "GLD": "Gold ETF",  # Alternative to GC=F (more reliable)
        }

        correlations = {}

        # Download data with fixed warnings
        try:
            data = yf.download(
                list(tickers.keys()),
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,  # Explicitly set to avoid FutureWarning
            )["Close"]

            if data.empty:
                logger.warning("No market data downloaded")
                return {"SPY": 0.0, "DXY": 0.0, "GOLD": 0.0}

        except Exception as e:
            logger.error(f"Failed to download market data: {e}")
            return {"SPY": 0.0, "DXY": 0.0, "GOLD": 0.0}

        # Handle single vs multiple tickers case
        if len(tickers) == 1 or not isinstance(data, pd.DataFrame):
            logger.warning("Insufficient market data structure")
            return {"SPY": 0.0, "DXY": 0.0, "GOLD": 0.0}

        # Calculate BTC returns with fixed warning
        if "BTC-EUR" not in data.columns:
            logger.warning("BTC/EUR data not available, trying BTC-EUR fallback")
            if "BTC-EUR" in data.columns:
                logger.info("Using BTC-EUR as fallback for correlation analysis")
                btc_returns = data["BTC-EUR"].pct_change(fill_method=None).dropna()
            else:
                logger.warning("No BTC data available for correlation analysis")
                return {"SPY": 0.0, "DXY": 0.0, "GOLD": 0.0}
        else:
            btc_returns = data["BTC-EUR"].pct_change(fill_method=None).dropna()

        # Calculate correlations for each asset
        for ticker, name in tickers.items():
            if ticker in ["BTC-EUR", "BTC-EUR"]:  # Skip BTC itself
                continue

            try:
                if ticker in data.columns:
                    other_returns = data[ticker].pct_change(fill_method=None).dropna()

                    # Align the series
                    aligned_btc, aligned_other = btc_returns.align(
                        other_returns, join="inner"
                    )

                    if len(aligned_btc) > 10:  # Need sufficient data
                        correlation = aligned_btc.corr(aligned_other)

                        # Map to standard names
                        if ticker == "SPY":
                            correlations["SPY"] = (
                                float(correlation) if not pd.isna(correlation) else 0.0
                            )
                        elif ticker == "UUP":
                            correlations["DXY"] = (
                                float(correlation) if not pd.isna(correlation) else 0.0
                            )  # UUP as DXY proxy
                        elif ticker == "GLD":
                            correlations["GOLD"] = (
                                float(correlation) if not pd.isna(correlation) else 0.0
                            )
                    else:
                        logger.warning(
                            f"Insufficient data for {ticker}: {len(aligned_btc)} points"
                        )
                        if ticker == "SPY":
                            correlations["SPY"] = 0.0
                        elif ticker == "UUP":
                            correlations["DXY"] = 0.0
                        elif ticker == "GLD":
                            correlations["GOLD"] = 0.0
                else:
                    logger.warning(f"Ticker {ticker} not in downloaded data")
                    if ticker == "SPY":
                        correlations["SPY"] = 0.0
                    elif ticker == "UUP":
                        correlations["DXY"] = 0.0
                    elif ticker == "GLD":
                        correlations["GOLD"] = 0.0

            except Exception as e:
                logger.warning(f"Failed to calculate correlation for {ticker}: {e}")
                if ticker == "SPY":
                    correlations["SPY"] = 0.0
                elif ticker == "UUP":
                    correlations["DXY"] = 0.0
                elif ticker == "GLD":
                    correlations["GOLD"] = 0.0

        # Ensure all expected keys exist
        for key in ["SPY", "DXY", "GOLD"]:
            if key not in correlations:
                correlations[key] = 0.0

        market_data_cache["timestamp"] = current_time
        market_data_cache["data"] = correlations

        logger.info(f"Market correlations: {correlations}")
        return correlations

    except Exception as e:
        logger.error(f"Failed to calculate market correlations: {e}")
        return {"SPY": 0.0, "DXY": 0.0, "GOLD": 0.0}


def detect_liquidation_cascade(
    prices: List[float], volumes: List[float]
) -> Dict[str, float]:
    """
    Detect potential liquidation cascades based on price and volume patterns.
    """
    if len(prices) < 10 or len(volumes) < 10:
        return {"cascade_probability": 0.0, "volume_spike": 0.0}

    recent_prices = np.array(prices[-10:])
    recent_volumes = np.array(volumes[-10:])

    # Calculate recent price change
    price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]

    # Calculate volume spike
    avg_volume = np.mean(recent_volumes[:-1])
    current_volume = recent_volumes[-1]
    volume_spike = current_volume / avg_volume if avg_volume > 0 else 1.0

    # Calculate volatility spike
    returns = np.diff(recent_prices) / recent_prices[:-1]
    volatility = np.std(returns)

    # Liquidation cascade probability
    cascade_probability = 0.0

    # High volume + large price drop = potential cascade
    if price_change < -0.03 and volume_spike > 2.0:  # 3% drop + 2x volume
        cascade_probability += 0.4

    # High volatility increases probability
    if volatility > 0.05:  # 5% volatility
        cascade_probability += 0.3

    # Sustained selling pressure
    if np.sum(np.diff(recent_prices) < 0) >= 7:  # 7 out of 9 candles red
        cascade_probability += 0.3

    cascade_probability = min(1.0, cascade_probability)

    return {
        "cascade_probability": cascade_probability,
        "volume_spike": volume_spike,
        "volatility": volatility,
        "price_change": price_change,
    }


def calculate_risk_adjusted_indicators(
    prices: List[float], volumes: List[float], news_analysis: Dict
) -> Dict[str, float]:
    """
    Calculate indicators with risk adjustment based on market conditions.
    """
    # Standard indicators
    rsi = calculate_rsi(prices) or 50
    macd, signal = calculate_macd(prices) or (0, 0)
    ma_short = calculate_moving_average(prices, 20) or prices[-1]
    ma_long = calculate_moving_average(prices, 50) or prices[-1]
    vwap = calculate_vwap(prices, volumes) or prices[-1]

    # Enhanced indicators
    correlations = get_market_correlations()
    liquidation_signals = detect_liquidation_cascade(prices, volumes)

    # Risk adjustment factor
    risk_factor = 1.0
    risk_factor += news_analysis.get("risk_off_probability", 0) * 0.5
    risk_factor += liquidation_signals.get("cascade_probability", 0) * 0.3

    # Adjust RSI thresholds based on risk
    adjusted_rsi_buy = (
        30 + (risk_factor - 1) * 20
    )  # Raise buy threshold in risky conditions
    adjusted_rsi_sell = (
        70 - (risk_factor - 1) * 10
    )  # Lower sell threshold in risky conditions

    return {
        "rsi": rsi,
        "adjusted_rsi_buy": adjusted_rsi_buy,
        "adjusted_rsi_sell": adjusted_rsi_sell,
        "macd": macd,
        "signal": signal,
        "ma_short": ma_short,
        "ma_long": ma_long,
        "vwap": vwap,
        "correlations": correlations,
        "liquidation_signals": liquidation_signals,
        "risk_factor": risk_factor,
    }


# Keep existing functions for backward compatibility
def calculate_sentiment(articles: Optional[List[dict]]) -> float:
    """Backward compatibility function."""
    enhanced = calculate_enhanced_sentiment(articles)
    return enhanced["sentiment"]


def fetch_latest_news(top_n: int = 10) -> Optional[List[dict]]:
    """Backward compatibility function."""
    return fetch_enhanced_news(top_n)


# Keep all other existing functions unchanged
def calculate_moving_average(prices: List[float], window: int) -> Optional[float]:
    """Calculate simple moving average."""
    if len(prices) < window:
        logger.debug(f"Insufficient data for MA{window}: {len(prices)} < {window}")
        return None
    return float(np.mean(prices[-window:]))


def calculate_rsi(prices: List[float], window: int = 14) -> Optional[float]:
    """Calculate Relative Strength Index."""
    if len(prices) < window + 1:
        logger.debug(f"Insufficient data for RSI{window}: {len(prices)} < {window + 1}")
        return None
    delta = np.diff(prices)
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gains[-window:])
    avg_loss = np.mean(losses[-window:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def calculate_macd(
    prices: List[float],
    short_window: int = 12,
    long_window: int = 26,
    signal_window: int = 9,
) -> Optional[tuple[float, float]]:
    """Calculate MACD and signal line."""
    if len(prices) < long_window:
        logger.debug(f"Insufficient data for MACD: {len(prices)} < {long_window}")
        return None, None
    prices_series = pd.Series(prices)
    short_ema = prices_series.ewm(span=short_window, adjust=False).mean()
    long_ema = prices_series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return float(macd.iloc[-1]), float(signal.iloc[-1])


def calculate_bollinger_bands(
    prices: List[float], period: int = 20, std_dev: float = 2
) -> Optional[tuple[float, float, float]]:
    """Calculate Bollinger Bands (upper, middle, lower)."""
    if len(prices) < period:
        logger.debug(
            f"Insufficient data for Bollinger Bands (period={period}): {len(prices)} < {period}"
        )
        return None, None, None
    middle_band = float(np.mean(prices[-period:]))
    std = np.std(prices[-period:])
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    logger.debug(
        f"Bollinger Bands - Upper: {upper_band:.2f}, Middle: {middle_band:.2f}, Lower: {lower_band:.2f}"
    )
    return upper_band, middle_band, lower_band


def calculate_vwap(prices: List[float], volumes: List[float]) -> Optional[float]:
    """Calculate Volume Weighted Average Price."""
    if len(prices) < 2 or len(volumes) < 2 or len(prices) != len(volumes):
        logger.debug(
            f"Insufficient data for VWAP: Prices={len(prices)}, Volumes={len(volumes)}"
        )
        return None
    try:
        # Ensure prices and volumes are flat lists of floats
        prices = [float(p) for p in np.array(prices).flatten()]
        volumes = [float(v) for v in np.array(volumes).flatten()]
        if len(prices) != len(volumes):
            logger.debug("Mismatched lengths after flattening")
            return None
        cum_volume = np.cumsum(volumes)
        if cum_volume[-1] == 0:
            logger.debug("Zero cumulative volume")
            return None
        vwap = np.sum(np.array(prices) * np.array(volumes)) / cum_volume[-1]
        return float(vwap)
    except Exception as e:
        logger.error(f"VWAP calculation failed: {e}")
        return None
