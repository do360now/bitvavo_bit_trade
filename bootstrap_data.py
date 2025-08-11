#!/usr/bin/env python3
"""
Bootstrap script to initialize the trading bot with proper historical data.
Run this before starting the main bot for the first time.
"""

import time
import json
from datetime import datetime, timedelta
from trade_executor import TradeExecutor
from data_manager import DataManager
from bitvavo_api import authenticate_exchange
from config import PRICE_HISTORY_FILE, BOT_LOGS_FILE
from logger_config import logger

def bootstrap_historical_data():
    """Bootstrap the bot with historical OHLC data"""
    
    logger.info("🚀 Bootstrapping trading bot with historical data...")
    
    # Initialize components
    bitvavo = authenticate_exchange()
    trade_executor = TradeExecutor(bitvavo)
    data_manager = DataManager(PRICE_HISTORY_FILE, BOT_LOGS_FILE)
    
    # Clear existing invalid data
    logger.info("Clearing existing price history...")
    with open(PRICE_HISTORY_FILE, 'w') as f:
        json.dump([], f)
    
    # Fetch historical data in chunks to avoid rate limits
    intervals_to_fetch = [
        (7, '15m'),    # Last 7 days of 15-minute data
        (30, '1h'),    # Last 30 days of hourly data 
        (90, '4h'),    # Last 90 days of 4-hour data
    ]
    
    total_candles = 0
    
    for days_back, interval in intervals_to_fetch:
        logger.info(f"Fetching {interval} data for last {days_back} days...")
        
        # Calculate since timestamp
        since_time = int(time.time() - (days_back * 24 * 3600))
        
        try:
            # Fetch OHLC data
            ohlc_data = trade_executor.get_ohlc_data(
                pair="BTC/EUR",
                interval=interval,
                since=since_time,
                limit=1000  # Get up to 1000 candles per request
            )
            
            if ohlc_data:
                # Append to data manager
                new_candles = data_manager.append_ohlc_data(ohlc_data)
                total_candles += new_candles
                logger.info(f"✅ Added {new_candles} candles from {interval} interval")
            else:
                logger.warning(f"❌ No data received for {interval} interval")
            
            # Rate limit protection
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Failed to fetch {interval} data: {e}")
            continue
    
    # Verify the loaded data
    logger.info("Verifying loaded price history...")
    prices, volumes = data_manager.load_price_history()
    
    if prices and len(prices) > 0:
        logger.info(f"✅ Successfully loaded {len(prices)} price points")
        logger.info(f"📊 Price range: €{min(prices):.2f} to €{max(prices):.2f}")
        logger.info(f"🕒 Data coverage: {len(prices) * 15 / 60:.1f} hours")
        
        # Show recent data
        if len(prices) >= 5:
            logger.info("📈 Last 5 prices:")
            for i in range(-5, 0):
                logger.info(f"   €{prices[i]:.2f}")
        
        logger.info("🎉 Bootstrap completed successfully!")
        return True
    else:
        logger.error("❌ Bootstrap failed - no valid price data loaded")
        return False

def validate_api_connection():
    """Validate that the API connection is working"""
    logger.info("🔍 Validating API connection...")
    
    try:
        bitvavo = authenticate_exchange()
        
        # Test balance fetch
        balance = bitvavo.fetch_balance()
        logger.info(f"✅ Balance fetch successful: {balance.get('BTC', {}).get('total', 0):.8f} BTC")
        
        # Test current price fetch
        trade_executor = TradeExecutor(bitvavo)
        current_price, volume = trade_executor.fetch_current_price()
        
        if current_price:
            logger.info(f"✅ Current BTC price: €{current_price:.2f}")
            return True
        else:
            logger.error("❌ Failed to fetch current price")
            return False
            
    except Exception as e:
        logger.error(f"❌ API connection failed: {e}")
        return False

def main():
    """Main bootstrap function"""
    logger.info("="*50)
    logger.info("TRADING BOT BOOTSTRAP")
    logger.info("="*50)
    
    # Step 1: Validate API connection
    if not validate_api_connection():
        logger.error("Bootstrap failed - API connection issues")
        return False
    
    # Step 2: Bootstrap historical data
    if not bootstrap_historical_data():
        logger.error("Bootstrap failed - could not load historical data")
        return False
    
    logger.info("="*50)
    logger.info("🎉 BOOTSTRAP COMPLETE!")
    logger.info("You can now start the main trading bot with: python3 main.py")
    logger.info("="*50)
    
    return True

if __name__ == "__main__":
    main()