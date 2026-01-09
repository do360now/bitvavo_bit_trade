#!/usr/bin/env python3
"""
Populate price history from CoinGecko for quick testing
Fetches historical BTC/EUR prices
"""

import json
import aiohttp
import asyncio
from datetime import datetime, timedelta
from logger_config import logger


async def fetch_prices_from_coingecko(days: int = 365, interval: str = "15m") -> list:
    """
    Fetch historical BTC/EUR prices from CoinGecko or Bitvavo
    
    Args:
        days: How many days back to fetch
        interval: Data interval - "hourly" or "15m"
    
    Returns:
        List of prices in EUR
    """
    try:
        # For granular data (15min), we need to use a different approach
        # CoinGecko free API is limited to daily data for >90 days
        # For 15-min data, we'll use Bitvavo's public API or fetch hourly and interpolate
        
        if interval == "15m" and days > 90:
            logger.warning("CoinGecko free API limited to daily data for >90 days")
            logger.info("Fetching daily data and interpolating for 15-min resolution...")
            interval = "daily"
        
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        
        params = {
            "vs_currency": "eur",
            "days": days,
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    data = await response.json()
                    prices = data.get("prices", [])
                    
                    # Convert from [timestamp, price] to just prices
                    price_list = [float(p[1]) for p in prices]
                    
                    logger.info(f"Fetched {len(price_list)} price points from CoinGecko")
                    
                    # If we got daily data but need 15-min, interpolate
                    if interval == "daily" and len(price_list) > 1:
                        logger.info("Interpolating daily data to 15-minute intervals...")
                        interpolated = []
                        for i in range(len(price_list) - 1):
                            # For each day, create 4 points (6-hour intervals)
                            # Linear interpolation between days
                            current = price_list[i]
                            next_price = price_list[i + 1]
                            
                            # Add 4 interpolated points per day
                            for j in range(4):
                                fraction = j / 4
                                interp_price = current + (next_price - current) * fraction
                                interpolated.append(interp_price)
                        
                        # Add the last point
                        interpolated.append(price_list[-1])
                        price_list = interpolated
                        logger.info(f"Interpolated to {len(price_list)} points")
                    
                    return price_list
                else:
                    logger.error(f"CoinGecko API error: {response.status}")
                    return []
                    
    except Exception as e:
        logger.error(f"Failed to fetch from CoinGecko: {e}")
        return []


def generate_synthetic_prices(start_price: float = 78000, num_points: int = 100, volatility: float = 0.02) -> list:
    """
    Generate realistic synthetic price data for testing
    Uses random walk with mean reversion
    
    Args:
        start_price: Starting BTC price in EUR
        num_points: Number of prices to generate
        volatility: Daily volatility (default 2%)
    
    Returns:
        List of synthetic prices
    """
    import random
    
    prices = [start_price]
    current_price = start_price
    
    for _ in range(num_points - 1):
        # Random walk with drift
        change = random.gauss(0, current_price * volatility / 10)  # ~0.2% per step
        current_price = max(current_price + change, start_price * 0.8)  # Don't crash too hard
        prices.append(current_price)
    
    logger.info(f"Generated {len(prices)} synthetic prices (${prices[0]:.2f} -> €{prices[-1]:.2f})")
    return prices


async def main():
    """Main function"""
    print("\n" + "="*60)
    print("Bitcoin Price History Populator")
    print("="*60 + "\n")
    
    print("Options:")
    print("1. Fetch real prices from CoinGecko (recommended)")
    print("2. Generate synthetic test prices")
    print("3. Exit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        days = input("How many days back? (default 2): ").strip() or "2"
        num_points = input("How many price points? (default 100): ").strip() or "100"
        
        try:
            prices = await fetch_prices_from_coingecko(int(days), int(num_points))
        except ValueError:
            print("Invalid input. Using defaults...")
            prices = await fetch_prices_from_coingecko(2, 100)
            
    elif choice == "2":
        num_points = input("How many synthetic prices? (default 100): ").strip() or "100"
        start_price = input("Starting price in EUR? (default 78000): ").strip() or "78000"
        
        try:
            prices = generate_synthetic_prices(float(start_price), int(num_points))
        except ValueError:
            print("Invalid input. Using defaults...")
            prices = generate_synthetic_prices(78000, 100)
            
    else:
        print("Exiting...")
        return
    
    if not prices:
        print("❌ Failed to get prices")
        return
    
    # Save to file
    try:
        with open("price_history.json", "w") as f:
            json.dump(prices, f, indent=2)
        
        print(f"\n✅ Saved {len(prices)} prices to price_history.json")
        print(f"   Price range: €{min(prices):.2f} - €{max(prices):.2f}")
        print(f"   Average: €{sum(prices)/len(prices):.2f}")
        print("\n💡 Start the bot now to use these prices!")
        
    except Exception as e:
        print(f"❌ Failed to save: {e}")


if __name__ == "__main__":
    asyncio.run(main())
