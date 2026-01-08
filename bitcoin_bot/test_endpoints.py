# test_endpoints.py
import asyncio
import os
from dotenv import load_dotenv
from bitvavo_api_async import AsyncBitvavoAPI

async def test_all_endpoints():
    """Test all endpoints used by the trading bot."""
    
    load_dotenv()
    api_key = os.getenv('BITVAVO_API_KEY')
    api_secret = os.getenv('BITVAVO_API_SECRET')
    
    async with AsyncBitvavoAPI(api_key, api_secret) as api:
        
        # Test 1: Ticker endpoint (used in trading loop)
        print("🔍 Testing ticker endpoint...")
        try:
            ticker = await api.get_ticker("BTC-EUR")
            print(f"✅ Ticker works: {ticker}")
        except Exception as e:
            print(f"❌ Ticker failed: {e}")
        
        # Test 2: Order book endpoint (used in trading loop)
        print("\n🔍 Testing order book endpoint...")
        try:
            orderbook = await api.get_order_book("BTC-EUR", depth=10)
            print(f"✅ Order book works: bids={len(orderbook.get('bids', []))}, asks={len(orderbook.get('asks', []))}")
        except Exception as e:
            print(f"❌ Order book failed: {e}")
            
            # Try alternative endpoint structures
            print("  🔧 Trying alternative order book endpoints...")
            
            # Try market-specific endpoint
            try:
                result = await api._make_request("GET", "BTC-EUR/book", {"depth": 10})
                print(f"  ✅ Alternative 1 works: BTC-EUR/book")
            except Exception as e2:
                print(f"  ❌ Alternative 1 failed: {e2}")
            
            # Try different parameter structure
            try:
                result = await api._make_request("GET", "orderbook", {"market": "BTC-EUR", "depth": 10})
                print(f"  ✅ Alternative 2 works: orderbook")
            except Exception as e3:
                print(f"  ❌ Alternative 2 failed: {e3}")
        
        # Test 3: OHLCV/Candles endpoint (used in trading loop)
        print("\n🔍 Testing OHLCV endpoint...")
        try:
            ohlcv = await api.get_ohlcv("BTC-EUR", "1m", limit=5)
            print(f"✅ OHLCV works: {len(ohlcv)} candles")
        except Exception as e:
            print(f"❌ OHLCV failed: {e}")
            
            # Try alternative endpoint
            print("  🔧 Trying alternative OHLCV endpoints...")
            try:
                result = await api._make_request("GET", "klines", {
                    "market": "BTC-EUR", 
                    "interval": "1m", 
                    "limit": 5
                })
                print(f"  ✅ Alternative works: klines")
            except Exception as e2:
                print(f"  ❌ Alternative failed: {e2}")
        
        # Test 4: Manual endpoint discovery
        print("\n🔍 Testing common API endpoints...")
        common_endpoints = [
            "markets",
            "time", 
            "assets",
            "ticker",
            "ticker/24hr",
            "depth",
            "trades"
        ]
        
        for endpoint in common_endpoints:
            try:
                result = await api._make_request("GET", endpoint)
                print(f"  ✅ {endpoint}: Available")
            except Exception as e:
                if "404" in str(e):
                    print(f"  ❌ {endpoint}: Not found")
                else:
                    print(f"  ⚠️  {endpoint}: {e}")

if __name__ == "__main__":
    asyncio.run(test_all_endpoints())