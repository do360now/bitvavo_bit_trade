# debug_bitvavo_api.py
import asyncio
import os
from dotenv import load_dotenv
from bitvavo_api_async import AsyncBitvavoAPI

async def debug_api():
    """Debug Bitvavo API connectivity and permissions."""
    
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv('BITVAVO_API_KEY')
    api_secret = os.getenv('BITVAVO_API_SECRET')
    
    print(f"API Key: {api_key[:8]}...{api_key[-4:] if api_key else 'NOT FOUND'}")
    print(f"API Secret: {'✓' if api_secret else '✗'} {'Present' if api_secret else 'Missing'}")
    
    if not api_key or not api_secret:
        print("❌ API credentials not found in .env file")
        return
    
    # Test API connectivity
    async with AsyncBitvavoAPI(api_key, api_secret) as api:
        
        # Test 1: Public endpoint (no authentication)
        print("\n🔍 Testing public endpoints...")
        try:
            ticker = await api.get_ticker("BTC-EUR")
            print(f"✅ Public API works - BTC price: €{ticker['price']}")
        except Exception as e:
            print(f"❌ Public API failed: {e}")
            return
        
        # Test 2: Authentication test with time endpoint
        print("\n🔍 Testing authentication...")
        try:
            # Try to get server time first (if available)
            response = await api._make_request("GET", "time")
            print(f"✅ Server time: {response}")
        except Exception as e:
            print(f"ℹ️  Server time endpoint: {e}")
        
        # Test 3: Balance endpoint (requires authentication)
        print("\n🔍 Testing balance endpoint...")
        try:
            balance = await api.get_balance()
            print("✅ Authentication successful!")
            print("Account balances:")
            for asset in balance:
                if float(asset['available']) > 0:
                    print(f"  {asset['symbol']}: {asset['available']}")
        except Exception as e:
            print(f"❌ Balance endpoint failed: {e}")
            
            # Check if it's a permission issue
            if "403" in str(e):
                print("\n💡 This is likely a permissions issue. Check:")
                print("   1. API key is activated on Bitvavo")
                print("   2. API key has 'View' permissions enabled")
                print("   3. API key has 'Trade' permissions if you plan to trade")
                print("   4. Your account is verified and not restricted")
        
        # Test 4: Check API key permissions
        print("\n🔍 Testing API permissions...")
        try:
            # Try to get open orders (another authenticated endpoint)
            orders = await api.get_open_orders()
            print(f"✅ Can access orders (found {len(orders)} open orders)")
        except Exception as e:
            print(f"❌ Cannot access orders: {e}")

if __name__ == "__main__":
    asyncio.run(debug_api())