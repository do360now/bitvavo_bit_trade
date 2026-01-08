# comprehensive_order_debug.py
import asyncio
import os
import json
import time
import aiohttp
from dotenv import load_dotenv
from bitvavo_api_async import AsyncBitvavoAPI

async def test_order_formats():
    """Test different order formats and headers."""
    
    load_dotenv()
    api_key = os.getenv('BITVAVO_API_KEY')
    api_secret = os.getenv('BITVAVO_API_SECRET')
    
    async with AsyncBitvavoAPI(api_key, api_secret) as api:
        
        # Get current price for calculations
        ticker = await api.get_ticker("BTC-EUR")
        current_price = float(ticker["price"])
        print(f"Current BTC price: €{current_price}")
        
        # Test 1: Proper decimal formatting
        print("\n🧪 Test 1: Proper decimal formatting")
        test_data = {
            "market": "BTC-EUR",
            "side": "buy",
            "orderType": "limit",
            "amount": "0.00010000",  # Proper decimal format, above minimum
            "price": f"{current_price * 0.99:.2f}",  # 1% below market, 2 decimals
            "timeInForce": "GTC"
        }
        print(f"Parameters: {json.dumps(test_data, indent=2)}")
        
        try:
            result = await api._make_request("POST", "order", data=test_data, signed=True)
            print(f"✅ Success: {result}")
            
            # Cancel the order immediately if successful
            try:
                if result.get('orderId'):
                    cancel_result = await api.cancel_order(result['orderId'], "BTC-EUR")
                    print(f"✅ Order cancelled: {cancel_result}")
            except:
                pass
                
        except Exception as e:
            print(f"❌ Failed: {e}")
        
        # Test 2: Market order with EUR amount
        print("\n🧪 Test 2: Market buy with EUR amount")
        test_data = {
            "market": "BTC-EUR",
            "side": "buy",
            "orderType": "market",
            "amountQuote": "10.00"  # €10 worth
        }
        print(f"Parameters: {json.dumps(test_data, indent=2)}")
        
        try:
            result = await api._make_request("POST", "order", data=test_data, signed=True)
            print(f"✅ Success: {result}")
        except Exception as e:
            print(f"❌ Failed: {e}")
        
        # Test 3: Market order with BTC amount
        print("\n🧪 Test 3: Market buy with BTC amount")
        test_data = {
            "market": "BTC-EUR",
            "side": "buy", 
            "orderType": "market",
            "amount": "0.00010000"  # Specific BTC amount
        }
        print(f"Parameters: {json.dumps(test_data, indent=2)}")
        
        try:
            result = await api._make_request("POST", "order", data=test_data, signed=True)
            print(f"✅ Success: {result}")
        except Exception as e:
            print(f"❌ Failed: {e}")
            
        # Test 4: Check what happens with raw HTTP request
        print("\n🧪 Test 4: Manual HTTP request inspection")
        try:
            # Test with minimal valid order
            timestamp = str(int(time.time() * 1000))
            test_body = json.dumps({
                "market": "BTC-EUR",
                "side": "buy",
                "orderType": "limit", 
                "amount": "0.00010000",
                "price": f"{current_price * 0.99:.2f}"
            }, separators=(',', ':'))
            
            signature = api._generate_signature(timestamp, "POST", "/v2/order", test_body)
            
            headers = {
                'Bitvavo-Access-Key': api.api_key,
                'Bitvavo-Access-Signature': signature,
                'Bitvavo-Access-Timestamp': timestamp,
                'Content-Type': 'application/json'
            }
            
            print(f"Headers: {json.dumps(headers, indent=2)}")
            print(f"Body: {test_body}")
            
            # Make raw request to see exact error
            async with api.session.post(
                f"{api.base_url}/order",
                data=test_body,
                headers=headers
            ) as response:
                response_text = await response.text()
                print(f"Status: {response.status}")
                print(f"Response: {response_text}")
                
        except Exception as e:
            print(f"Manual request error: {e}")

if __name__ == "__main__":
    import time
    asyncio.run(test_order_formats())