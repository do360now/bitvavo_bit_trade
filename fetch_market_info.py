#!/usr/bin/env python3
"""
Fetch Bitvavo Market Info for BTC-EUR
Determines the correct tick size and precision requirements
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bitvavo_api import EnhancedBitvavoAPI, load_api_credentials


async def get_market_info():
    """Fetch and display BTC-EUR market specifications"""
    print("\n" + "=" * 70)
    print("BITVAVO BTC-EUR MARKET SPECIFICATIONS")
    print("=" * 70)
    
    try:
        api_key, api_secret = load_api_credentials()
        api = EnhancedBitvavoAPI(api_key, api_secret)
        
        # Fetch market info
        response = await api._make_request("GET", "markets", {"market": "BTC-EUR"})
        
        if response:
            market_info = response[0] if isinstance(response, list) else response
            
            print(f"\nMarket: {market_info.get('market')}")
            print(f"Status: {market_info.get('status')}")
            print("\n--- PRICE SETTINGS ---")
            print(f"Tick Size: {market_info.get('tickSize', 'N/A')}")
            print(f"Price Precision: {market_info.get('pricePrecision', 'N/A')}")
            
            print("\n--- AMOUNT SETTINGS ---")
            print(f"Min Order in Base Asset: {market_info.get('minOrderInBaseAsset', 'N/A')}")
            print(f"Min Order in Quote Asset: €{market_info.get('minOrderInQuoteAsset', 'N/A')}")
            print(f"Max Order in Base Asset: {market_info.get('maxOrderInBaseAsset', 'N/A')}")
            print(f"Max Order in Quote Asset: €{market_info.get('maxOrderInQuoteAsset', 'N/A')}")
            
            print("\n--- PRECISION ---")
            print(f"Base Precision: {market_info.get('baseDecimals', 'N/A')} decimals")
            print(f"Quote Precision: {market_info.get('quoteDecimals', 'N/A')} decimals")
            
            # Calculate example valid prices
            tick_size = float(market_info.get('tickSize', 1))
            current_price = 77800.0
            
            print("\n--- EXAMPLE VALID PRICES @ €77,800 ---")
            print(f"Tick Size: €{tick_size}")
            
            # Round to nearest tick
            rounded_price = round(current_price / tick_size) * tick_size
            print(f"Valid price (rounded): €{rounded_price:,.2f}")
            print(f"Next tick up: €{rounded_price + tick_size:,.2f}")
            print(f"Next tick down: €{rounded_price - tick_size:,.2f}")
            
            # Show why your order failed
            your_price = 77800.12
            print(f"\n--- YOUR ORDER ---")
            print(f"Your price: €{your_price:,.2f}")
            print(f"Is valid? {(your_price % tick_size == 0)}")
            if your_price % tick_size != 0:
                print(f"❌ NOT a multiple of tick size (€{tick_size})")
                print(f"✅ Should be: €{round(your_price / tick_size) * tick_size:,.2f}")
            
            print("\n" + "=" * 70)
            print("FULL MARKET INFO:")
            print("=" * 70)
            for key, value in market_info.items():
                print(f"{key}: {value}")
            print("=" * 70)
            
        else:
            print("❌ Failed to fetch market info")
        
        await api.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(get_market_info())