#!/usr/bin/env python3
"""
Reconcile Bot State from Bitvavo Exchange History
Gets actual trades from exchange and calculates true average buy price
"""

from bitvavo_api import EnhancedBitvavoAPI
import json
from datetime import datetime, timedelta

def reconcile_from_exchange():
    """Query exchange and rebuild accurate state"""
    
    print("=" * 70)
    print("  BITVAVO STATE RECONCILIATION")
    print("=" * 70)
    print()
    
    # Initialize API
    print("🔌 Connecting to Bitvavo...")
    api = EnhancedBitvavoAPI()
    print("✅ Connected!\n")
    
    # Get current balance
    print("📊 Fetching current balances...")
    balances = api.exchange.fetch_balance()
    
    btc_balance = float(balances['BTC']['total']) if 'BTC' in balances else 0.0
    eur_balance = float(balances['EUR']['free']) if 'EUR' in balances else 0.0
    
    print(f"   BTC Balance: {btc_balance:.8f} BTC")
    print(f"   EUR Balance: €{eur_balance:.2f}")
    print()
    
    # Get trade history (all filled orders)
    print("📜 Fetching trade history...")
    
    # Get last 90 days of trades (should cover all bot activity)
    since = int((datetime.now() - timedelta(days=90)).timestamp() * 1000)
    trades = api.exchange.fetch_my_trades('BTC/EUR', since=since, limit=1000)
    
    print(f"   Found {len(trades)} trades in last 90 days")
    print()
    
    # Separate buys and sells
    buy_trades = [t for t in trades if t['side'] == 'buy']
    sell_trades = [t for t in trades if t['side'] == 'sell']
    
    print(f"   Buy trades: {len(buy_trades)}")
    print(f"   Sell trades: {len(sell_trades)}")
    print()
    
    # Calculate total BTC bought and cost
    total_btc_bought = 0.0
    total_eur_spent = 0.0
    total_fees_eur = 0.0
    
    print("📈 BUY TRADES:")
    print("-" * 70)
    
    for trade in buy_trades:
        btc_amount = float(trade['amount'])
        eur_cost = float(trade['cost'])
        fee = float(trade['fee']['cost']) if trade.get('fee') else 0.0
        price = float(trade['price'])
        timestamp = datetime.fromtimestamp(trade['timestamp'] / 1000)
        
        total_btc_bought += btc_amount
        total_eur_spent += eur_cost
        total_fees_eur += fee
        
        print(f"{timestamp.strftime('%Y-%m-%d %H:%M')} | "
              f"{btc_amount:.8f} BTC @ €{price:,.2f} | "
              f"Cost: €{eur_cost:.2f} | Fee: €{fee:.4f}")
    
    print("-" * 70)
    print(f"TOTAL BOUGHT: {total_btc_bought:.8f} BTC for €{total_eur_spent:.2f}")
    print(f"TOTAL FEES: €{total_fees_eur:.2f}")
    print()
    
    # Calculate average buy price (including fees)
    if total_btc_bought > 0:
        avg_buy_price_no_fees = total_eur_spent / total_btc_bought
        avg_buy_price_with_fees = (total_eur_spent + total_fees_eur) / total_btc_bought
        
        print("💰 CALCULATED AVERAGE BUY PRICE:")
        print(f"   Without fees: €{avg_buy_price_no_fees:,.2f}")
        print(f"   With fees:    €{avg_buy_price_with_fees:,.2f}")
    else:
        avg_buy_price_with_fees = 0.0
        print("⚠️  No buy trades found")
    
    print()
    
    # Show sells for reference
    if sell_trades:
        total_btc_sold = sum(float(t['amount']) for t in sell_trades)
        total_eur_received = sum(float(t['cost']) for t in sell_trades)
        
        print("📉 SELL TRADES:")
        print("-" * 70)
        for trade in sell_trades:
            btc_amount = float(trade['amount'])
            eur_received = float(trade['cost'])
            price = float(trade['price'])
            timestamp = datetime.fromtimestamp(trade['timestamp'] / 1000)
            
            print(f"{timestamp.strftime('%Y-%m-%d %H:%M')} | "
                  f"{btc_amount:.8f} BTC @ €{price:,.2f} | "
                  f"Received: €{eur_received:.2f}")
        
        print("-" * 70)
        print(f"TOTAL SOLD: {total_btc_sold:.8f} BTC for €{total_eur_received:.2f}")
        print()
    
    # Net position
    net_btc = total_btc_bought - (sum(float(t['amount']) for t in sell_trades) if sell_trades else 0)
    
    print("=" * 70)
    print("  RECONCILIATION SUMMARY")
    print("=" * 70)
    print()
    
    print("📊 Exchange Data (Source of Truth):")
    print(f"   Current BTC Balance: {btc_balance:.8f} BTC")
    print(f"   Total BTC Bought:    {total_btc_bought:.8f} BTC")
    print(f"   Net Position:        {net_btc:.8f} BTC")
    print(f"   Average Buy Price:   €{avg_buy_price_with_fees:,.2f} (with fees)")
    print(f"   Total Fees Paid:     €{total_fees_eur:.2f}")
    print()
    
    # Load current bot state
    try:
        with open('bot_state.json', 'r') as f:
            bot_state = json.load(f)
        
        print("📝 Current Bot State:")
        print(f"   Average Buy Price:   €{bot_state.get('avg_buy_price', 0):,.2f}")
        print(f"   Total BTC Bought:    {bot_state.get('total_btc_bought', 0):.8f}")
        print()
        
        # Check for discrepancies
        btc_diff = abs(bot_state.get('total_btc_bought', 0) - total_btc_bought)
        price_diff = abs(bot_state.get('avg_buy_price', 0) - avg_buy_price_with_fees)
        
        if btc_diff > 0.00001 or price_diff > 1.0:
            print("⚠️  DISCREPANCY DETECTED!")
            print(f"   BTC difference:   {btc_diff:.8f} BTC")
            print(f"   Price difference: €{price_diff:.2f}")
            print()
            
            # Offer to update
            response = input("Update bot_state.json with exchange data? (y/n): ")
            
            if response.lower() == 'y':
                # Backup first
                import shutil
                backup_file = f"bot_state.json.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copy('bot_state.json', backup_file)
                print(f"   📦 Backup saved: {backup_file}")
                
                # Update state
                bot_state['avg_buy_price'] = avg_buy_price_with_fees
                bot_state['total_btc_bought'] = total_btc_bought
                bot_state['total_fees_eur'] = total_fees_eur
                bot_state['last_updated'] = datetime.now().isoformat()
                
                with open('bot_state.json', 'w') as f:
                    json.dump(bot_state, f, indent=2)
                
                print("   ✅ bot_state.json updated with exchange data!")
            else:
                print("   ⏭️  Skipped update")
        else:
            print("✅ Bot state matches exchange data!")
        
    except FileNotFoundError:
        print("❌ No bot_state.json found")
        
        response = input("\nCreate new bot_state.json from exchange data? (y/n): ")
        if response.lower() == 'y':
            new_state = {
                'avg_buy_price': avg_buy_price_with_fees,
                'total_btc_bought': total_btc_bought,
                'total_fees_eur': total_fees_eur,
                'peak_price': 0.0,
                'peak_timestamp': 0,
                'last_buy_price': 0.0,
                'last_buy_timestamp': 0,
                'last_sell_price': 0.0,
                'last_sell_timestamp': 0,
                'total_round_trips': 0,
                'successful_round_trips': 0,
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
            }
            
            with open('bot_state.json', 'w') as f:
                json.dump(new_state, f, indent=2)
            
            print("   ✅ Created bot_state.json from exchange data!")
    
    print()
    print("=" * 70)
    print("✅ Reconciliation complete!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        reconcile_from_exchange()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()