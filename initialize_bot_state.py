#!/usr/bin/env python3
"""
Initialize Bot State - One-time setup to create bot_state.json

This script helps you set up the initial bot state from either:
1. Manual entry of your current position
2. Analysis of your performance_history.json
3. Analysis of your bot_logs.csv

Run this ONCE before switching to the new bot version.
"""

import json
import os
import sys
import pandas as pd
from datetime import datetime


def manual_entry():
    """Manually enter your current position"""
    print("\n" + "=" * 60)
    print("MANUAL STATE ENTRY")
    print("=" * 60)
    print("\nEnter your current bot position:")
    
    try:
        avg_buy = float(input("Average buy price (EUR): "))
        total_btc = float(input("Total BTC bought: "))
        
        state = {
            'avg_buy_price': avg_buy,
            'total_btc_bought': total_btc,
            'peak_price': 0.0,
            'peak_timestamp': 0,
            'last_buy_price': avg_buy,
            'last_buy_timestamp': datetime.now().timestamp(),
            'last_sell_price': 0.0,
            'last_sell_timestamp': 0,
            'total_round_trips': 0,
            'successful_round_trips': 0,
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
        }
        
        return state
        
    except ValueError as e:
        print(f"Invalid input: {e}")
        return None


def from_performance_history():
    """Calculate state from performance_history.json"""
    print("\n" + "=" * 60)
    print("ANALYZING PERFORMANCE HISTORY")
    print("=" * 60)
    
    if not os.path.exists("./performance_history.json"):
        print("❌ No performance_history.json found")
        return None
    
    try:
        with open("./performance_history.json", 'r') as f:
            data = json.load(f)
        
        trades = data.get('trades', [])
        
        if not trades:
            print("❌ No trades found in performance history")
            return None
        
        # Calculate average buy price from buy trades
        buy_trades = [t for t in trades if t['side'] == 'buy']
        
        if not buy_trades:
            print("❌ No buy trades found")
            return None
        
        total_cost = sum(t['volume'] * t['price'] for t in buy_trades)
        total_volume = sum(t['volume'] for t in buy_trades)
        avg_buy = total_cost / total_volume if total_volume > 0 else 0
        
        # Find last buy and sell
        sorted_trades = sorted(trades, key=lambda t: t.get('timestamp', 0))
        last_buy = next((t for t in reversed(sorted_trades) if t['side'] == 'buy'), None)
        last_sell = next((t for t in reversed(sorted_trades) if t['side'] == 'sell'), None)
        
        # Count round trips
        sell_trades = [t for t in trades if t['side'] == 'sell']
        
        state = {
            'avg_buy_price': avg_buy,
            'total_btc_bought': total_volume,
            'peak_price': 0.0,
            'peak_timestamp': 0,
            'last_buy_price': last_buy['price'] if last_buy else 0,
            'last_buy_timestamp': last_buy.get('timestamp', 0) if last_buy else 0,
            'last_sell_price': last_sell['price'] if last_sell else 0,
            'last_sell_timestamp': last_sell.get('timestamp', 0) if last_sell else 0,
            'total_round_trips': len(sell_trades),
            'successful_round_trips': len([s for s in sell_trades if s['price'] > avg_buy]),
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
        }
        
        print(f"\n✅ Analyzed {len(trades)} trades:")
        print(f"   - Buy trades: {len(buy_trades)}")
        print(f"   - Sell trades: {len(sell_trades)}")
        print(f"   - Avg buy price: €{avg_buy:,.2f}")
        print(f"   - Total BTC bought: {total_volume:.8f}")
        
        return state
        
    except Exception as e:
        print(f"❌ Error analyzing performance history: {e}")
        import traceback
        traceback.print_exc()
        return None


def from_bot_logs():
    """Calculate state from bot_logs.csv"""
    print("\n" + "=" * 60)
    print("ANALYZING BOT LOGS")
    print("=" * 60)
    
    if not os.path.exists("./bot_logs.csv"):
        print("❌ No bot_logs.csv found")
        return None
    
    try:
        df = pd.read_csv("./bot_logs.csv", encoding='utf-8', on_bad_lines='skip')
        
        # Filter for actual trades (not just strategy logs)
        buy_trades = df[
            (df['side'] == 'buy') & 
            (df['trade_volume'].notna()) & 
            (df['trade_volume'] > 0)
        ]
        
        if buy_trades.empty:
            print("❌ No buy trades found in logs")
            return None
        
        # Calculate weighted average buy price
        buy_trades['total_cost'] = buy_trades['price'] * buy_trades['trade_volume']
        total_cost = buy_trades['total_cost'].sum()
        total_volume = buy_trades['trade_volume'].sum()
        avg_buy = total_cost / total_volume if total_volume > 0 else 0
        
        # Count sells
        sell_trades = df[
            (df['side'] == 'sell') & 
            (df['trade_volume'].notna()) & 
            (df['trade_volume'] > 0)
        ]
        
        state = {
            'avg_buy_price': avg_buy,
            'total_btc_bought': total_volume,
            'peak_price': 0.0,
            'peak_timestamp': 0,
            'last_buy_price': buy_trades.iloc[-1]['price'] if not buy_trades.empty else 0,
            'last_buy_timestamp': 0,
            'last_sell_price': sell_trades.iloc[-1]['price'] if not sell_trades.empty else 0,
            'last_sell_timestamp': 0,
            'total_round_trips': len(sell_trades),
            'successful_round_trips': len(sell_trades[sell_trades['price'] > avg_buy]),
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
        }
        
        print(f"\n✅ Analyzed bot logs:")
        print(f"   - Buy trades: {len(buy_trades)}")
        print(f"   - Sell trades: {len(sell_trades)}")
        print(f"   - Avg buy price: €{avg_buy:,.2f}")
        print(f"   - Total BTC bought: {total_volume:.8f}")
        
        return state
        
    except Exception as e:
        print(f"❌ Error analyzing bot logs: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_state(state):
    """Save state to bot_state.json"""
    if not state:
        return False
    
    try:
        # Backup existing state if present
        if os.path.exists("./bot_state.json"):
            backup = f"./bot_state.json.backup.{int(datetime.now().timestamp())}"
            os.rename("./bot_state.json", backup)
            print(f"📦 Backed up existing state to {backup}")
        
        with open("./bot_state.json", 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"\n✅ State saved to bot_state.json")
        print("\n" + "=" * 60)
        print("STATE SUMMARY")
        print("=" * 60)
        print(f"Average Buy Price: €{state['avg_buy_price']:,.2f}")
        print(f"Total BTC Bought: {state['total_btc_bought']:.8f}")
        print(f"Round Trips: {state['total_round_trips']} ({state['successful_round_trips']} profitable)")
        print("=" * 60)
        print("\n🚀 You can now start the updated bot!")
        print("   It will load this state on startup.\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving state: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("BOT STATE INITIALIZATION")
    print("=" * 60)
    print("\nThis will create bot_state.json for the updated bot.")
    print("\nChoose initialization method:")
    print("  1. Manual entry (you know your avg buy price)")
    print("  2. From performance_history.json (if you have it)")
    print("  3. From bot_logs.csv (analyze trade logs)")
    print("  4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    state = None
    
    if choice == "1":
        state = manual_entry()
    elif choice == "2":
        state = from_performance_history()
    elif choice == "3":
        state = from_bot_logs()
    elif choice == "4":
        print("Exiting...")
        return
    else:
        print("Invalid choice")
        return
    
    if state:
        if save_state(state):
            sys.exit(0)
        else:
            print("Failed to save state")
            sys.exit(1)
    else:
        print("\n❌ Could not create state")
        print("Try manual entry or check your files")
        sys.exit(1)


if __name__ == "__main__":
    main()
