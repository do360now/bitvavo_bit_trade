#!/usr/bin/env python3
"""
Fix Stuck Pending Orders
Processes orders showing "closed" status but stuck in pending
"""

import json
import sys
import os

def fix_pending_orders():
    """Manually process stuck pending orders"""
    
    print("=" * 70)
    print("  PENDING ORDERS FIX SCRIPT")
    print("=" * 70)
    print()
    
    # Check if files exist
    if not os.path.exists("pending_updates.json"):
        print("❌ No pending_updates.json found")
        print("   This is normal if you haven't used AtomicTradeManager yet")
        return
    
    # Load pending updates
    with open("pending_updates.json", "r") as f:
        pending_updates = json.load(f)
    
    if not pending_updates:
        print("✅ No pending updates to process")
        return
    
    print(f"Found {len(pending_updates)} pending updates\n")
    
    # Load state
    try:
        with open("bot_state.json", "r") as f:
            state = json.load(f)
    except FileNotFoundError:
        print("❌ No bot_state.json found")
        return
    
    # Import after checking files exist
    try:
        from atomic_trade_manager import AtomicTradeManager
        from order_manager import OrderManager
        from bot_state_manager import BotStateManager
        from bitvavo_api import EnhancedBitvavoAPI
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you're running from the project root directory")
        return
    
    # Initialize components
    print("Initializing components...")
    api = EnhancedBitvavoAPI()
    state_mgr = BotStateManager()
    order_mgr = OrderManager(api)
    atomic = AtomicTradeManager(order_mgr, state_mgr, None)
    
    print(f"Current state:")
    print(f"  Avg Buy: €{state_mgr.get_avg_buy_price():,.2f}")
    print(f"  Total BTC: {state_mgr.get_total_btc_bought():.8f}")
    print(f"  Total Fees: €{state_mgr.state.get('total_fees_eur', 0):.2f}")
    print()
    
    processed = 0
    failed = 0
    
    for pid, update in list(atomic.pending_updates.items()):
        order_id = update.get('order_id')
        
        print("-" * 70)
        print(f"Pending Update: {pid}")
        print(f"  Order ID: {order_id}")
        print(f"  Type: {update.get('type').upper()}")
        print(f"  Expected Price: €{update.get('price'):,.2f}")
        print(f"  Expected Volume: {update.get('volume'):.8f} BTC")
        
        # Check order status
        try:
            # Try to get order info
            order_data = api.get_order_async(order_id)
            
            if not order_data:
                print(f"  ⚠️  Order not found in active orders")
                print(f"  → Checking order history...")
                
                # Check history
                order_data = api.get_order_from_history(order_id, lookback_days=30)
            
            if order_data:
                status = order_data.get('status', '').lower()
                filled_amount = float(order_data.get('filledAmount', 0))
                actual_price = float(order_data.get('price', 0))
                side = order_data.get('side', update.get('type'))
                
                print(f"  Found order!")
                print(f"    Status: {status}")
                print(f"    Filled: {filled_amount:.8f} BTC @ €{actual_price:,.2f}")
                
                # Check if filled/closed
                if status in ['filled', 'closed'] and filled_amount > 0:
                    print(f"  ✅ Processing as FILLED")
                    
                    # Apply state update
                    if side == 'buy':
                        state_mgr.update_buy(actual_price, filled_amount)
                        print(f"  ✅ Applied BUY state update")
                    elif side == 'sell':
                        state_mgr.update_sell(actual_price, filled_amount)
                        print(f"  ✅ Applied SELL state update")
                    else:
                        print(f"  ❌ Unknown side: {side}")
                        failed += 1
                        continue
                    
                    # Remove from pending
                    del atomic.pending_updates[pid]
                    atomic._save_pending()
                    processed += 1
                    
                elif status in ['canceled', 'cancelled']:
                    print(f"  ⚠️  Order was cancelled - removing from pending")
                    del atomic.pending_updates[pid]
                    atomic._save_pending()
                    processed += 1
                    
                else:
                    print(f"  ⏳ Order still pending (status: {status})")
                    
            else:
                print(f"  ❌ Order not found in active or history")
                print(f"     This might be a very old order (>30 days)")
                failed += 1
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed += 1
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Processed: {processed}")
    print(f"Failed: {failed}")
    print(f"Remaining pending: {len(atomic.pending_updates)}")
    print()
    
    # Show updated state
    print("Updated state:")
    print(f"  Avg Buy: €{state_mgr.get_avg_buy_price():,.2f}")
    print(f"  Total BTC: {state_mgr.get_total_btc_bought():.8f}")
    print(f"  Total Fees: €{state_mgr.state.get('total_fees_eur', 0):.2f}")
    print()
    
    if processed > 0:
        print("✅ Successfully processed stuck orders!")
        print("   You can now restart your bot.")
    elif len(atomic.pending_updates) == 0:
        print("✅ No pending updates remaining!")
    else:
        print("⚠️  Some orders are still pending")
        print("   They may not be filled yet - check your Bitvavo account")
    
    print("=" * 70)


if __name__ == "__main__":
    try:
        fix_pending_orders()
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)