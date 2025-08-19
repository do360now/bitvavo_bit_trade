#!/usr/bin/env python3
"""
Diagnostic script to identify and fix trading bot issues
Run this to check your bot's current state and identify problems
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta


def check_file_exists(filepath, description):
    """Check if a file exists and report its status"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
        print(
            f"✅ {description}: {filepath} (size: {size} bytes, modified: {mod_time})"
        )
        return True
    else:
        print(f"❌ {description}: {filepath} - FILE NOT FOUND")
        return False


def analyze_order_history():
    """Analyze order history file for issues"""
    print("\n🔍 ANALYZING ORDER HISTORY")
    print("=" * 50)

    if not check_file_exists("./order_history.json", "Order History"):
        return False

    try:
        with open("./order_history.json", "r") as f:
            order_data = json.load(f)

        filled_orders = order_data.get("filled_orders", {})
        cancelled_orders = order_data.get("cancelled_orders", {})
        last_updated = order_data.get("last_updated", 0)

        print(f"📊 Total filled orders: {len(filled_orders)}")
        print(f"❌ Total cancelled orders: {len(cancelled_orders)}")
        print(
            f"⏰ Last updated: {datetime.fromtimestamp(last_updated) if last_updated else 'Never'}"
        )

        # Analyze recent orders (last 24 hours)
        cutoff_time = datetime.now().timestamp() - (24 * 3600)
        recent_filled = []

        for order_id, order_info in filled_orders.items():
            filled_at = order_info.get("filled_at", order_info.get("timestamp", 0))
            if filled_at > cutoff_time:
                recent_filled.append(order_info)

        print(f"📈 Recent filled orders (24h): {len(recent_filled)}")

        if recent_filled:
            print("\n🔍 Recent filled orders:")
            for order in recent_filled[-5:]:  # Last 5
                side = order.get("side", "unknown")
                volume = order.get("executed_volume", order.get("volume", 0))
                price = order.get("average_price", order.get("price", 0))
                filled_time = datetime.fromtimestamp(order.get("filled_at", 0))
                print(
                    f"  {side.upper()}: {volume:.8f} BTC @ €{price:.2f} at {filled_time.strftime('%H:%M:%S')}"
                )

        return True

    except Exception as e:
        print(f"❌ Error analyzing order history: {e}")
        return False


def analyze_trade_session():
    """Analyze current trade session"""
    print("\n🔍 ANALYZING TRADE SESSION")
    print("=" * 50)

    if not check_file_exists("./trade_session.json", "Trade Session"):
        print("ℹ️ No trade session file found - this is normal for first run")
        return True

    try:
        with open("./trade_session.json", "r") as f:
            session_data = json.load(f)

        session_date = session_data.get("date", "1970-01-01")
        daily_trades = session_data.get("daily_trade_count", 0)
        last_updated = session_data.get("last_updated", "Never")

        print(f"📅 Session date: {session_date}")
        print(f"📊 Daily trade count: {daily_trades}")
        print(f"⏰ Last updated: {last_updated}")

        # Check if it's current date
        current_date = datetime.now().date().isoformat()
        if session_date != current_date:
            print("⚠️ Session date is old - will reset on next bot run")

        return True

    except Exception as e:
        print(f"❌ Error analyzing trade session: {e}")
        return False


def analyze_bot_logs():
    """Analyze bot logs for trading activity"""
    print("\n🔍 ANALYZING BOT LOGS")
    print("=" * 50)

    if not check_file_exists("./bot_logs.csv", "Bot Logs"):
        return False

    try:
        df = pd.read_csv("./bot_logs.csv")
        print(f"📊 Total log entries: {len(df)}")

        # Check for actual trades
        buy_trades = df[df["buy_decision"].astype(str).str.lower().isin(["true", "1"])]
        sell_trades = df[
            df["sell_decision"].astype(str).str.lower().isin(["true", "1"])
        ]

        print(f"🟢 Buy decisions: {len(buy_trades)}")
        print(f"🔴 Sell decisions: {len(sell_trades)}")

        # Check recent activity (last 24 hours)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_logs = df[df["timestamp"] > cutoff_time]

        print(f"📈 Recent log entries (24h): {len(recent_logs)}")

        if len(recent_logs) > 0:
            print(
                f"📊 Recent price range: €{recent_logs['price'].min():.0f} - €{recent_logs['price'].max():.0f}"
            )

            recent_buys = recent_logs[
                recent_logs["buy_decision"].astype(str).str.lower().isin(["true", "1"])
            ]
            recent_sells = recent_logs[
                recent_logs["sell_decision"].astype(str).str.lower().isin(["true", "1"])
            ]

            print(f"🟢 Recent buys (24h): {len(recent_buys)}")
            print(f"🔴 Recent sells (24h): {len(recent_sells)}")

            if len(recent_buys) > 0 or len(recent_sells) > 0:
                print("\n📝 Recent trading activity:")
                recent_trades = recent_logs[
                    (
                        recent_logs["buy_decision"]
                        .astype(str)
                        .str.lower()
                        .isin(["true", "1"])
                    )
                    | (
                        recent_logs["sell_decision"]
                        .astype(str)
                        .str.lower()
                        .isin(["true", "1"])
                    )
                ]

                for _, trade in recent_trades.tail(5).iterrows():
                    action = "BUY" if trade["buy_decision"] else "SELL"
                    timestamp = trade["timestamp"].strftime("%H:%M:%S")
                    price = trade["price"]
                    volume = trade.get("trade_volume", 0)
                    print(f"  {action}: {volume:.8f} BTC @ €{price:.2f} at {timestamp}")

        return True

    except Exception as e:
        print(f"❌ Error analyzing bot logs: {e}")
        return False


def analyze_recent_buys():
    """Analyze recent buys tracking"""
    print("\n🔍 ANALYZING RECENT BUYS")
    print("=" * 50)

    if not check_file_exists("./recent_buys.json", "Recent Buys"):
        print("ℹ️ No recent buys file found - bot hasn't made any purchases yet")
        return True

    try:
        with open("./recent_buys.json", "r") as f:
            recent_buys = json.load(f)

        print(f"📊 Recent buy entries: {len(recent_buys)}")

        if recent_buys:
            total_volume = sum(buy[1] for buy in recent_buys)
            total_cost = sum(buy[0] * buy[1] for buy in recent_buys)
            avg_price = total_cost / total_volume if total_volume > 0 else 0

            print(f"💰 Total volume: {total_volume:.8f} BTC")
            print(f"💰 Total cost: €{total_cost:.2f}")
            print(f"💰 Average buy price: €{avg_price:.2f}")

            print("\n📝 Recent buy details:")
            for i, (price, volume) in enumerate(recent_buys[-5:], 1):
                print(f"  {i}. {volume:.8f} BTC @ €{price:.2f}")

        return True

    except Exception as e:
        print(f"❌ Error analyzing recent buys: {e}")
        return False


def test_api_connectivity():
    """Test API connectivity and basic functions"""
    print("\n🔍 TESTING API CONNECTIVITY")
    print("=" * 50)

    try:
        from bitvavo_api import authenticate_exchange
        from trade_executor import TradeExecutor

        print("🔐 Authenticating with Bitvavo...")
        bitvavo = authenticate_exchange()
        print("✅ Bitvavo authentication successful")

        print("📊 Testing trade executor...")
        executor = TradeExecutor(bitvavo)

        current_price, volume = executor.fetch_current_price()
        if current_price:
            print(f"✅ Current BTC price: €{current_price:.2f} (volume: {volume:.2f})")
        else:
            print("❌ Failed to fetch current price")
            return False

        btc_balance = executor.get_total_btc_balance()
        eur_balance = executor.get_available_balance("EUR")

        print(f"💎 BTC balance: {btc_balance:.8f}")
        print(f"💶 EUR balance: €{eur_balance:.2f}")
        print(
            f"💰 Total portfolio value: €{eur_balance + (btc_balance * current_price):.2f}"
        )

        # Test order book
        order_book = executor.get_btc_order_book()
        if order_book:
            bids = len(order_book.get("bids", []))
            asks = len(order_book.get("asks", []))
            print(f"📋 Order book: {bids} bids, {asks} asks")

        return True

    except Exception as e:
        print(f"❌ API connectivity test failed: {e}")
        return False


def check_order_manager_state():
    """Check for potential order manager issues"""
    print("\n🔍 CHECKING ORDER MANAGER STATE")
    print("=" * 50)

    try:
        from bitvavo_api import authenticate_exchange
        from order_manager import OrderManager

        bitvavo = authenticate_exchange()
        order_manager = OrderManager(bitvavo)

        # Get current state
        pending = order_manager.get_pending_orders()
        filled_24h = order_manager.get_filled_orders(24)
        stats = order_manager.get_order_statistics()

        print(f"📊 Current pending orders: {len(pending)}")
        print(f"📈 Filled orders (24h): {len(filled_24h)}")
        print(f"📊 Fill rate: {stats['fill_rate']:.1%}")
        print(f"💰 Total fees paid: €{stats['total_fees_paid']:.4f}")

        if pending:
            print("\n⏳ Current pending orders:")
            for order_id, order_info in pending.items():
                age = datetime.now().timestamp() - order_info["timestamp"]
                side = order_info["side"]
                volume = order_info["volume"]
                price = order_info["price"]
                print(
                    f"  {side.upper()}: {volume:.8f} BTC @ €{price:.2f} (age: {age/60:.1f}min)"
                )

        if filled_24h:
            print("\n✅ Recent filled orders:")
            for order_id, order_info in list(filled_24h.items())[-3:]:
                side = order_info["side"]
                volume = order_info.get("executed_volume", order_info["volume"])
                price = order_info.get("average_price", order_info["price"])
                filled_time = datetime.fromtimestamp(order_info.get("filled_at", 0))
                print(
                    f"  {side.upper()}: {volume:.8f} BTC @ €{price:.2f} at {filled_time.strftime('%H:%M')}"
                )

        return True

    except Exception as e:
        print(f"❌ Order manager check failed: {e}")
        return False


def generate_recommendations():
    """Generate recommendations based on findings"""
    print("\n💡 RECOMMENDATIONS")
    print("=" * 50)

    recommendations = []

    # Check if bot has been running
    if not os.path.exists("./order_history.json"):
        recommendations.append(
            "🔧 Bot hasn't been running long enough to generate orders"
        )
        recommendations.append(
            "   → Let the bot run for at least 30 minutes to see activity"
        )

    # Check for stale session
    if os.path.exists("./trade_session.json"):
        try:
            with open("./trade_session.json", "r") as f:
                session_data = json.load(f)
            session_date = session_data.get("date", "1970-01-01")
            current_date = datetime.now().date().isoformat()
            if session_date != current_date:
                recommendations.append("🔧 Trade session is from a previous day")
                recommendations.append("   → Daily counters will reset on next bot run")
        except:
            pass

    # Check for recent activity
    if os.path.exists("./bot_logs.csv"):
        try:
            df = pd.read_csv("./bot_logs.csv")
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            recent = df[df["timestamp"] > (datetime.now() - timedelta(hours=2))]

            if len(recent) == 0:
                recommendations.append("⚠️ No recent bot activity in logs")
                recommendations.append("   → Check if bot is running and not stuck")
            elif len(recent) < 5:
                recommendations.append("⚠️ Very little recent activity")
                recommendations.append(
                    "   → Bot might be in 'hold' mode due to market conditions"
                )
        except:
            pass

    # General recommendations
    recommendations.extend(
        [
            "🔧 To force refresh order state: restart the bot",
            "🔧 To check real-time status: python main.py status",
            "🔧 Monitor logs with: tail -f trading_bot.log",
            "🔧 If orders seem stuck: check Bitvavo web interface",
        ]
    )

    for rec in recommendations:
        print(rec)

def test_performance_tracker():
    """Test the performance tracker independently"""
    try:
        from performance_tracker import PerformanceTracker
        
        # Test initialization
        tracker = PerformanceTracker(0.008, 45.0)
        print("✅ Performance tracker created")
        
        # Test recording a trade
        tracker.record_trade(
            order_id="test_001",
            side="buy",
            volume=0.001,
            price=96900.0,
            fee=0.24
        )
        print("✅ Trade recorded")
        
        # Test equity update
        tracker.update_equity(0.009, 42.0, 96900.0)
        print("✅ Equity updated")
        
        # Test report generation
        report = tracker.generate_performance_report()
        print("✅ Report generated")
        print(f"Current equity: {report['equity']['current']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance tracker test failed: {e}")
        return False


def main():
    """Run complete diagnostic check"""
    print("🔍 BITCOIN TRADING BOT DIAGNOSTIC")
    print("=" * 60)
    print(f"⏰ Diagnostic run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    checks = [
        ("API Connectivity", test_api_connectivity),
        ("Order History", analyze_order_history),
        ("Trade Session", analyze_trade_session),
        ("Bot Logs", analyze_bot_logs),
        ("Recent Buys", analyze_recent_buys),
        ("Order Manager State", check_order_manager_state),
        ("Performance Tracker", test_performance_tracker),
    ]

    results = {}

    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"❌ {check_name} check failed: {e}")
            results[check_name] = False

    # Summary
    print("\n📊 DIAGNOSTIC SUMMARY")
    print("=" * 50)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    print(f"✅ Checks passed: {passed}/{total}")

    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {check_name}")

    if passed < total:
        print(f"\n⚠️ {total - passed} checks failed - see details above")

    generate_recommendations()

    print("\n🏁 Diagnostic complete!")


if __name__ == "__main__":
    main()
