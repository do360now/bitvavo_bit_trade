import time
from datetime import datetime
import json
from bitvavo_api import authenticate_exchange
from trading.executor import TradeExecutor
from core.data_manager import DataManager

bitvavo = authenticate_exchange()
executor = TradeExecutor(bitvavo)
data_manager = DataManager("./price_history.json", "./bot_logs.csv")

print("🔄 Fetching 14 days of fresh data...")
fresh_data = executor.get_ohlc_data("BTC/EUR", '15m', 
    since=int(time.time()) - (14 * 24 * 3600), limit=1344)

if fresh_data:
    print(f"📥 Got {len(fresh_data)} candles")
    with open("./price_history.json", 'w') as f:
        json.dump([], f)
    added = data_manager.append_ohlc_data(fresh_data)
    print(f"✅ Added {added} candles")
else:
    print("❌ Failed to fetch data")