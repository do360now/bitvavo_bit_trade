import json
import os
import pandas as pd
import csv
from typing import List, Tuple
from utils.logger import logger
import fcntl
import time
from datetime import datetime


class DataManager:
    # Define headers as a class attribute
    HEADERS = [
        "timestamp",
        "price",
        "trade_volume",
        "side",
        "reason",
        "dip",
        "rsi",
        "macd",
        "signal",
        "ma_short",
        "ma_long",
        "upper_band",
        "lower_band",
        "sentiment",
        "fee_rate",
        "netflow",
        "volume",
        "old_utxos",
        "buy_decision",
        "sell_decision",
        "btc_balance",
        "eur_balance",
        "avg_buy_price",
        "profit_margin",
    ]

    def __init__(self, price_history_file: str, bot_logs_file: str):
        self.price_history_file = price_history_file
        self.bot_logs_file = bot_logs_file
        if not os.path.exists(self.price_history_file):
            with open(self.price_history_file, "w") as f:
                json.dump([], f)
        self._initialize_bot_logs()

    def _initialize_bot_logs(self):
        if (
            not os.path.exists(self.bot_logs_file)
            or os.path.getsize(self.bot_logs_file) == 0
        ):
            with open(self.bot_logs_file, "w", newline="") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
                writer.writerow(self.HEADERS)
            logger.info(f"Initialized {self.bot_logs_file} with headers")
        else:
            try:
                df = pd.read_csv(
                    self.bot_logs_file, nrows=1, encoding="utf-8", on_bad_lines="skip"
                )
                if not all(col in df.columns for col in self.HEADERS):
                    logger.error(
                        f"Invalid headers in {self.bot_logs_file}. Recreating file."
                    )
                    with open(self.bot_logs_file, "w", newline="") as f:
                        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
                        writer.writerow(self.HEADERS)
                    logger.info(
                        f"Reinitialized {self.bot_logs_file} with correct headers"
                    )
            except Exception as e:
                logger.error(
                    f"Failed to validate {self.bot_logs_file}: {e}. Recreating file."
                )
                with open(self.bot_logs_file, "w", newline="") as f:
                    writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
                    writer.writerow(self.HEADERS)
                logger.info(f"Reinitialized {self.bot_logs_file} with headers")

    def load_price_history(self) -> Tuple[List[float], List[float]]:
        try:
            with open(self.price_history_file, "r") as f:
                data = json.load(f)
            if not isinstance(data, list):
                logger.error(
                    f"Invalid price history format: expected list, got {type(data)}"
                )
                return [], []

            # Deduplicate and sort by timestamp
            seen_timestamps = set()
            unique_data = []
            for candle in data:
                if (
                    not isinstance(candle, list) or len(candle) < 6
                ):  # FIXED: Changed from 7 to 6
                    logger.debug(f"Skipping invalid candle format: {candle}")
                    continue
                timestamp = candle[0]
                if timestamp in seen_timestamps:
                    continue
                seen_timestamps.add(timestamp)
                unique_data.append(candle)

            # Sort by timestamp
            unique_data.sort(key=lambda x: x[0])

            # Log timestamp range for debugging
            if unique_data:
                min_ts = unique_data[0][0]
                max_ts = unique_data[-1][0]
                min_dt = datetime.fromtimestamp(min_ts).isoformat()
                max_dt = datetime.fromtimestamp(max_ts).isoformat()
                logger.info(f"Price history timestamp range: {min_dt} to {max_dt}")

            prices = []
            volumes = []
            for i, candle in enumerate(unique_data):
                try:
                    close_price = float(candle[4])  # Close price
                    volume = float(candle[5])  # FIXED: Changed from index 6 to 5

                    # FIXED: Add basic validation but don't reject valid Bitcoin prices
                    if close_price > 1000 and close_price < 1000000 and volume >= 0:
                        prices.append(close_price)
                        volumes.append(volume)
                    else:
                        logger.debug(
                            f"Skipping candle with invalid price/volume: price={close_price}, volume={volume}"
                        )

                except (ValueError, TypeError, IndexError) as e:
                    logger.debug(
                        f"Skipping candle at index {i} due to error: {e}, candle: {candle}"
                    )
                    continue

            logger.info(
                f"Loaded {len(prices)} valid prices from {self.price_history_file}"
            )
            return prices, volumes

        except Exception as e:
            logger.error(f"Failed to load price history: {e}")
            return [], []

    def append_ohlc_data(self, ohlc_data: List[List]) -> int:
        """Append new OHLC data to price history, avoiding duplicates - FIXED VERSION"""
        if not ohlc_data:
            logger.debug("No OHLC data to append")
            return 0
        
        try:
            # Load existing data
            existing_data = []
            if os.path.exists(self.price_history_file):
                with open(self.price_history_file, 'r') as f:
                    existing_data = json.load(f)
            
            # Get existing timestamps for duplicate detection
            existing_timestamps = set()
            if existing_data:
                for candle in existing_data:
                    if isinstance(candle, list) and len(candle) > 0:
                        existing_timestamps.add(int(candle[0]))
            
            # FIXED: More lenient timestamp validation
            current_time = int(time.time())
            min_valid_timestamp = current_time - (30 * 24 * 3600)  # 30 days ago (was too restrictive)
            max_valid_timestamp = current_time + (1 * 24 * 3600)   # 1 day in future
            
            logger.info(f"Current time: {current_time} ({datetime.fromtimestamp(current_time)})")
            logger.info(f"Processing {len(ohlc_data)} potential new candles...")
            logger.info(f"Existing data points: {len(existing_data)}")
            logger.info(f"Existing timestamps range: {min(existing_timestamps) if existing_timestamps else 'None'} to {max(existing_timestamps) if existing_timestamps else 'None'}")
            
            new_candles = []
            processed_count = 0
            
            for candle in ohlc_data:
                processed_count += 1
                try:
                    if not isinstance(candle, list) or len(candle) < 6:
                        logger.debug(f"Skipping invalid candle format: {candle}")
                        continue
                    
                    # Extract timestamp and convert if needed
                    timestamp = int(candle[0])
                    
                    # Log first few timestamps for debugging
                    if processed_count <= 3:
                        logger.info(f"Processing timestamp {processed_count}: {timestamp} ({datetime.fromtimestamp(timestamp)})")
                    
                    # FIXED: Less restrictive timestamp validation
                    if timestamp < min_valid_timestamp:
                        if processed_count <= 3:
                            logger.debug(f"Skipping old timestamp: {timestamp} (older than 30 days)")
                        continue
                    
                    if timestamp > max_valid_timestamp:
                        if processed_count <= 3:
                            logger.debug(f"Skipping future timestamp: {timestamp}")
                        continue
                    
                    # Skip duplicates
                    if timestamp in existing_timestamps:
                        if processed_count <= 3:
                            logger.debug(f"Skipping duplicate timestamp: {timestamp}")
                        continue
                    
                    # Validate OHLC values
                    open_price = float(candle[1])
                    high_price = float(candle[2])
                    low_price = float(candle[3])
                    close_price = float(candle[4])
                    volume = float(candle[5])
                    
                    # Basic price validation
                    if not (0 < low_price <= high_price and 
                        low_price <= open_price <= high_price and 
                        low_price <= close_price <= high_price and
                        volume >= 0):
                        logger.debug(f"Skipping invalid OHLC values: {candle}")
                        continue
                    
                    # Price sanity check (Bitcoin should be > $1000 and < $1M)
                    if not (1000 < close_price < 1000000):
                        logger.debug(f"Skipping unrealistic price: {close_price}")
                        continue
                    
                    new_candles.append([timestamp, open_price, high_price, low_price, close_price, volume])
                    existing_timestamps.add(timestamp)
                    
                    if len(new_candles) <= 3:
                        logger.info(f"Added valid candle: {timestamp} @ €{close_price:.2f}")
                    
                except (ValueError, TypeError, IndexError) as e:
                    logger.debug(f"Skipping invalid OHLC candle: {candle} - {e}")
                    continue
            
            if new_candles:
                # Add new candles to existing data
                existing_data.extend(new_candles)
                
                # Sort by timestamp and remove any remaining duplicates
                existing_data.sort(key=lambda x: x[0])
                
                # Keep only last 5000 candles to manage file size (increased from 2000)
                if len(existing_data) > 5000:
                    existing_data = existing_data[-5000:]
                
                # Save updated data
                with open(self.price_history_file, 'w') as f:
                    json.dump(existing_data, f)
                
                logger.info(f"✅ Successfully added {len(new_candles)} new OHLC candles")
                logger.info(f"📊 Total candles in history: {len(existing_data)}")
                
                if new_candles:
                    latest = new_candles[-1]
                    earliest = new_candles[0]
                    logger.info(f"📅 Date range: {datetime.fromtimestamp(earliest[0])} to {datetime.fromtimestamp(latest[0])}")
                    logger.info(f"💰 Price range: €{min(c[4] for c in new_candles):.2f} to €{max(c[4] for c in new_candles):.2f}")
            else:
                logger.info(f"ℹ️  No new valid candles to append from {len(ohlc_data)} processed")
            
            return len(new_candles)
            
        except Exception as e:
            logger.error(f"Failed to append OHLC data: {e}")
            return 0
    
    def log_strategy(self, **kwargs) -> None:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if "timestamp" not in kwargs:
                    kwargs["timestamp"] = datetime.now().isoformat()

                # Define default values for all columns
                defaults = {
                    "price": None,
                    "trade_volume": None,
                    "side": "",
                    "reason": "",
                    "dip": None,
                    "rsi": None,
                    "macd": None,
                    "signal": None,
                    "ma_short": None,
                    "ma_long": None,
                    "upper_band": None,
                    "lower_band": None,
                    "sentiment": None,
                    "fee_rate": None,
                    "netflow": None,
                    "volume": None,
                    "old_utxos": None,
                    "buy_decision": "False",
                    "sell_decision": "False",
                    "btc_balance": None,
                    "eur_balance": None,
                    "avg_buy_price": None,
                    "profit_margin": None,
                }
                # Update defaults with provided kwargs
                for key in defaults:
                    if key not in kwargs:
                        kwargs[key] = defaults[key]

                df = pd.DataFrame([kwargs])
                # Ensure buy_decision and sell_decision are strings
                df["buy_decision"] = df["buy_decision"].apply(
                    lambda x: (
                        "True"
                        if str(x).lower() in ["true", "1"]
                        else "False" if pd.notnull(x) else "False"
                    )
                )
                df["sell_decision"] = df["sell_decision"].apply(
                    lambda x: (
                        "True"
                        if str(x).lower() in ["true", "1"]
                        else "False" if pd.notnull(x) else "False"
                    )
                )
                # Ensure side is lowercase
                if "side" in df:
                    df["side"] = df["side"].str.lower()
                with open(self.bot_logs_file, "a", newline="") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        df.to_csv(
                            f,
                            index=False,
                            header=False,
                            quoting=csv.QUOTE_NONNUMERIC,
                            encoding="utf-8",
                            columns=self.HEADERS,
                        )
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                logger.debug(f"Logged strategy metrics to {self.bot_logs_file}")
                return
            except Exception as e:
                logger.error(
                    f"Failed to log strategy (attempt {attempt+1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(1)
        logger.error(f"Failed to log strategy after {max_retries} attempts")

    def validate_bot_logs(self):
        if not os.path.exists(self.bot_logs_file):
            logger.debug(f"No bot logs found at {self.bot_logs_file}")
            return False
        df = pd.read_csv(
            self.bot_logs_file,
            dtype={"buy_decision": str, "sell_decision": str},
            encoding="utf-8",
            on_bad_lines="skip",
        )
        expected_columns = [
            "timestamp",
            "price",
            "trade_volume",
            "side",
            "reason",
            "buy_decision",
            "sell_decision",
        ]
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns in bot_logs.csv: {missing_cols}")
            return False
        buy_trades = df[df["buy_decision"].str.lower().isin(["true", "1"])]
        if buy_trades.empty:
            logger.debug("No valid buy trades in bot_logs.csv")
            return False
        logger.info(f"Validated bot_logs.csv: {len(buy_trades)} buy trades found")
        return True
