import json
import os
import pandas as pd
import csv
import numpy as np
from typing import List, Tuple, Dict, Any
from logger_config import logger
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
                if not isinstance(candle, list) or len(candle) < 7:
                    logger.debug(f"Skipping invalid candle: {candle}")
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
                    volume = float(candle[6])  # Volume
                    prices.append(close_price)
                    volumes.append(volume)
                except (ValueError, TypeError) as e:
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

    def append_ohlc_data(self, ohlc: List[List]) -> None:
        try:
            # Load existing data
            with open(self.price_history_file, "r") as f:
                existing_data = json.load(f)
            if not isinstance(existing_data, list):
                logger.warning(
                    f"Resetting invalid price history format: {type(existing_data)}"
                )
                existing_data = []

            # Deduplicate existing data
            seen_timestamps = {
                candle[0]
                for candle in existing_data
                if isinstance(candle, list) and len(candle) >= 7
            }
            valid_ohlc = []
            for candle in ohlc:
                if not isinstance(candle, list) or len(candle) < 7:
                    logger.debug(f"Skipping invalid OHLC candle: {candle}")
                    continue
                try:
                    timestamp = float(candle[0])
                    if timestamp in seen_timestamps:
                        logger.debug(
                            f"Skipping duplicate OHLC candle at timestamp {timestamp}"
                        )
                        continue
                    float(candle[4])  # Validate close price
                    float(candle[6])  # Validate volume
                    valid_ohlc.append(candle)
                    seen_timestamps.add(timestamp)
                except (ValueError, TypeError) as e:
                    logger.debug(
                        f"Skipping invalid OHLC candle due to error: {e}, candle: {candle}"
                    )
                    continue

            existing_data.extend(valid_ohlc)
            # Sort all data by timestamp
            existing_data.sort(
                key=lambda x: (
                    x[0] if isinstance(x, list) and len(x) >= 1 else float("inf")
                )
            )
            with open(self.price_history_file, "w") as f:
                json.dump(existing_data, f)
            logger.info(
                f"Appended {len(valid_ohlc)} valid OHLC candles to {self.price_history_file}"
            )
        except Exception as e:
            logger.error(f"Failed to append OHLC data: {e}")

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
