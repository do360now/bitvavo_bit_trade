# onchain_analyzer.py
import time
from typing import Dict
from bitcoinrpc.authproxy import AuthServiceProxy, JSONRPCException
from config import (
    RPC_HOST, RPC_PORT, RPC_USER, RPC_PASSWORD,
    MIN_BTC, EXCHANGE_ADDRESSES, ONCHAIN_CACHE_DURATION
)
from logger_config import logger

class OnChainAnalyzer:
    def __init__(self):
        self.rpc = None
        self.connect_rpc()
        self.onchain_cache = {"fee_rate": 0, "netflow": 0, "volume": 0, "old_utxos": 0}
        self.onchain_cache_time = 0
        self.onchain_cache_duration = ONCHAIN_CACHE_DURATION
        self.block_cache = {}
        self.mempool_cache = None
        self.mempool_cache_time = 0

    def connect_rpc(self):
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                rpc_url = f"http://{RPC_USER}:{RPC_PASSWORD}@{RPC_HOST}:{RPC_PORT}"
                logger.debug(f"Connecting to node: {RPC_HOST}:{RPC_PORT}, attempt {attempt + 1}")
                self.rpc = AuthServiceProxy(rpc_url, timeout=30)
                self.rpc.getblockcount()  # Health check
                logger.debug("RPC connection established")
                return
            except Exception as e:
                logger.error(f"Node connection failed: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.warning("Max RPC connection attempts reached")
                    self.rpc = None

    def check_rpc_health(self):
        try:
            if not self.rpc:
                self.connect_rpc()
            self.rpc.getblockcount()
            return True
        except:
            logger.debug("RPC health check failed, reconnecting")
            self.connect_rpc()
            return bool(self.rpc)

    def satoshis_to_btc(self, value):
        """
        Convert a value (possibly in scientific notation or various types) to BTC float.
        Handles strings like "0E-8" or "1e-8" natively via float conversion.
        Ensures non-negative result and defaults to 0.0 on invalid inputs.
        """
        if value is None or value == "":
            return 0.0
        try:
            converted = float(value)
            return max(converted, 0.0)
        except (ValueError, TypeError) as e:
            logger.debug(f"Invalid value for BTC conversion: {value} (type: {type(value)}), error: {e}")
            return 0.0

    def is_exchange_address(self, address):
        return EXCHANGE_ADDRESSES.get(address, "Unknown")

    def get_utxo_age(self, txid, vout):
        try:
            if not self.check_rpc_health():
                return None
            tx = self.rpc.getrawtransaction(txid, True)
            block_height = tx.get("blockheight", 0)
            current_height = self.rpc.getblockcount()
            return (current_height - block_height) * 10 / 1440 if block_height else None
        except:
            return None

    def get_onchain_signals(self) -> Dict[str, float]:
        if not self.check_rpc_health():
            logger.warning("No node connection; using cached on-chain signals")
            return self.onchain_cache

        current_time = time.time()
        if (current_time - self.onchain_cache_time) < self.onchain_cache_duration:
            return self.onchain_cache

        retries = 3
        for attempt in range(retries):
            try:
                # Mempool
                if not self.mempool_cache or (current_time - self.mempool_cache_time) > 60:
                    mempool = self.rpc.getmempoolinfo()
                    fee_rate = float(mempool["total_fee"]) / mempool["size"] * 1e8 if mempool["size"] > 0 else 0
                    self.mempool_cache = {"fee_rate": fee_rate}
                    self.mempool_cache_time = current_time
                else:
                    fee_rate = self.mempool_cache["fee_rate"]

                # Blocks (reduce to 2 blocks for speed)
                current_height = self.rpc.getblockcount()
                netflow = 0
                volume = 0
                old_utxos = 0
                tx_count = 0
                vout_count = 0
                error_count = 0
                for height in range(max(current_height - 2, 0), current_height + 1):
                    block_hash = self.rpc.getblockhash(height)
                    if block_hash in self.block_cache:
                        block = self.block_cache[block_hash]
                    else:
                        block = self.rpc.getblock(block_hash, 2)
                        self.block_cache[block_hash] = block
                        if len(self.block_cache) > 4:
                            self.block_cache.pop(list(self.block_cache.keys())[0])
                    block_error_count = 0
                    block_volume = 0
                    for tx in block["tx"]:
                        if "coinbase" in [vin.get("coinbase", "") for vin in tx.get("vin", [])]:
                            continue
                        tx_count += 1
                        tx_vouts = [vout for vout in tx["vout"] if vout.get("value") is not None and self.satoshis_to_btc(vout.get("value")) >= 0.000001]
                        if not tx_vouts:
                            if block_error_count < 5:
                                logger.debug(f"Tx in block {height} has no valid vouts, skipping, txid: {tx.get('txid')[:8]}...")
                            block_error_count += 1
                            error_count += 1
                            continue
                        tx_value = sum(self.satoshis_to_btc(vout["value"]) for vout in tx_vouts)
                        if tx_value == 0 and tx_vouts and block_error_count < 5:
                            block_error_count += 1
                            error_count += 1
                            logger.debug(f"Tx in block {height} returned 0 value; sample vout: {tx_vouts[0].get('value')}, txid: {tx.get('txid')[:8]}...")
                        block_volume += tx_value
                        volume += tx_value
                        vout_count += len(tx_vouts)
                        for vout in tx_vouts:
                            amount = self.satoshis_to_btc(vout["value"])
                            if amount >= MIN_BTC:
                                address = vout.get("scriptPubKey", {}).get("address", "N/A")
                                exchange = self.is_exchange_address(address)
                                netflow += amount if exchange != "Unknown" else -amount
                        if netflow < -50:
                            for vin in tx.get("vin", [])[:10]:
                                prev_txid = vin.get("txid")
                                prev_vout = vin.get("vout")
                                if prev_txid and prev_vout is not None:
                                    age = self.get_utxo_age(prev_txid, prev_vout)
                                    if age and age > 365:
                                        old_utxos += 1
                    logger.debug(f"Block {height} volume: {block_volume:.2f} BTC")
                    if block_error_count > 0:
                        logger.debug(f"Block {height} had {block_error_count} txs with invalid vouts")

                # Mempool fallback if low volume
                if volume < 5000:
                    logger.debug("Low block volume; checking mempool")
                    mempool_txs = self.rpc.getrawmempool(True)
                    mempool_error_count = 0
                    for txid in list(mempool_txs.keys())[:50]:  # Reduce to 50 txs for speed
                        try:
                            tx = self.rpc.getrawtransaction(txid, True)
                            tx_vouts = [vout for vout in tx["vout"] if vout.get("value") is not None and self.satoshis_to_btc(vout.get("value")) >= 0.000001]
                            if not tx_vouts:
                                if mempool_error_count < 5:
                                    logger.debug(f"Mempool tx {txid[:8]} has no valid vouts")
                                mempool_error_count += 1
                                continue
                            tx_value = sum(self.satoshis_to_btc(vout["value"]) for vout in tx_vouts)
                            if tx_value == 0 and tx_vouts and mempool_error_count < 5:
                                mempool_error_count += 1
                                error_count += 1
                                logger.debug(f"Mempool tx {txid[:8]} returned 0 value")
                            volume += tx_value
                            vout_count += len(tx_vouts)
                            for vout in tx_vouts:
                                amount = self.satoshis_to_btc(vout["value"])
                                if amount >= MIN_BTC:
                                    address = vout.get("scriptPubKey", {}).get("address", "N/A")
                                    exchange = self.is_exchange_address(address)
                                    netflow += amount if exchange != "Unknown" else -amount
                        except:
                            continue
                    if mempool_error_count > 0:
                        logger.debug(f"Mempool had {mempool_error_count} txs with invalid vouts")

                if error_count > 0:
                    logger.debug(f"Total {error_count} txs with invalid vouts across blocks and mempool")

                signals = {
                    "fee_rate": fee_rate,
                    "netflow": netflow,
                    "volume": volume,
                    "old_utxos": old_utxos
                }
                self.onchain_cache = signals
                self.onchain_cache_time = current_time
                logger.debug(f"On-chain: Fee={fee_rate:.2f} sat/vB, Netflow={netflow:.2f} BTC, Volume={volume:.2f} BTC, Old_UTXOs={old_utxos}, Txs={tx_count}, Vouts={vout_count}")
                return signals
            except Exception as e:
                logger.error(f"On-chain query failed (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    self.connect_rpc()
                    time.sleep(2 ** attempt)
                else:
                    logger.warning("Max retries reached; using cached on-chain signals")
                    return self.onchain_cache