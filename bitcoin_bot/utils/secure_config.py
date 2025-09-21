# utils/secure_config.py
import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from typing import Optional
import keyring
import logging

logger = logging.getLogger(__name__)

class SecureConfig:
    """Secure configuration manager with encryption and keyring support."""
    
    def __init__(self):
        self._cipher_suite = None
        self._init_encryption()
    
    def _init_encryption(self):
        """Initialize encryption using system keyring or environment key."""
        try:
            # Try to get master key from system keyring
            master_key = keyring.get_password("bitcoin_bot", "master_key")
            
            if not master_key:
                # Generate new master key if none exists
                master_key = Fernet.generate_key().decode()
                keyring.set_password("bitcoin_bot", "master_key", master_key)
                logger.info("Generated new master encryption key")
            
            self._cipher_suite = Fernet(master_key.encode())
            
        except Exception as e:
            logger.warning(f"Keyring unavailable, using environment fallback: {e}")
            # Fallback to environment-based key derivation
            password = os.getenv("BOT_MASTER_PASSWORD", "default_insecure_password")
            salt = os.getenv("BOT_SALT", "default_salt").encode()
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            self._cipher_suite = Fernet(key)
    
    def get_api_credentials(self) -> tuple[str, str]:
        """Securely retrieve API credentials."""
        try:
            # Try keyring first
            api_key = keyring.get_password("bitcoin_bot", "bitvavo_api_key")
            api_secret = keyring.get_password("bitcoin_bot", "bitvavo_api_secret")
            
            if api_key and api_secret:
                return api_key, api_secret
            
            # Fallback to encrypted environment variables
            encrypted_key = os.getenv("BITVAVO_API_KEY_ENCRYPTED")
            encrypted_secret = os.getenv("BITVAVO_API_SECRET_ENCRYPTED")
            
            if encrypted_key and encrypted_secret:
                api_key = self._cipher_suite.decrypt(encrypted_key.encode()).decode()
                api_secret = self._cipher_suite.decrypt(encrypted_secret.encode()).decode()
                return api_key, api_secret
            
            # Last resort: plain environment variables (log warning)
            api_key = os.getenv("BITVAVO_API_KEY")
            api_secret = os.getenv("BITVAVO_API_SECRET")
            
            if api_key and api_secret:
                logger.warning("Using plain text API credentials - consider encrypting")
                return api_key, api_secret
            
            raise ValueError("No API credentials found")
            
        except Exception as e:
            logger.error(f"Failed to retrieve API credentials: {e}")
            raise
    
    def store_api_credentials(self, api_key: str, api_secret: str):
        """Securely store API credentials."""
        try:
            keyring.set_password("bitcoin_bot", "bitvavo_api_key", api_key)
            keyring.set_password("bitcoin_bot", "bitvavo_api_secret", api_secret)
            logger.info("API credentials stored securely in keyring")
        except Exception as e:
            logger.error(f"Failed to store credentials: {e}")
            raise

# Enhanced configuration with validation
class TradingConfig:
    """Type-safe trading configuration with validation."""
    
    def __init__(self):
        self.secure_config = SecureConfig()
        self._validate_config()
    
    @property
    def api_credentials(self) -> tuple[str, str]:
        return self.secure_config.get_api_credentials()
    
    @property
    def trading_params(self) -> dict:
        return {
            "max_daily_trades": self._get_int("MAX_DAILY_TRADES", 8, min_val=1, max_val=50),
            "base_position_size": self._get_float("BASE_POSITION_SIZE", 0.08, min_val=0.01, max_val=0.25),
            "stop_loss_pct": self._get_float("STOP_LOSS_PCT", 0.025, min_val=0.005, max_val=0.1),
            "take_profit_pct": self._get_float("TAKE_PROFIT_PCT", 0.08, min_val=0.02, max_val=0.3),
            "min_confidence": self._get_float("MIN_CONFIDENCE", 0.35, min_val=0.1, max_val=0.9),
        }
    
    def _get_int(self, key: str, default: int, min_val: int = None, max_val: int = None) -> int:
        try:
            value = int(os.getenv(key, default))
            if min_val is not None and value < min_val:
                logger.warning(f"{key} below minimum, using {min_val}")
                return min_val
            if max_val is not None and value > max_val:
                logger.warning(f"{key} above maximum, using {max_val}")
                return max_val
            return value
        except ValueError:
            logger.warning(f"Invalid {key}, using default {default}")
            return default
    
    def _get_float(self, key: str, default: float, min_val: float = None, max_val: float = None) -> float:
        try:
            value = float(os.getenv(key, default))
            if min_val is not None and value < min_val:
                logger.warning(f"{key} below minimum, using {min_val}")
                return min_val
            if max_val is not None and value > max_val:
                logger.warning(f"{key} above maximum, using {max_val}")
                return max_val
            return value
        except ValueError:
            logger.warning(f"Invalid {key}, using default {default}")
            return default
    
    def _validate_config(self):
        """Validate critical configuration parameters."""
        try:
            api_key, api_secret = self.api_credentials
            if not api_key or not api_secret:
                raise ValueError("API credentials are required")
            
            if len(api_key) < 10 or len(api_secret) < 10:
                raise ValueError("API credentials appear invalid")
                
            logger.info("Configuration validation passed")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise