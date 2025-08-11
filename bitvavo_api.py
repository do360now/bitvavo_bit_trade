# filename: authenticate.py
import ccxt
import os
from dotenv import load_dotenv
from logger_config import logger  # Import the standardized logger
from config import BITVAVO_API_KEY, BITVAVO_API_SECRET

# Load environment variables from the .env file (redundant if already loaded in config, but safe)
load_dotenv()

# Check if all API credentials are available
logger.info("Checking if all API credentials are available...")
missing_creds = [key for key, val in {"BITVAVO_API_KEY": BITVAVO_API_KEY, "BITVAVO_API_SECRET": BITVAVO_API_SECRET}.items() if not val]
if missing_creds:
    raise ValueError(f"Missing API credentials: {', '.join(missing_creds)}")

def authenticate_exchange():
    """
    Authenticate with Bitvavo using CCXT and return the exchange object.
    """
    logger.info("Authenticating with Bitvavo via CCXT...")

    bitvavo = ccxt.bitvavo({
        'apiKey': BITVAVO_API_KEY,
        'secret': BITVAVO_API_SECRET,
    })
    
    
    # Verify authentication by fetching balance (optional, but good for validation)
    try:
        bitvavo_balance = bitvavo.fetch_balance()
        logger.debug(f"Bitvavo balance: {bitvavo_balance}")
        logger.info("Successfully authenticated with Bitvavo.")
    except Exception as e:
        logger.error(f"Authentication verification failed: {e}")
        raise

    return bitvavo