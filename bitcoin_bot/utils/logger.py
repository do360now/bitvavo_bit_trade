import logging
from logging.handlers import RotatingFileHandler
import os

# Set log level from environment variable, defaulting to INFO
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()

# Configure logger
logger = logging.getLogger("trading_bot")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# Console handler for logger
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Rotating file handler for persistent logging
log_file = os.getenv("LOG_FILE", "trading_bot.log")
log_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=2)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler.setFormatter(file_formatter)
logger.addHandler(log_handler)

# Disable propagation to prevent duplicate logs
logger.propagate = False