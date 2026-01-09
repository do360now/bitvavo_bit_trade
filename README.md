# Bitvavo Bitcoin Trading Bot

A high-performance asynchronous Bitcoin trading bot for the Bitvavo exchange, built with sentiment analysis, technical indicators, and risk management.

## Features

- **Async/Await Architecture**: Non-blocking I/O for high-performance trading
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages, VWAP
- **Sentiment Analysis**: Real-time news sentiment tracking
- **Risk Management**: Position sizing, stop-loss orders, portfolio allocation
- **Performance Tracking**: Win rate, Sharpe ratio, drawdown analysis
- **Order Management**: Limit orders with automatic timeout cancellation
- **Rate Limiting**: Bitvavo API rate limit compliance with circuit breaker
- **Comprehensive Logging**: Detailed execution logs for debugging

## Structure

```
CMC_BITVAVO_BIT_TRADE/
├── main.py                    # Main bot entry point
├── bitvavo_api.py            # Enhanced Bitvavo API client with async support
├── order_manager.py          # Order placement and tracking
├── trade_executor.py         # Balance and price management
├── data_manager.py           # OHLC and price history management
├── performance_tracker.py    # Performance metrics and reporting
├── indicators.py             # Technical indicators and sentiment analysis
├── circuit_breaker.py        # Circuit breaker pattern for resilience
├── logger_config.py          # Logging configuration
├── requirements.txt          # Python dependencies
├── .env.template             # Environment configuration template
└── README.md                 # This file
```

## Installation

1. **Clone the repository**
   ```bash
   cd /home/cmc/git/Bit/CMC_BITVAVO_BIT_TRADE
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.template .env
   # Edit .env with your Bitvavo API credentials
   ```

4. **Set up API credentials**
   - Create API keys at https://bitvavo.com/en/account/api
   - Add to `.env`:
     ```
     BITVAVO_API_KEY=your_key_here
     BITVAVO_API_SECRET=your_secret_here
     ```

## Usage

### Running the Bot

```bash
python3 main.py
```

### Configuration

Edit `.env` file to customize:
- API credentials
- News API key (for sentiment analysis)
- Bitcoin node connection (for on-chain analysis)
- Logging level

## How It Works

### Strategy Flow

1. **Fetch Market Data**
   - Current price from Bitvavo ticker
   - Order book depth
   - OHLCV candle data

2. **Calculate Indicators**
   - RSI (overbought/oversold levels)
   - MACD (trend direction)
   - Bollinger Bands (volatility)
   - Moving Averages (trend)
   - VWAP (volume-weighted price)

3. **Analyze Sentiment**
   - Fetch recent news articles
   - Calculate sentiment score
   - Assess market risk-off probability

4. **Decision Making**
   - Combine indicators with sentiment
   - Generate buy/sell/hold signals
   - Size positions based on risk

5. **Execute Trades**
   - Place limit orders at optimal prices
   - Track order status
   - Update performance metrics

6. **Monitor & Log**
   - Check pending orders
   - Update equity curve
   - Log all decisions

### Trading Signals

**BUY Signal:**
- RSI < 30 (oversold)
- Positive sentiment
- Sufficient EUR balance

**SELL Signal:**
- RSI > 70 (overbought)
- BTC holdings to sell

**HOLD Signal:**
- Neutral indicators
- No strong signals

## API Integration

### Bitvavo API Methods

The bot uses asynchronous wrappers around the Bitvavo API:

```python
await bitvavo_api.get_ticker_async(symbol)      # Current price
await bitvavo_api.get_order_book_async(symbol)  # Order book
await bitvavo_api.get_ohlcv_async(symbol)       # Candle data
await bitvavo_api.get_balance_async()           # Account balance
await bitvavo_api.place_order_async(order_data) # Place order
await bitvavo_api.get_order_async(order_id)     # Order status
await bitvavo_api.cancel_order_async(market, id)# Cancel order
```

## Performance Metrics

The bot tracks:
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Fill Rate**: Percentage of orders successfully filled
- **Average Fill Time**: Time from placement to execution

## Logging

Logs are output to:
- Console (INFO level and above)
- `trading_bot.log` file (DEBUG level and above)

Log levels controlled via `.env`:
```
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## Error Handling

- **Circuit Breaker**: Stops API calls after 5 consecutive failures, 5-minute timeout
- **Rate Limiting**: Bitvavo API rate limit compliance with exponential backoff
- **Order Timeouts**: Automatic cancellation of orders older than specified timeout
- **Graceful Degradation**: Continues with neutral sentiment if news fetch fails

## Development

### Running Tests

```bash
python3 test_suite.py        # Unit tests
python3 test_integration.py  # Integration tests
```

### Code Quality

Run the code quality analyzer:

```bash
python3 code_quality_analyzer.py
```

## Security

- API credentials never logged or exposed
- Credentials cleared from memory on cleanup
- HMAC-SHA256 request signing
- Timestamp validation on all requests
- Rate limiting to prevent abuse

## Troubleshooting

### Connection Issues
```
ERROR: Failed to authenticate with Bitvavo
```
- Verify API credentials in `.env`
- Check API key has correct permissions
- Ensure IP is whitelisted on Bitvavo account

### Insufficient Balance
```
WARNING: Insufficient EUR balance for trading
```
- Add funds to your Bitvavo account
- Reduce position size in configuration
- Check for locked funds in pending orders

### No Trading Signals
- Bot may be in HOLD state (normal)
- Indicators may not align with trade conditions
- Check logs for signal details

## Contributing

Improvements welcome:
- Additional indicators
- Improved sentiment analysis
- Better risk management
- Performance optimizations

## License

See LICENSE file

## Resources

- [Bitvavo API Documentation](https://docs.bitvavo.com/)
- [Technical Analysis](https://en.wikipedia.org/wiki/Technical_analysis)
- [Sentiment Analysis](https://newsapi.org/)
