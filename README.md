# 🤖 Bitcoin Trading Bot - Bitvavo Edition

An advanced, AI-powered Bitcoin trading bot with machine learning capabilities, risk management, and comprehensive market analysis. Built specifically for the Bitvavo exchange with enhanced features for automated Bitcoin accumulation.

## 🚀 Features

### Core Trading Features
- **Automated Bitcoin Trading** on Bitvavo exchange
- **Smart Position Sizing** with risk-adjusted calculations
- **Advanced Order Management** with timeout handling and fill tracking
- **15-minute Trading Intervals** with precise scheduling
- **Multi-layered Risk Management** including peak avoidance and macro risk analysis

### Advanced AI & ML Features
- **Machine Learning Engine** that learns from past trades
- **Peak Avoidance System** to prevent buying at local tops
- **Adaptive Position Sizing** based on performance and market conditions
- **News Sentiment Analysis** with macro-economic risk assessment
- **On-chain Analysis** including Bitcoin network metrics

### Risk Management
- **Dynamic Stop-loss** based on market volatility
- **Risk-off Detection** using news analysis
- **Liquidation Cascade Prevention**
- **Portfolio Allocation Controls** (max 25% per trade)
- **Daily Trade Limits** to prevent overtrading

### Monitoring & Analytics
- **Real-time Performance Tracking** with Sharpe ratio, drawdown analysis
- **Comprehensive Logging** of all decisions and trades
- **Web-based Metrics Server** (Prometheus compatible)
- **Trade Session Management** with daily counters
- **Order History Tracking** with detailed fill analysis

## 📊 Performance Metrics

The bot tracks and optimizes based on:
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Fill Rate**: Percentage of orders successfully executed
- **Average Fill Time**: Speed of order execution

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Bitvavo API credentials
- Bitcoin node (optional, for on-chain analysis)
- News API key (optional, for enhanced sentiment analysis)

### Dependencies
```bash
pip install ccxt pandas numpy scikit-learn nltk yfinance requests python-dotenv tenacity
```

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd bitcoin-trading-bot
```

2. Create environment file:
```bash
cp .env.example .env
```

3. Configure your `.env` file:
```env
BITVAVO_API_KEY=your_api_key_here
BITVAVO_API_SECRET=your_api_secret_here
NEWS_API_KEY=your_news_api_key_here
RPC_USER=your_bitcoin_node_user
RPC_PASSWORD=your_bitcoin_node_password
RPC_HOST=localhost
RPC_PORT=8332
```

4. Initialize the bot:
```bash
python3 main.py
```

## 📈 Usage

### Basic Operation
```bash
# Run the trading bot
python3 main.py

# Check status only
python3 main.py status

# Show help
python3 main.py help
```

### Configuration

Key settings in `config.py`:
- `MAX_DAILY_TRADES`: Maximum trades per day (default: 8)
- `BASE_POSITION_PCT`: Base position size as % of balance (default: 8%)
- `MIN_EUR_FOR_TRADE`: Minimum EUR needed to trade (default: €10)
- `GLOBAL_TRADE_COOLDOWN`: Cooldown between trades (default: 180s)

### Trading Strategies

The bot includes three trading modes:

1. **Standard Bot**: Basic technical analysis with risk management
2. **Enhanced Bot**: Adds ML learning and adaptive features
3. **Ultimate Adaptive Bot**: Full AI with peak avoidance and comprehensive analytics

## 🧠 AI & Machine Learning

### Learning Engine
- **Random Forest Classifier** for trade success prediction
- **Feature Engineering** using 12+ market indicators
- **Continuous Learning** from trade outcomes
- **Performance-based Adaptation** of position sizes

### Peak Avoidance System
- **Historical Pattern Analysis** to identify price peaks
- **Real-time Peak Detection** using multiple algorithms
- **Adaptive Entry Strategies** to avoid buying at tops
- **Pattern Database** that grows with market experience

### Risk Analysis
- **Macro-economic News Monitoring** for risk-off events
- **Market Correlation Analysis** with traditional assets
- **Volatility Regime Detection** for adaptive strategies
- **Liquidation Cascade Prevention** using volume/price patterns

## 📊 Monitoring

### Web Interface
Access real-time metrics at:
- **Health Check**: `http://localhost:8082/health`
- **Prometheus Metrics**: `http://localhost:8082/metrics`
- **JSON Stats**: `http://localhost:8082/stats`

### Log Files
- `trading_bot.log`: Main application logs
- `bot_logs.csv`: Detailed trading decisions and market data
- `order_history.json`: Complete order tracking
- `enhanced_decisions.json`: AI decision logs
- `risk_decisions.json`: Risk analysis logs

## 🔧 Architecture

### Core Components

```
├── main.py                     # Main application entry point
├── trading_bot.py              # Core trading logic
├── enhanced_trading_bot.py     # ML-enhanced trading
├── complete_integration.py     # Ultimate adaptive bot
├── order_manager.py            # Order execution and tracking
├── trade_executor.py           # Exchange interface
├── data_manager.py             # Data storage and retrieval
├── indicators.py               # Technical and sentiment analysis
├── onchain_analyzer.py         # Bitcoin network analysis
├── peak_avoidance_system.py    # Peak detection and avoidance
├── performance_tracker.py      # Portfolio performance analysis
├── metrics_server.py           # Web-based monitoring
└── config.py                   # Configuration management
```

### Data Flow

1. **Market Data Collection**: Price, volume, news, on-chain metrics
2. **Analysis Pipeline**: Technical indicators, sentiment, ML predictions
3. **Decision Engine**: Risk assessment, position sizing, trade signals
4. **Order Management**: Smart order placement with timeout handling
5. **Performance Tracking**: Real-time P&L, risk metrics, trade analysis

## ⚙️ Configuration Options

### Trading Parameters
```python
TRADING_PARAMS = {
    'USE_STOP_LOSS': True,
    'STOP_LOSS_PERCENT': 0.03,      # 3% stop loss
    'USE_TAKE_PROFIT': True,
    'TAKE_PROFIT_PERCENT': 0.08,    # 8% take profit
    'MAX_POSITION_SIZE': 0.15,      # 15% max position
    'MIN_TRADE_VOLUME': 0.0001,     # Minimum BTC trade size
}
```

### Risk Management
```python
ENHANCED_RISK_PARAMS = {
    'BASE_STOP_LOSS_PCT': 0.03,
    'MAX_RISK_OFF_THRESHOLD': 0.6,
    'HIGH_VOLATILITY_THRESHOLD': 0.05,
    'MIN_CONFIDENCE_THRESHOLD': 60.0,
}
```

### News Analysis
```python
NEWS_CONFIG = {
    'MAX_NEWS_ARTICLES': 20,
    'NEWS_CACHE_MINUTES': 30,
    'RISK_OFF_WEIGHT': 2.0,
    'MACRO_NEWS_WEIGHT': 2.0,
}
```

## 📚 API Reference

### TradingBot Class
```python
bot = TradingBot(data_manager, trade_executor, onchain_analyzer, order_manager)

# Execute trading strategy
bot.execute_strategy()

# Check order status
bot.check_pending_orders()

# Get performance summary
status = bot.get_trading_status_summary()
```

### Enhanced Features
```python
# Enhance existing bot with ML
enhanced_bot = enhance_existing_bot(original_bot)

# Run enhanced strategy
enhanced_bot.execute_enhanced_strategy()

# Get ML performance metrics
performance = enhanced_bot.get_performance_summary()
```

## 🚨 Risk Warnings

- **Trading Risk**: Cryptocurrency trading involves substantial risk of loss
- **API Security**: Keep your API keys secure and use IP restrictions
- **Start Small**: Begin with small position sizes to test the system
- **Monitor Actively**: Regularly check bot performance and logs
- **Backup Important**: Keep backups of configuration and historical data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive logging for new features
- Include unit tests for critical components
- Update documentation for API changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes. The authors are not responsible for any financial losses incurred through the use of this bot. Always do your own research and never invest more than you can afford to lose.

## 🙏 Acknowledgments

- [CCXT](https://github.com/ccxt/ccxt) for exchange connectivity
- [Bitvavo](https://bitvavo.com) for reliable API services
- [scikit-learn](https://scikit-learn.org/) for machine learning tools
- The Bitcoin and cryptocurrency community for inspiration

## 📞 Support

- **Documentation**: Check the inline code documentation
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Join community discussions in GitHub Discussions
- **Security**: Report security issues privately to maintainers

---
## Donations
If you find this trading bot useful and profitable, please consider making a donation. 
**BTC: bc1qyxdhqef7tszr75wy6y3w7rdfpr9y00cg6w0e8e**

---
**Happy Trading! 🚀📈**