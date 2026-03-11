# Bitcoin Trading Bot - Claude Guidance

This is a Bitcoin trading bot running on Bitvavo using a cycle-aware DCA strategy.

## Running the Bot

```bash
python3 main.py
```

## Core Modules

| File | Purpose |
|------|---------|
| main.py | Bot orchestration, health check, main loop |
| bitvavo_api.py | CCXT wrapper with circuit breaker |
| trade_executor.py | Order execution, position sizing |
| order_manager.py | Order tracking and management |
| atomic_trade_manager.py | Atomic state updates |
| cycle_trading_deep_module.py | Strategy: DCA in corrections |
| circuit_breaker.py | API failure protection |
| bot_state_manager.py | Persistent state |
| performance_tracker.py | P&L tracking |
| validators.py | Order validation |
| indicators.py | Technical indicators |

## Key Improvements

1. **Circuit Breaker** - `@circuit_breaker` decorator (3 failures → 60s cooldown)
2. **Startup Health Check** - Verifies API before trading
3. **Order Book Pricing** - Better fills using real-time order book
4. **Trailing Stop** - 10% drawdown from peak triggers sell
5. **Exponential Backoff** - Consecutive failures wait longer

## Testing

```bash
python3 tests/test_suite.py
```

## Scripts

One-time utilities in `scripts/` folder (not part of main bot):
- fetch_market_info.py
- initialize_bot_state.py
- fix_pending_orders.py
- etc.

## Security

- Never hardcode API keys - use `.env`
- Circuit breaker prevents API hammering
- Health check validates connectivity on startup
