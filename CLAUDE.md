# Bitcoin Trading Bot v2 - Claude Guidance

Macro-aware profit engine on Bitvavo. Actively trades buy AND sell.

## Running

```bash
python3 main_v2.py    # v2 profit engine
python3 main.py       # legacy accumulation
```

## Core Modules (v2)

| File | Purpose |
|------|---------|
| main_v2.py | Orchestration: OHLCV + macro → ProfitEngine → execute |
| profit_engine.py | **NEW** Multi-signal active trading strategy |
| bitvavo_api.py | CCXT wrapper with circuit breaker |
| trade_executor.py | Order execution, order book pricing |
| order_manager.py | Order tracking |
| atomic_trade_manager.py | Atomic state updates |
| indicators.py | Technical indicators, news sentiment, correlations |
| config.py | All configuration |

## Signal Architecture

```
Composite = Technical(0.45) + Momentum(0.25) + Cycle(0.15) + Volatility(0.15)
            × MacroRegime dampener
Range: -1.0 (strong sell) to +1.0 (strong buy)
```

## v2 Bug Fixes
- Duplicate `_health_check` and `_check_trailing_stop` in main.py
- `atomic_sell()` → `execute_sell()` (method didn't exist)
- RSI: SMA → Wilder smoothing
