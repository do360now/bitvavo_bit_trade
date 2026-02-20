#!/usr/bin/env python3
"""
Test: Old vs New Interface Comparison

This demonstrates how much simpler the refactored code is to use.
"""

print("=" * 70)
print("OLD INTERFACE (Shallow Modules)")
print("=" * 70)
print("""
# User must understand cycles, phases, drawdowns, multipliers...

from bitcoin_cycle_detector import BitcoinCycleDetector
from cycle_aware_strategy import CycleAwareStrategy

# Initialize (needs 5 objects!)
detector = BitcoinCycleDetector(current_cycle=4)
strategy = CycleAwareStrategy(
    cycle_detector=detector,
    state_manager=state_manager,
    trade_executor=trade_executor,
    atomic_trade_manager=atomic_trade_manager,
    base_position_pct=0.10
)

# Get cycle analysis (user must understand these concepts)
cycle_summary = detector.get_cycle_summary(price=58000)
phase = cycle_summary['cycle_phase']
confidence = cycle_summary['confidence']
multiplier = cycle_summary['position_multiplier']
drawdown = cycle_summary['drawdown_from_ath']

# Get risk assessment (another method call)
risk = strategy.get_risk_assessment(price=58000)

# Check if should buy (complex return value)
should_buy, btc_amount, reasoning = strategy.should_buy(
    current_price=58000,
    available_eur=1000
)

# Check if should sell (yet another method)
should_sell, btc_amount, reasoning = strategy.should_sell(
    current_price=58000,
    btc_balance=0.037
)

# Finally, execute
if should_buy:
    execute_buy(btc_amount, price)
elif should_sell:
    execute_sell(btc_amount, price)
else:
    hold()

# TOTAL: ~25 lines of code, 8+ method calls, must understand cycles
""")

print("\n")
print("=" * 70)
print("NEW INTERFACE (Deep Module)")
print("=" * 70)
print("""
# User doesn't need to understand cycles at all!

from cycle_trading_deep_module import CycleAwareTrading

# Initialize (just 1 object, 2 parameters)
trading = CycleAwareTrading(base_position_pct=0.10)

# Get decision (just 1 method call!)
decision = trading.decide(
    price=58000,
    btc_held=0.037,
    eur_available=1000,
    avg_buy_price=81000
)

# Execute
if decision.should_buy:
    execute_buy(decision.btc_amount, decision.price)
elif decision.should_sell:
    execute_sell(decision.btc_amount, decision.price)
else:
    hold()

# TOTAL: ~10 lines of code, 1 method call, zero cycle knowledge needed
""")

print("\n")
print("=" * 70)
print("ACTUAL WORKING EXAMPLE")
print("=" * 70)

# Import the new module
from cycle_trading_deep_module import CycleAwareTrading

# Create trading instance
trading = CycleAwareTrading(base_position_pct=0.10)

# Make decision at current price (€58,000)
print("\nScenario: BTC @ €58,000, you bought @ €81,000")
print("-" * 70)

decision = trading.decide(
    price=58000,
    btc_held=0.037,
    eur_available=1000,
    avg_buy_price=81000
)

print(f"Should Buy: {decision.should_buy}")
print(f"Should Sell: {decision.should_sell}")
print(f"Amount: {decision.btc_amount:.8f} BTC")
print(f"Reasoning: {decision.reasoning}")
print(f"Risk Level: {decision.risk_level}")
print(f"Phase: {decision.phase}")

# Get market context (optional)
print("\nMarket Context (optional):")
print("-" * 70)
context = trading.get_market_context(58000)
for key, value in context.items():
    if isinstance(value, float):
        if 'pct' in key:
            print(f"{key}: {value:.1%}")
        else:
            print(f"{key}: €{value:,.0f}" if value > 100 else f"{key}: {value:.2f}")
    else:
        print(f"{key}: {value}")

# Test at different prices
print("\n" + "=" * 70)
print("DECISION AT DIFFERENT PRICES")
print("=" * 70)

test_prices = [
    (80000, "Near breakeven"),
    (58000, "Current correction"),
    (45000, "Mid correction"),
    (36000, "Deep correction"),
    (28000, "Near historical bottom"),
]

for price, description in test_prices:
    decision = trading.decide(
        price=price,
        btc_held=0.037,
        eur_available=1000,
        avg_buy_price=81000
    )
    
    action = "BUY" if decision.should_buy else "SELL" if decision.should_sell else "HOLD"
    print(f"\n€{price:,} ({description})")
    print(f"  Action: {action}")
    print(f"  Phase: {decision.phase}")
    print(f"  Risk: {decision.risk_level}")
    if decision.should_buy:
        print(f"  Buy: {decision.btc_amount:.8f} BTC (€{decision.btc_amount * price:,.2f})")
    print(f"  Reason: {decision.reasoning}")

print("\n" + "=" * 70)
print("SIMPLICITY ACHIEVED! ✅")
print("=" * 70)
print("""
KEY IMPROVEMENTS:
✅ Interface: 1 method instead of 8+
✅ Parameters: 2 instead of 5 objects
✅ Code lines: 4 instead of 25
✅ Cognitive load: Minimal - no cycle knowledge needed
✅ Flexibility: Same power, simpler interface
✅ Testability: Pure function, easy to test
✅ Maintainability: All logic in one place

This is what Ousterhout calls a "DEEP MODULE":
- Simple interface (decide)
- Complex implementation (hidden inside)
- Information hiding (cycles are internal)
- Pull complexity downward (users don't suffer)
""")
