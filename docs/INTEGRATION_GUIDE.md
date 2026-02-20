"""
INTEGRATION GUIDE: Adding Cycle Awareness to Your Bitcoin Bot
===============================================================

This guide shows exactly how to integrate the cycle-aware strategy
into your existing trading bot with minimal code changes.


STEP 1: Add New Files to Your Project
======================================

Copy these 3 new files to your project directory:
1. bitcoin_cycle_detector.py
2. cycle_aware_strategy.py  
3. CRITICAL_MARKET_ANALYSIS_FEB_2026.py (reference only)


STEP 2: Update config.py
=========================

Add these new configuration parameters:

```python
# Cycle-aware trading parameters
CYCLE_CONFIG = {
    'ENABLED': True,
    'CURRENT_CYCLE': 4,  # 2024 halving = cycle 4
    'BASE_POSITION_PCT': 0.10,  # 10% base position size
    'CYCLE_ADJUSTMENTS': True,  # Allow cycle to adjust position sizes
    'MIN_DRAWDOWN_FOR_BUY': 0.30,  # Don't buy unless 30%+ down from ATH
    'AGGRESSIVE_ACCUMULATION_THRESHOLD': 0.70,  # 70%+ down = aggressive buy
}

# Override existing params for bear market (Feb 2026)
TRADING_PARAMS = {
    'USE_STOP_LOSS': True,
    'STOP_LOSS_PERCENT': 0.05,  # Tighter 5% stops in bear market
    'USE_TAKE_PROFIT': True,
    'TAKE_PROFIT_PERCENT': 0.20,  # 20% take profit (was 8%)
}
```


STEP 3: Modify main.py - Option A (Minimal Changes)
====================================================

If you want minimal changes to your existing main.py:

```python
# At the top of main.py, add imports:
from bitcoin_cycle_detector import BitcoinCycleDetector
from cycle_aware_strategy import CycleAwareStrategy
from config import CYCLE_CONFIG

# After initializing your existing components, add:
if CYCLE_CONFIG['ENABLED']:
    cycle_detector = BitcoinCycleDetector(
        current_cycle=CYCLE_CONFIG['CURRENT_CYCLE']
    )
    
    cycle_strategy = CycleAwareStrategy(
        cycle_detector=cycle_detector,
        state_manager=bot_state,
        trade_executor=trade_executor,
        atomic_trade_manager=atomic_trade_manager,
        base_position_pct=CYCLE_CONFIG['BASE_POSITION_PCT']
    )
    
    logger.info("✅ Cycle-aware trading ENABLED")
else:
    cycle_strategy = None
    logger.info("ℹ️ Cycle-aware trading DISABLED - using legacy strategy")

# In your main trading loop, REPLACE the buy/sell decision logic with:

if cycle_strategy:
    # Use cycle-aware strategy
    trade_result = cycle_strategy.execute_cycle_aware_trade(
        current_price=current_price,
        available_eur=available_eur,
        btc_balance=btc_balance
    )
    
    # The strategy handles everything - just log result
    logger.info(f"Cycle-aware result: {trade_result['action']} - {trade_result['reasoning']}")

else:
    # Your existing trading logic here
    # (keep your current buy/sell decisions as fallback)
    pass
```


STEP 4: Modify main.py - Option B (Full Integration)
=====================================================

For complete integration, replace your main loop with:

```python
def main():
    # Initialize all your existing components
    bitvavo_api = EnhancedBitvavoAPI(api_key, api_secret)
    bot_state = BotStateManager()
    trade_executor = TradeExecutor(bitvavo_api)
    order_manager = OrderManager(bitvavo_api, trade_executor)
    atomic_trade_manager = AtomicTradeManager(order_manager, bot_state, trade_executor)
    
    # Initialize cycle detector
    cycle_detector = BitcoinCycleDetector(current_cycle=4)
    
    # Initialize cycle-aware strategy
    strategy = CycleAwareStrategy(
        cycle_detector=cycle_detector,
        state_manager=bot_state,
        trade_executor=trade_executor,
        atomic_trade_manager=atomic_trade_manager,
        base_position_pct=0.10
    )
    
    logger.info("🚀 Starting Cycle-Aware Bitcoin Trading Bot")
    
    # Print initial market analysis
    current_price = trade_executor.get_current_price()
    if current_price:
        cycle_detector.print_cycle_analysis(current_price)
    
    iteration = 0
    
    while True:
        try:
            iteration += 1
            logger.info(f"\n{'='*70}")
            logger.info(f"ITERATION #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'='*70}")
            
            # Get current market state
            current_price = trade_executor.get_current_price()
            available_eur = trade_executor.get_available_balance("EUR")
            btc_balance = trade_executor.get_total_btc_balance()
            
            if not current_price:
                logger.error("Failed to get price, skipping iteration")
                time.sleep(300)
                continue
            
            # Process filled orders
            fill_results = atomic_trade_manager.process_filled_orders()
            
            # Print balances
            logger.info(f"💰 Balances: {btc_balance:.8f} BTC | €{available_eur:,.2f} EUR")
            logger.info(f"📊 BTC Price: €{current_price:,.2f}")
            
            # Get risk assessment
            risk_assessment = strategy.get_risk_assessment(current_price)
            logger.info(f"⚠️ Risk Level: {risk_assessment['risk_level']}")
            
            # Print any warnings
            if risk_assessment['warnings']:
                for warning in risk_assessment['warnings']:
                    logger.warning(warning)
            
            # Execute cycle-aware trading
            trade_result = strategy.execute_cycle_aware_trade(
                current_price=current_price,
                available_eur=available_eur,
                btc_balance=btc_balance
            )
            
            logger.info(f"📈 Action: {trade_result['action']}")
            logger.info(f"💭 Reason: {trade_result['reasoning']}")
            
            if trade_result['order_id']:
                logger.info(f"📝 Order ID: {trade_result['order_id']}")
            
            # Print cycle statistics every 10 iterations
            if iteration % 10 == 0:
                cycle_stats = strategy.get_cycle_statistics()
                logger.info("\n" + "="*70)
                logger.info("CYCLE STATISTICS")
                logger.info("="*70)
                for key, value in cycle_stats.items():
                    logger.info(f"{key}: {value}")
                logger.info("="*70)
            
            # Sleep between iterations
            sleep_duration = SLEEP_DURATION  # From config (e.g., 900 = 15 min)
            logger.info(f"😴 Sleeping for {sleep_duration//60} minutes...")
            time.sleep(sleep_duration)
            
        except KeyboardInterrupt:
            logger.info("\n🛑 Bot stopped by user")
            break
            
        except Exception as e:
            logger.error(f"❌ Error in main loop: {e}", exc_info=True)
            logger.info("⏸️ Waiting 60 seconds before retry...")
            time.sleep(60)
    
    logger.info("👋 Bot shutdown complete")


if __name__ == "__main__":
    main()
```


STEP 5: Testing & Validation
=============================

Before running live, test the integration:

1. Test cycle detection:
```python
python CRITICAL_MARKET_ANALYSIS_FEB_2026.py
```

This will show current market analysis and recommendations.

2. Dry-run test:
```python
# Add to your config.py temporarily:
DRY_RUN_MODE = True  # Don't actually place orders

# Then run your bot - it will log decisions without executing
```

3. Check these outputs:
   - ✅ Cycle phase correctly identified (should be CORRECTION)
   - ✅ Position multiplier is reduced (should be ~0.5x)
   - ✅ Buy signals are cautious (should say "wait for deeper drop")
   - ✅ Stop losses are tight (should be 5%)


STEP 6: Current Market Configuration (Feb 2026)
================================================

For the current bear market, your bot should show:

```
Phase: CORRECTION
Confidence: 75-85%
Risk Level: HIGH
Position Multiplier: 0.5x
Buy Signal: HOLD or LIGHT_BUY
Recommended Action: Wait for €36k EUR and below
Estimated Bottom: €23,000 - €28,000 EUR
Current Price: €59,000 EUR
ATH: €114,395 EUR
Drawdown: -48.4%
```

If you see different values, the cycle detector may need adjustment.


STEP 7: Monitor These Metrics
==============================

Track these in your logs to ensure it's working:

1. **Cycle Phase Changes**: Watch for transitions
   - CORRECTION → ACCUMULATION (when drawdown hits 70%+)
   - ACCUMULATION → GROWTH (when price recovers)

2. **Position Sizing**: Verify it's adjusting (EUR)
   - At €59k: Small positions (0.5x multiplier)
   - At €36k: Moderate positions (1.0x multiplier)
   - At €28k: Large positions (2.0x multiplier)

3. **Stop Losses**: Should be tight in bear market
   - Current: 5%
   - Will widen in bull market (15-25%)

4. **Buy Signals** (EUR):
   - At €59k: Minimal/None
   - At €45k: Light buy
   - At €36k: Moderate buy
   - At €28k: Strong buy
   - Below €23k: Extreme buy


STEP 8: Rollback Plan
======================

If something goes wrong, you can quickly disable:

```python
# In config.py:
CYCLE_CONFIG = {
    'ENABLED': False,  # ← Set to False
    # ... rest of config
}
```

This will use your original trading logic.


TROUBLESHOOTING
===============

Issue: "Bot still buying aggressively at €59k EUR"
Fix: Check CYCLE_CONFIG['ENABLED'] = True and MIN_DRAWDOWN_FOR_BUY = 0.30

Issue: "Cycle phase shows BUBBLE but we're clearly in correction"
Fix: Update CURRENT_CYCLE and verify cycle_ath = 114395 EUR in detector

Issue: "No buy signals even at €28k EUR"  
Fix: Check MIN_DRAWDOWN_FOR_BUY isn't set too high (should be 0.30-0.40)

Issue: "Position sizes still too large"
Fix: Reduce BASE_POSITION_PCT to 0.05 or 0.03 for more conservative


EXPECTED BEHAVIOR SUMMARY
==========================

At current price (€59k EUR):
- ❌ Should NOT be aggressively buying
- ✅ Should be preserving capital
- ⏳ Should be waiting for €36k EUR and below
- 📊 Should show CORRECTION phase
- ⚠️ Should warn about early bear market

At €36k EUR (future):
- 🟡 Start moderate buying (10-15% positions)
- 📊 Transitioning to ACCUMULATION phase
- ✅ Historical bottom zone approaching

At €23-28k EUR (future):
- 🟢🟢 Aggressive buying (20-30% positions)
- 💰 Deep value zone
- 🎯 Likely near cycle bottom


CRITICAL REMINDER
=================

The cycle detector is based on HISTORICAL PATTERNS. Bitcoin can:
- Deviate from past cycles
- React to new macro factors
- Experience unprecedented events

Always:
- Monitor manually
- Use stop losses
- Keep cash reserves
- Don't bet everything on historical patterns
- Adjust as new information emerges

The goal is to IMPROVE your edge, not guarantee profits.


NEXT STEPS
==========

1. ✅ Copy the 3 new files to your project
2. ✅ Update config.py with CYCLE_CONFIG
3. ✅ Integrate into main.py (Option A or B)
4. ✅ Test in dry-run mode
5. ✅ Monitor first 24 hours closely
6. ✅ Adjust parameters based on behavior
7. ✅ Review this guide monthly as market evolves

Good luck, and remember: In bear markets, patience is your greatest weapon. 🐻❄️
"""
