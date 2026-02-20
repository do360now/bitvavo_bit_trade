# 🚀 Quick Cleanup Reference

## Before You Start

```bash
# 1. Verify everything is safe
python3 verify_before_cleanup.py

# 2. Backup everything (just in case)
tar -czf bot_backup_$(date +%Y%m%d).tar.gz *.py *.json *.csv *.md
```

## Option A: Minimal Cleanup (Recommended - 2 minutes)

**What it does:**
- Archives old shallow modules
- Deletes test/cache files
- Keeps current structure

```bash
# Make script executable
chmod +x cleanup_minimal.sh

# Run it
./cleanup_minimal.sh
```

**Or manual one-liner:**
```bash
mkdir -p archive && mv bitcoin_cycle_detector.py cycle_aware_strategy.py archive/ 2>/dev/null; mv main.bak archive/ 2>/dev/null; rm -rf __pycache__ tests/__pycache__ backup; rm -f test_interface_comparison.py test_logs.csv test_prices.json tree.tx; echo "✅ Done!"
```

## Option B: Full Reorganization (Advanced - 15 minutes)

**What it does:**
- Creates professional directory structure
- Organizes code into core/, utils/, data/, docs/
- Much cleaner but requires import updates

```bash
# Make script executable
chmod +x cleanup_full.sh

# Run it
./cleanup_full.sh

# Then update imports in main.py
# (Manual step - see CLEANUP_GUIDE.md)
```

## After Cleanup

```bash
# 1. Verify bot still works
python3 main.py

# Expected: Bot starts, shows "✅ Cycle-aware trading enabled"

# 2. If it works, you can delete archive/ later
rm -rf archive/

# 3. Optional: Remove cleanup scripts
rm cleanup_*.sh verify_before_cleanup.py CLEANUP_GUIDE.md
```

## Quick File Status

| File | Keep? | Why |
|------|-------|-----|
| `cycle_trading_deep_module.py` | ✅ YES | NEW refactored code |
| `bitcoin_cycle_detector.py` | 📦 Archive | OLD code (replaced) |
| `cycle_aware_strategy.py` | 📦 Archive | OLD code (replaced) |
| `main.py` | ✅ YES | Entry point |
| `bot_state.json` | ✅ YES | **CRITICAL - trading state!** |
| `test_*.py` | 🗑️ Delete | Test files |
| `__pycache__/` | 🗑️ Delete | Cache (regenerates) |

## Emergency: Undo Cleanup

```bash
# Restore from backup
tar -xzf bot_backup_*.tar.gz

# Or restore from archive
cp archive/*.py ./
```

## My Recommendation

**Just run the minimal cleanup:**

```bash
chmod +x cleanup_minimal.sh
./cleanup_minimal.sh
python3 main.py  # Verify it works
```

**Done!** Your directory is now clean and organized. 🎉

---

**Questions?** Check `CLEANUP_GUIDE.md` for full details.
