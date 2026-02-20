#!/usr/bin/env python3
"""
Pre-Cleanup Verification Script

Run this BEFORE cleanup to see what would be affected.
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath):
    """Check if file exists and return size"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        return True, size
    return False, 0

def format_size(size):
    """Format file size"""
    for unit in ['B', 'KB', 'MB']:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"

def main():
    print("=" * 70)
    print("PRE-CLEANUP VERIFICATION")
    print("=" * 70)
    print()
    
    # Check critical files (must exist)
    print("🔍 Checking CRITICAL files (must exist)...")
    critical_files = [
        'main.py',
        'config.py',
        'cycle_trading_deep_module.py',
        'bot_state.json',
        'order_history.json',
    ]
    
    critical_ok = True
    for file in critical_files:
        exists, size = check_file_exists(file)
        if exists:
            print(f"  ✅ {file} ({format_size(size)})")
        else:
            print(f"  ❌ {file} - MISSING!")
            critical_ok = False
    
    if not critical_ok:
        print()
        print("⚠️  CRITICAL FILES MISSING! Do not proceed with cleanup.")
        return 1
    
    # Check old files (will be archived)
    print()
    print("📦 Checking OLD files (will be archived)...")
    old_files = [
        'bitcoin_cycle_detector.py',
        'cycle_aware_strategy.py',
        'main.bak',
    ]
    
    old_count = 0
    for file in old_files:
        exists, size = check_file_exists(file)
        if exists:
            print(f"  📦 {file} ({format_size(size)}) - will archive")
            old_count += 1
        else:
            print(f"  ⏭️  {file} - not found (already cleaned?)")
    
    # Check temp files (will be deleted)
    print()
    print("🗑️  Checking TEMP files (will be deleted)...")
    temp_files = [
        'test_interface_comparison.py',
        'test_logs.csv',
        'test_prices.json',
        'tree.tx',
    ]
    
    temp_count = 0
    temp_size = 0
    for file in temp_files:
        exists, size = check_file_exists(file)
        if exists:
            print(f"  🗑️  {file} ({format_size(size)}) - will delete")
            temp_count += 1
            temp_size += size
        else:
            print(f"  ⏭️  {file} - not found")
    
    # Check cache (will be deleted)
    print()
    print("🗑️  Checking CACHE directories (will be deleted)...")
    cache_dirs = ['__pycache__', 'tests/__pycache__']
    
    cache_count = 0
    for dir in cache_dirs:
        if os.path.exists(dir):
            files = list(Path(dir).rglob('*.pyc'))
            print(f"  🗑️  {dir}/ ({len(files)} .pyc files) - will delete")
            cache_count += len(files)
        else:
            print(f"  ⏭️  {dir}/ - not found")
    
    # Check if main.py uses deep module
    print()
    print("🔍 Checking main.py configuration...")
    if os.path.exists('main.py'):
        with open('main.py', 'r') as f:
            content = f.read()
            if 'cycle_trading_deep_module' in content:
                print("  ✅ main.py uses refactored deep module")
            else:
                print("  ⚠️  main.py may not be using deep module!")
                print("  You may need to update main.py after cleanup")
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Critical files: ✅ All present")
    print(f"Files to archive: {old_count}")
    print(f"Files to delete: {temp_count} ({format_size(temp_size)})")
    print(f"Cache files to delete: {cache_count}")
    print()
    
    if old_count > 0 or temp_count > 0 or cache_count > 0:
        print("✅ Cleanup is recommended")
        print()
        print("Run cleanup:")
        print("  Minimal:  bash cleanup_minimal.sh")
        print("  Full:     bash cleanup_full.sh")
    else:
        print("ℹ️  Directory is already clean!")
    
    print()
    return 0

if __name__ == '__main__':
    sys.exit(main())
