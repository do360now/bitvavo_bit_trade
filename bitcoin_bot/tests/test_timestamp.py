#!/usr/bin/env python3
import time
from datetime import datetime

# Test the timestamps from your logs
test_timestamps = [
    1754219700,  # From your logs
    1754220600,
    1754221500,
    1754222400,
    1754223300
]

current_time = int(time.time())
print(f"Current time: {current_time} ({datetime.fromtimestamp(current_time)})")
print()

for ts in test_timestamps:
    dt = datetime.fromtimestamp(ts)
    diff_hours = (ts - current_time) / 3600
    print(f"Timestamp: {ts}")
    print(f"Date/Time: {dt}")
    print(f"Difference from now: {diff_hours:.1f} hours")
    print(f"Valid? {abs(diff_hours) < 24}")  # Within 24 hours should be fine
    print()