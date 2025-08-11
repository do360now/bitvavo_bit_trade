#!/usr/bin/env python3
import json

# Manually create valid price history from your logs
valid_candles = [
    [1754219700, 98850.0, 98855.0, 98685.0, 98685.0, 13.37195015],
    [1754220600, 98665.0, 98680.0, 98587.0, 98621.0, 3.09386794],
    [1754221500, 98646.0, 98711.0, 98540.0, 98670.0, 6.18975744],
    [1754222400, 98539.0, 98550.0, 98381.0, 98420.0, 3.71712348],
    [1754223300, 98420.0, 98474.0, 98340.0, 98460.0, 2.61625445],
    [1754224200, 98459.0, 98676.0, 98452.0, 98665.0, 5.1826598],
    [1754225100, 98660.0, 98760.0, 98641.0, 98660.0, 4.40895256],
    [1754226000, 98661.0, 98729.0, 98599.0, 98719.0, 3.67522442],
    [1754226900, 98723.0, 98729.0, 98506.0, 98539.0, 1.49503036],
    [1754227800, 98539.0, 98539.0, 98539.0, 98539.0, 0.00334989]
]

# Save to price history file
with open('./price_history.json', 'w') as f:
    json.dump(valid_candles, f)

print("✅ Manually created price history with 10 valid candles")
print(f"Price range: €{min(c[4] for c in valid_candles):.2f} to €{max(c[4] for c in valid_candles):.2f}")
print("You can now run: python3 main.py")
