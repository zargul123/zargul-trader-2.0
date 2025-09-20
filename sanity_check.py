import sys
import os
import pandas as pd

# Add project root to path to allow script imports
# This ensures that when you run `python sanity_check.py`, it can find the `scripts` module
sys.path.insert(0, os.getcwd())

from scripts.core.data_engine import DataMaster

print('Fetching data for ATR sanity check...')
# Use DataMaster to get the same data the bot uses
df = DataMaster().get_training_data('BTC-USD', '1h', days=1095)

if df is not None and not df.empty and 'atr_norm' in df.columns:
    # The 'atr_norm' column is already ATR as a percentage of close price
    median_atr_pct = df['atr_norm'].median()
    quantile_90_atr_pct = df['atr_norm'].quantile(0.9)

    print('\n--- ATR Sanity Check Results ---')
    print(f'Median ATR (as % of price): {median_atr_pct:.2f}%')
    print(f'90th Percentile ATR (as % of price): {quantile_90_atr_pct:.2f}%')
    print('------------------------------------')
else:
    print('Failed to fetch data or \'atr_norm\' column not found.')
