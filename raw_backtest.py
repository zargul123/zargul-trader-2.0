import sys
import os
import argparse
import pandas as pd
from copy import deepcopy

# Add project root to path to allow script imports
sys.path.insert(0, os.getcwd())

from scripts.backtest.backtest_engine import BacktestEngine
from scripts.config import ASSETS, STRATEGIES

def run_raw_backtest(asset, strategy, confidence_override):
    """
    Runs a backtest that overrides ALL risk parameters to test the raw
    AI signal without interference from the risk engine's thresholds.
    """
    print("="*80)
    print("🚀 STARTING RAW SIGNAL BACKTEST 🚀")
    print(f"Asset: {asset} | Strategy: {strategy} | Confidence Override: {confidence_override}")
    print("="*80)

    try:
        # 1. Create a deep copy of the strategy config to modify safely
        temp_config = deepcopy(STRATEGIES[asset][strategy])
        
        # 2. Override ALL risk thresholds to effectively disable them
        print("   - Overriding risk parameters for raw signal analysis...")
        regime_types = ['Trending', 'Ranging', 'Chaotic']
        for regime in regime_types:
            if regime in temp_config:
                # Set confidence to the low override value
                temp_config[regime]['min_confidence'] = confidence_override
                # Set ATR threshold to near-zero to ensure it always passes
                temp_config[regime]['atr_threshold_multiplier'] = 0.0001 
                # --- DEFINITIVE FIX: Also override the static thresholds ---
                temp_config[regime]['long_threshold'] = 0.0001
                temp_config[regime]['short_threshold'] = 0.0001 
        
        # Also explicitly enable the Chaotic regime for a true raw test
        if 'Chaotic' in temp_config:
            temp_config['Chaotic']['enabled'] = True
        
        # 3. Initialize and run the backtest engine
        engine = BacktestEngine()
        results = engine.run_backtest(
            symbol=asset,
            strategy_type=strategy,
            days=1095, # Use 3 years of data for a robust test
            temp_strategy_config=temp_config
        )

        # 4. Print the results
        print("\n" + "="*80)
        print("✨ RAW BACKTEST COMPLETE ✨")
        if results:
            results_series = pd.Series(results)
            print(results_series.to_string())
        else:
            print("No results were generated.")
        print("="*80)

    except KeyError:
        print(f"❌ ERROR: Strategy '{strategy}' not found for asset '{asset}' in config.py.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    all_strategies = set()
    for asset_strategies in STRATEGIES.values():
        for strategy_name in asset_strategies.keys():
            all_strategies.add(strategy_name)
    strategy_choices = sorted(list(all_strategies))

    parser = argparse.ArgumentParser(description="Run a raw signal backtest for a given strategy.")
    parser.add_argument('--asset', type=str, required=True, choices=ASSETS, help="Asset to test")
    parser.add_argument('--strategy', type=str, required=True, choices=strategy_choices, help="Strategy to test")
    parser.add_argument('--confidence', type=float, required=True, help="Confidence threshold to override (e.g., 0.40)")
    args = parser.parse_args()

    run_raw_backtest(args.asset, args.strategy, args.confidence)
