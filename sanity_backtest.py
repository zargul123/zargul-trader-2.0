import sys
import os
import argparse
import pandas as pd

# Add project root to path to allow script imports
sys.path.insert(0, os.getcwd())

from scripts.backtest.backtest_engine import BacktestEngine
from scripts.config import ASSETS, STRATEGIES

def run_sanity_backtest(asset, strategy, confidence_override):
    """
    Runs a simple, non-optimized backtest with a fixed confidence threshold.
    """
    print("="*80)
    print("🚀 STARTING SANITY BACKTEST 🚀")
    print(f"Asset: {asset} | Strategy: {strategy} | Confidence Override: {confidence_override}")
    print("="*80)

    try:
        # 1. Create a temporary, modified strategy config
        # We are overriding the confidence level for this test
        temp_config = STRATEGIES[asset][strategy].copy()
        
        # The confidence is now set inside the regime dictionaries
        if 'Trending' in temp_config:
            temp_config['Trending']['min_confidence'] = confidence_override
        if 'Ranging' in temp_config:
            temp_config['Ranging']['min_confidence'] = confidence_override
        if 'Chaotic' in temp_config:
            # We still respect if the Chaotic regime is disabled
            if temp_config['Chaotic'].get('enabled', False):
                 temp_config['Chaotic']['min_confidence'] = confidence_override
        
        # 2. Initialize the backtest engine
        # We pass the specific symbol and strategy to the AI Analyst for fast, focused loading
        engine = BacktestEngine()

        # 3. Run the backtest
        # We use 1095 days (3 years) of data for a robust test
        results = engine.run_backtest(
            symbol=asset,
            strategy_type=strategy,
            days=1095,
            temp_strategy_config=temp_config
        )

        # 4. Print the results
        print("\n" + "="*80)
        print("✨ SANITY BACKTEST COMPLETE ✨")
        if results:
            # Use pandas to format the output nicely
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

    parser = argparse.ArgumentParser(description="Run a sanity backtest for a given strategy.")
    parser.add_argument('--asset', type=str, required=True, choices=ASSETS, help="Asset to test (e.g., 'BTC-USD')")
    parser.add_argument('--strategy', type=str, required=True, choices=strategy_choices, help="Strategy to test (e.g., 'main')")
    parser.add_argument('--confidence', type=float, required=True, help="Confidence threshold to override (e.g., 0.50)")
    args = parser.parse_args()

    run_sanity_backtest(args.asset, args.strategy, args.confidence)
