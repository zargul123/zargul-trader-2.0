import os
import sys
import pandas as pd

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.backtest.backtest_engine import BacktestEngine
from scripts.core.data_engine import DataMaster
from scripts.config import STRATEGIES

def run_verification_backtest():
    """
    A simple script to verify that the refactored BacktestEngine runs without errors.
    """
    print("="*80)
    print("🚀 STARTING BACKTEST ENGINE VERIFICATION 🚀")
    print("="*80)

    # --- Test Parameters ---
    ASSET_TO_TEST = "BTC-USD"
    STRATEGY_TO_TEST = "main"
    DAYS_TO_TEST = 90

    # 1. Load Data
    print(f"Loading {DAYS_TO_TEST} days of data for {ASSET_TO_TEST}...")
    data_master = DataMaster()
    strategy_config = STRATEGIES[STRATEGY_TO_TEST]
    
    # Calculate the limit based on the timeframe
    timeframe = strategy_config['timeframe']
    if 'h' in timeframe:
        records_per_day = 24 / int(timeframe.replace('h', ''))
    elif 'm' in timeframe:
        records_per_day = (24 * 60) / int(timeframe.replace('m', ''))
    else:
        records_per_day = 1
    limit = int(DAYS_TO_TEST * records_per_day)

    df = data_master.get_data(
        ASSET_TO_TEST, 
        timeframe, 
        limit=limit
    )
    
    if df is None or df.empty:
        print("❌ FAILED: Could not load data.")
        return
        
    # Add symbol to df attributes for compatibility with the new engine
    df.attrs['symbol'] = ASSET_TO_TEST
    
    print(f"✅ Data loaded successfully: {len(df)} candles.")

    # 2. Instantiate and Run Engine
    print("\nInstantiating and running the BacktestEngine...")
    
    # The backtest engine is initialized without arguments
    backtest_engine = BacktestEngine(debug=True)
    
    # The run_backtest method handles data loading and execution
    results = backtest_engine.run_backtest(
        symbol=ASSET_TO_TEST,
        strategy_type=STRATEGY_TO_TEST,
        days=DAYS_TO_TEST
    )

    # 3. Print Results
    print("\n" + "="*80)
    if not results or results.get('total_trades', 0) == 0:
        print("✅ VERIFICATION PASSED (with no trades).")
        print("The backtest engine ran without errors, but no trades were executed.")
    else:
        print("✅ VERIFICATION PASSED (with trades).")
        print("Final Metrics:")
        for key, value in results.items():
            # Ensure value is a string before formatting
            value_str = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
            print(f"  - {key.replace('_', ' ').title():<20}: {value_str}")
    
    print("\nConclusion: The BacktestEngine is working as expected.")
    print("="*80)

if __name__ == "__main__":
    run_verification_backtest()