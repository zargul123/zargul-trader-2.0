import sys
import os
import pandas as pd

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.backtest.backtest_engine import BacktestEngine
from scripts.core.analysis_engine import AIAnalyst

def run_simple_backtest():
    """
    Runs a straightforward backtest with detailed logging for a single asset and strategy.
    """
    symbol = "BTC-USD"
    strategy = "main"
    days = 90

    print("="*80)
    print(f"STARTING SIMPLE BACKTEST for {symbol} / {strategy} / {days} days")
    print("="*80)

    # Initialize the AI Analyst specifically for the model we need.
    # This is much faster than loading all models.
    print("\nInitializing AI Analyst for the specific model...")
    
    # Auto-train BTC main model if missing
    import os
    model_files = [f'trained_models/{symbol}_{strategy}_model.h5', f'trained_models/{symbol}_{strategy}_scaler.joblib', f'trained_models/{symbol}_{strategy}_calibrator.joblib']
    train_needed = not all(os.path.exists(f) for f in model_files)
    
    analyst = AIAnalyst(symbol=symbol, strategy_type=strategy, train_all=train_needed)
    print("✅ AI Analyst ready.")

    # Initialize the backtest engine with the pre-loaded analyst
    engine = BacktestEngine(analyst=analyst, debug=True)

    # Run the backtest
    metrics = engine.run_backtest(symbol=symbol, strategy_type=strategy, days=days)

    # Print the final report
    engine.generate_report(symbol)

    print("\n="*80)
    print("SIMPLE BACKTEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    run_simple_backtest()
