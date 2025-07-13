
#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.backtest.backtest_engine import BacktestEngine

def main():
    print("🧪 Testing Simple Backtest for BTC-USD")
    
    # Initialize backtest engine
    engine = BacktestEngine(debug=True)
    
    # Run a short 7-day backtest
    print("\n📊 Running 7-day backtest...")
    engine.run_backtest("BTC-USD", "main", 7)
    
    # Show results
    print(f"\n📈 RESULTS:")
    print(f"Total trades executed: {len(engine.trade_history)}")
    
    if engine.trade_history:
        print("\nFirst 3 trades:")
        for i, trade in enumerate(engine.trade_history[:3]):
            print(f"Trade {i+1}: {trade['type']} | PnL: {trade['pnl']:.2f}%")
    else:
        print("❌ No trades found - check AI model predictions")

if __name__ == "__main__":
    main()
