
from scripts.backtest.backtest_engine import BacktestEngine
from scripts.config import ASSETS

def main():
    print("🚀 Starting Zargul Trader Backtest\n")
    backtester = BacktestEngine()
    
    for asset in ASSETS:
        # Test all strategies
        for strategy in ['main', 'swing', 'scalp']:
            try:
                print(f"\n🔍 Testing {asset} - {strategy.upper()} Strategy")
                metrics = backtester.run_backtest(
                    symbol=asset,
                    strategy_type=strategy,  # Changed from 'strategy'
                    days=30
                )
                
                print(f"\n📊 Results for {asset} ({strategy.upper()}):")
                for k, v in metrics.items():
                    print(f"{k.replace('_', ' ').title()}: {round(v, 2)}")

                # Generate report
                backtester.generate_report(asset)
                
            except Exception as e:
                print(f"❌ Error testing {asset} with {strategy} strategy: {str(e)}")
                continue

    print("\n✅ All backtests completed! Check the backtest_reports folder")

if __name__ == "__main__":
    main()
