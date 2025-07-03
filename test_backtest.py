from scripts.backtest.backtest_engine import BacktestEngine
from scripts.config import ASSETS

def main():
    print("🚀 Starting Zargul Trader Backtest")

    # Initialize backtester
    backtester = BacktestEngine()

    # Run backtest for each asset
    for asset in ASSETS:
        print(f"\n🔍 Backtesting {asset}...")
        metrics = backtester.run_backtest(asset, strategy="main", days=90)

        print("\n📊 Results:")
        for k, v in metrics.items():
            print(f"{k.replace('_', ' ').title()}: {round(v, 2)}")

        # Generate report
        backtester.generate_report(asset)

    print("\n✅ All backtests completed! Check the backtest_reports folder")

if __name__ == "__main__":
    main()