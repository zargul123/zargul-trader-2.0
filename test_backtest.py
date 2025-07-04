
import argparse
from scripts.backtest.backtest_engine import BacktestEngine
from scripts.config import ASSETS, STRATEGIES

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=90)
    parser.add_argument('--timeframe', type=str, default='4h')
    parser.add_argument('--assets', nargs='+', default=ASSETS)
    args = parser.parse_args()

    print(f"🚀 Starting Backtest with {args.days} days of {args.timeframe} data")
    
    backtester = BacktestEngine()
    
    for asset in args.assets:
        # Test all strategies
        for strategy in ['main', 'swing', 'scalp']:
            try:
                print(f"\n🔍 Testing {asset} - {strategy.upper()} Strategy")
                
                # Use strategy-specific timeframe from config, or fallback to args
                strategy_timeframe = STRATEGIES[strategy]['timeframe'] if strategy in STRATEGIES else args.timeframe
                
                metrics = backtester.run_backtest(
                    symbol=asset,
                    strategy_type=strategy,
                    days=args.days
                )
                
                print(f"\n📊 Results for {asset} ({strategy.upper()}):")
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        print(f"{k.replace('_', ' ').title()}: {round(v, 2)}")
                    else:
                        print(f"{k.replace('_', ' ').title()}: {v}")

                # Generate report
                backtester.generate_report(asset)
                
            except Exception as e:
                print(f"❌ Error testing {asset} with {strategy} strategy: {str(e)}")
                continue

    print(f"\n✅ All backtests completed! Check the backtest_reports folder")
    print(f"📊 Tested {len(args.assets)} assets over {args.days} days using {args.timeframe} timeframe")

if __name__ == "__main__":
    main()
