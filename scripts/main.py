#!/usr/bin/env python3
import sys
import os
import time
import argparse
import warnings
import atexit
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

from scripts.core.data_engine import DataMaster
from scripts.core.analysis_engine import AIAnalyst
from scripts.config import ASSETS, STRATEGIES
from scripts.core.data_monitor import DataHealthMonitor
from scripts.core.risk_engine import RiskManager

def goodbye():
    sys.stdout.flush()

atexit.register(goodbye)

class ZargulTrader:
    def __init__(self, train_all=False, run_once=False):
        print("\n🤖 Initializing Trading System...")
        self.data = DataMaster()
        self.ai = AIAnalyst(train_all)
        self.risk_manager = RiskManager()
        self.data_monitor = DataHealthMonitor()
        self.update_interval = 300
        self.run_once = run_once
        self.positions_file = 'open_positions.csv'
        self.journal_file = 'trading_journal.csv'
        self._load_open_positions()
        print("✅ System ready for trading.")

    def _load_open_positions(self):
        if os.path.exists(self.positions_file) and os.path.getsize(self.positions_file) > 0:
            self.open_positions = pd.read_csv(self.positions_file)
        else:
            self.open_positions = pd.DataFrame(columns=['trade_id', 'asset', 'direction', 'entry_price', 'strategy_name', 'timestamp', 'stop_loss', 'take_profit'])

    def _save_open_positions(self):
        self.open_positions.to_csv(self.positions_file, index=False)

    def analyze_asset(self, asset):
        print("-" * 60)
        print(f"🔍 Analyzing {asset} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Ensure open_positions is not None and is a DataFrame
        if self.open_positions is None:
            self._load_open_positions()

        asset_position_df = self.open_positions[self.open_positions['asset'] == asset]
        
        if not asset_position_df.empty:
            self._manage_open_position(asset_position_df.iloc[0])
        else:
            self._look_for_new_trade(asset)

    def _manage_open_position(self, position):
        asset = position['asset']
        print(f"   - Managing open {position['direction'].upper()} position for {asset}.")
        
        main_timeframe = STRATEGIES['main']['timeframe']
        df = self.data.get_data(asset, main_timeframe)
        if df is None or df.empty:
            print(f"   - ❌ Skipping position management for {asset} due to data failure.")
            return

        current_price = df['close'].iloc[-1]
        outcome = None
        close_price = None

        # 1. Check for Take Profit
        if position['direction'] == 'long' and current_price >= position['take_profit']:
            outcome = 'TAKE_PROFIT'
        elif position['direction'] == 'short' and current_price <= position['take_profit']:
            outcome = 'TAKE_PROFIT'

        # 2. Check for Stop Loss
        if not outcome:
            if position['direction'] == 'long' and current_price <= position['stop_loss']:
                outcome = 'STOP_LOSS'
            elif position['direction'] == 'short' and current_price >= position['stop_loss']:
                outcome = 'STOP_LOSS'

        # 3. Check for Reversal Signal
        if not outcome:
            prediction = self.ai.predict(asset, df) # Using main strategy for exit signal
            if prediction:
                self._print_prediction(prediction)
                if (position['direction'] == 'long' and prediction['direction'] == 'short') or \
                   (position['direction'] == 'short' and prediction['direction'] == 'long'):
                    if self.risk_manager.should_execute(prediction, 'main'): # Check if exit signal is valid
                        outcome = 'REVERSAL_CLOSE'

        if outcome:
            close_price = current_price
            print(f"   - ✅ CLOSING TRADE: {asset} position closed due to {outcome} at ${close_price:,.2f}")
            self._close_trade(position['trade_id'], close_price, outcome)

    def _look_for_new_trade(self, asset):
        try:
            main_timeframe = STRATEGIES['main']['timeframe']
            df = self.data.get_data(asset, main_timeframe)

            if df is None:
                print(f"   - ❌ Skipping analysis for {asset} due to data failure.")
                return

            self.data_monitor.log_result(self.data.last_used_source, not df.empty)

            if len(df) < STRATEGIES['main']['sequence_length']:
                print(f"   - ⚠️ Insufficient data for {asset} on {main_timeframe} timeframe. Skipping analysis.")
                return

            print("   - Getting AI predictions for all strategies...")
            predictions = {}
            strategies_to_run = ['main', 'scalp']
            if asset == 'BTC-USD':
                strategies_to_run.append('btc-swing')
            else:
                strategies_to_run.append('swing')

            for strategy_name in strategies_to_run:
                if strategy_name not in STRATEGIES: continue
                strategy_config = STRATEGIES[strategy_name]
                timeframe = strategy_config['timeframe']
                df_strategy = self.data.get_data(asset, timeframe)

                if df_strategy is None or len(df_strategy) < strategy_config['sequence_length']:
                    print(f"   - ⚠️ Insufficient data for {asset} on {timeframe} for {strategy_name}. Skipping.")
                    continue

                # Generate prediction based on strategy
                if strategy_name == 'main': predictions['main'] = self.ai.predict(asset, df_strategy)
                elif strategy_name == 'swing': predictions['swing'] = self.ai.predict_swing(asset, df_strategy)
                elif strategy_name == 'btc-swing': predictions['btc-swing'] = self.ai.predict(asset, df_strategy, strategy_name='btc-swing')
                elif strategy_name == 'scalp': predictions['scalp'] = self.ai.predict_scalp(asset, df_strategy)

            for strategy_name, pred in predictions.items():
                if pred:
                    print(f"\n   --- Strategy: {strategy_name.upper()} ---")
                    self._print_prediction(pred)
                    if self.risk_manager.should_execute(pred, strategy_name):
                        print(f"   └ ✅ NEW TRADE SIGNAL! {pred['direction'].upper()} signal is valid.")
                        self._open_trade(pred, strategy_name)
                        return # Stop after opening one trade per asset
                    else:
                        print(f"   └ ⚠️ Signal rejected by risk manager or strategy rules.")
        except Exception as e:
            print(f"💥 A critical error occurred while analyzing {asset}: {e}")
            import traceback
            traceback.print_exc()

    def _open_trade(self, prediction, strategy_name):
        trade_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{prediction['asset']}"
        rules = STRATEGIES[strategy_name]
        levels = self.risk_manager.calculate_levels(prediction['current_price'], prediction['direction'], rules)
        
        # Log to journal
        journal_entry = f"{trade_id},{prediction['timestamp'].isoformat()},{prediction['asset']},{prediction['direction']},{prediction['confidence']:.4f},{prediction['pct_change']:.4f},{strategy_name},{levels.get('stop_loss', 0):.4f},{levels.get('take_profit', 0):.4f},,\n"
        try:
            with open(self.journal_file, "a") as f:
                f.write(journal_entry)
            print("   └ 📝 Trade logged to trading_journal.csv")
        except IOError as e:
            print(f"   └ ❌ Error: Could not log trade to journal: {e}")

        # Add to open positions
        position_entry = {
            'trade_id': trade_id, 'asset': prediction['asset'], 'direction': prediction['direction'],
            'entry_price': prediction['current_price'], 'strategy_name': strategy_name,
            'timestamp': prediction['timestamp'], 'stop_loss': levels.get('stop_loss', 0),
            'take_profit': levels.get('take_profit', 0)
        }
        
        # Use concat instead of append
        self.open_positions = pd.concat([self.open_positions, pd.DataFrame([position_entry])], ignore_index=True)
        self._save_open_positions()
        print(f"   └ 📖 Position added to {self.positions_file}")

    def _close_trade(self, trade_id, close_price, outcome):
        # Remove from open positions
        self.open_positions = self.open_positions[self.open_positions['trade_id'] != trade_id].copy()
        self._save_open_positions()
        print(f"   └ 📖 Position removed from {self.positions_file}")

        # Update journal
        try:
            journal_df = pd.read_csv(self.journal_file)
            trade_index = journal_df[journal_df['trade_id'] == trade_id].index
            if not trade_index.empty:
                journal_df.loc[trade_index, 'close_price'] = close_price
                journal_df.loc[trade_index, 'outcome'] = outcome
                journal_df.to_csv(self.journal_file, index=False)
                print(f"   └ 📝 Journal updated for trade {trade_id}")
        except Exception as e:
            print(f"   └ ❌ Error updating journal: {e}")

    def _print_prediction(self, prediction):
        print(f"   │ Direction: {prediction['direction'].upper():<5} | Confidence: {prediction['confidence']*100: >3.0f}% | Move: {prediction['pct_change']: >5.2f}% | Current Price: ${prediction['current_price']:,.2f}")

    def run(self):
        print("\n🚀 Zargul Trader 2.0 - Live Operations Starting")
        print("=" * 60)
        while True:
            # Reload positions at the start of each cycle for resilience
            self._load_open_positions()
            for asset in ASSETS:
                self.analyze_asset(asset)
            if self.run_once:
                print("\n✅ Run once complete.")
                break
            print("\n" + "=" * 60)
            print(f"⏳ Next analysis cycle in {self.update_interval / 60:.0f} minutes...")
            time.sleep(self.update_interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zargul Trader 2.0")
    parser.add_argument('--train-all', action='store_true', help='Force retraining of all AI models.')
    parser.add_argument('--run-once', action='store_true', help='Run the analysis loop only once and then exit.')
    args = parser.parse_args()
    try:
        trader = ZargulTrader(train_all=args.train_all, run_once=args.run_once)
        trader.run()
    except Exception as e:
        print(f"\n💥 A fatal error occurred: {e}")
        sys.exit(1)
