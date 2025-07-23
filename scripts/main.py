#!/usr/bin/env python3
import sys
import os
import time
import argparse
import warnings
import atexit
import pandas as pd
from datetime import datetime, timedelta

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
        self.cooldown_until = {} # Cooldown tracker
        self._initialize_journal_file()
        self._load_open_positions()
        print("✅ System ready for trading.")

    def _initialize_journal_file(self):
        """
        Checks if the journal file exists and has the correct header.
        If not, it creates a new one and backs up the old one if necessary.
        """
        new_header = "trade_id,timestamp,asset,direction,entry_price,confidence,pct_change,strategy_name,stop_loss,take_profit,close_price,outcome\n"
        
        if not os.path.exists(self.journal_file):
            with open(self.journal_file, 'w') as f:
                f.write(new_header)
        else:
            with open(self.journal_file, 'r+') as f:
                header = f.readline()
                if header.strip() != new_header.strip():
                    print("⚠️ Detected old journal format. Backing up and creating new file.")
                    f.seek(0)
                    old_content = f.read()
                    backup_filename = f"trading_journal.csv.backup.{int(time.time())}"
                    with open(backup_filename, 'w') as backup_f:
                        backup_f.write(old_content)
                    
                    f.seek(0)
                    f.truncate()
                    f.write(new_header)
                    print(f"✅ Backup saved to {backup_filename}")

    def _load_open_positions(self):
        if os.path.exists(self.positions_file) and os.path.getsize(self.positions_file) > 0:
            self.open_positions = pd.read_csv(self.positions_file)
        else:
            self.open_positions = pd.DataFrame(columns=['trade_id', 'asset', 'direction', 'entry_price', 'strategy_name', 'timestamp', 'stop_loss', 'take_profit'])

    def _save_open_positions(self):
        try:
            with open(self.positions_file, 'w') as f:
                self.open_positions.to_csv(f, index=False)
                f.flush()
                os.fsync(f.fileno())
        except IOError as e:
            print(f"   └ ❌ CRITICAL: Could not save open positions to {self.positions_file}: {e}")

    def analyze_asset(self, asset):
        print("-" * 60)
        print(f"🔍 Analyzing {asset} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.open_positions is None:
            self._load_open_positions()

        asset_position_df = self.open_positions[self.open_positions['asset'] == asset]
        
        if not asset_position_df.empty:
            self._manage_open_position(asset_position_df.iloc[0])
        else:
            # --- COOLDOWN CHECK ---
            if asset in self.cooldown_until and datetime.now() < self.cooldown_until[asset]:
                print(f"   - ❄️ {asset} is in a cooldown period until {self.cooldown_until[asset].strftime('%H:%M:%S')}. Skipping analysis.")
                return
            elif asset in self.cooldown_until:
                print(f"   - ✅ Cooldown for {asset} has ended.")
                del self.cooldown_until[asset]
            
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

        if position['direction'] == 'long' and current_price >= position['take_profit']:
            outcome = 'TAKE_PROFIT'
        elif position['direction'] == 'short' and current_price <= position['take_profit']:
            outcome = 'TAKE_PROFIT'

        if not outcome:
            if position['direction'] == 'long' and current_price <= position['stop_loss']:
                outcome = 'STOP_LOSS'
            elif position['direction'] == 'short' and current_price >= position['stop_loss']:
                outcome = 'STOP_LOSS'

        if not outcome:
            prediction = self.ai.predict(asset, df)
            if prediction:
                self._print_prediction(prediction)
                if (position['direction'] == 'long' and prediction['direction'] == 'short') or \
                   (position['direction'] == 'short' and prediction['direction'] == 'long'):
                    if self.risk_manager.should_execute(prediction, 'main'):
                        outcome = 'REVERSAL_CLOSE'

        if outcome:
            close_price = current_price
            print(f"   - ✅ CLOSING TRADE: {asset} position closed due to {outcome} at ${close_price:,.2f}")
            self._close_trade(position, close_price, outcome)

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

                if strategy_name == 'main': predictions['main'] = self.ai.predict(asset, df_strategy)
                elif strategy_name == 'swing': predictions['swing'] = self.ai.predict_swing(asset, df_strategy)
                elif strategy_name == 'btc-swing': predictions['btc-swing'] = self.ai.predict(asset, df_strategy, strategy_name='btc-swing')
                elif strategy_name == 'scalp': predictions['scalp'] = self.ai.predict_scalp(asset, df_strategy)

            for strategy_name, pred in predictions.items():
                if pred:
                    print(f"\n   --- Strategy: {strategy_name.upper()} ---")
                    self._print_prediction(pred)
                    if self.risk_manager.should_execute(pred, strategy_name):
                        print(f"   └ ✅ Signal passed initial risk checks.")

                        # --- MTF CONFIRMATION FILTER ---
                        if strategy_name == 'main':
                            print("   └ 🧠 Applying Multi-Timeframe (MTF) Confirmation Filter...")
                            higher_timeframe = STRATEGIES['swing']['timeframe'] # 4h
                            df_higher = self.data.get_data(asset, higher_timeframe)

                            if df_higher is None or df_higher.empty or 'ema_50' not in df_higher.columns:
                                print(f"   └ ❌ MTF Rejection: Could not get {higher_timeframe} data for confirmation. Skipping trade for safety.")
                                continue

                            last_price_higher = df_higher['close'].iloc[-1]
                            ema_50_higher = df_higher['ema_50'].iloc[-1]
                            
                            is_uptrend = last_price_higher > ema_50_higher
                            is_downtrend = last_price_higher < ema_50_higher
                            
                            signal_direction = pred['direction']

                            if (signal_direction == 'long' and is_uptrend) or \
                               (signal_direction == 'short' and is_downtrend):
                                print(f"   └ ✅ MTF Confirmation: 1h signal aligns with 4h trend ({'UP' if is_uptrend else 'DOWN'}). Opening trade.")
                                self._open_trade(pred, strategy_name, df_strategy)
                                return # Trade opened, exit loop for this asset
                            else:
                                print(f"   └ ❌ MTF Rejection: 1h signal '{signal_direction}' conflicts with 4h trend ({'UP' if is_uptrend else 'DOWN'}).")

                        # For other strategies, open trade directly without MTF check
                        else:
                            print(f"   └ ✅ NEW TRADE SIGNAL! ({strategy_name.upper()} does not require MTF). Opening trade.")
                            self._open_trade(pred, strategy_name, df_strategy)
                            return # Trade opened, exit loop for this asset
                    else:
                        print(f"   └ ⚠️ Signal rejected by risk manager or strategy rules.")
        except Exception as e:
            print(f"💥 A critical error occurred while analyzing {asset}: {e}")
            import traceback
            traceback.print_exc()

    def _open_trade(self, prediction, strategy_name, df):
        trade_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{prediction['asset']}"
        rules = STRATEGIES[strategy_name]
        levels = self.risk_manager.calculate_levels(prediction, df)

        # --- ATOMIC OPERATION: Step 1. Update and save state FIRST ---
        position_entry = {
            'trade_id': trade_id, 'asset': prediction['asset'], 'direction': prediction['direction'],
            'entry_price': prediction['current_price'], 'strategy_name': strategy_name,
            'timestamp': prediction['timestamp'], 'stop_loss': levels.get('stop_loss', 0),
            'take_profit': levels.get('take_profit', 0)
        }
        self.open_positions = pd.concat([self.open_positions, pd.DataFrame([position_entry])], ignore_index=True)
        self._save_open_positions()
        print(f"   └ 📖 Position added to {self.positions_file}")

        # --- ATOMIC OPERATION: Step 2. Log to journal SECOND ---
        journal_entry = (
            f"{trade_id},{prediction['timestamp'].isoformat()},{prediction['asset']},"
            f"{prediction['direction']},{prediction['current_price']:.4f},{prediction['confidence']:.4f},"
            f"{prediction['pct_change']:.4f},{strategy_name},{levels.get('stop_loss', 0):.4f},"
            f"{levels.get('take_profit', 0):.4f},,\n"
        )
        try:
            with open(self.journal_file, "a") as f:
                f.write(journal_entry)
                f.flush()
                os.fsync(f.fileno())
            print("   └ 📝 Trade logged to trading_journal.csv")
        except IOError as e:
            print(f"   └ ❌ Error: Could not log trade to journal: {e}")

    def _close_trade(self, position, close_price, outcome):
        asset = position['asset']
        strategy_name = position['strategy_name']
        trade_id = position['trade_id']

        # --- ATOMIC OPERATION: Step 1. Update and save state FIRST ---
        self.open_positions = self.open_positions[self.open_positions['trade_id'] != trade_id].copy()
        self._save_open_positions()
        print(f"   └ 📖 Position removed from {self.positions_file}")

        # --- INITIATE COOLDOWN ---
        try:
            timeframe = STRATEGIES[strategy_name]['timeframe']
            if 'h' in timeframe:
                candle_period = timedelta(hours=int(timeframe.replace('h', '')))
            elif 'm' in timeframe:
                candle_period = timedelta(minutes=int(timeframe.replace('m', '')))
            else:
                candle_period = timedelta(hours=1) # Default fallback
            
            cooldown_duration = candle_period * 3
            cooldown_end = datetime.now() + cooldown_duration
            self.cooldown_until[asset] = cooldown_end
            print(f"   └ ❄️ Cooldown initiated for {asset} for {cooldown_duration}. No new trades until {cooldown_end.strftime('%Y-%m-%d %H:%M:%S')}.")
        except Exception as e:
            print(f"   └ ⚠️ Could not set cooldown for {asset}: {e}")

        # --- ATOMIC OPERATION: Step 2. Update journal SECOND ---
        try:
            # This read/modify/write operation is not truly atomic, but we add flushing for resilience.
            journal_df = pd.read_csv(self.journal_file)
            trade_index = journal_df[journal_df['trade_id'] == trade_id].index
            if not trade_index.empty:
                journal_df.loc[trade_index, 'close_price'] = close_price
                journal_df.loc[trade_index, 'outcome'] = outcome
                
                # Use a file object to ensure flush and fsync
                with open(self.journal_file, 'w') as f:
                    journal_df.to_csv(f, index=False)
                    f.flush()
                    os.fsync(f.fileno())
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
