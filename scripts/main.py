#!/usr/bin/env python3
import sys
import os
# --- Force load environment variables from .env file for Replit ---
from dotenv import load_dotenv
load_dotenv()
# --------------------------------------------------------------------
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
from scripts.core.database_manager import DatabaseManager
from scripts.core.csv_logger import CsvLogger
from scripts.core.regime_filter import MarketRegimeFilter # <-- Import the new filter
from scripts.config import ASSETS, STRATEGIES, REGIME_CONFIG # <-- Import the new config

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
        self.db = DatabaseManager() # Initialize the database manager
        self.csv_logger = CsvLogger() # Initialize the CSV logger
        self.update_interval = 300
        self.run_once = run_once
        self.cooldown_until = {} # Cooldown tracker
        
        # Create a separate regime filter instance for each asset to maintain state
        self.regime_filters = {asset: MarketRegimeFilter() for asset in ASSETS}
        
        # Load open positions from the database, which is now the single source of truth
        self._load_open_positions()
        print("✅ System ready for trading.")

    def _load_open_positions(self):
        """
        Loads all currently open positions from the database.
        This is the single source of truth for the system's state.
        """
        print("   - Loading open positions from database...")
        self.open_positions = self.db.load_open_positions()
        print(f"   - ✅ Found {len(self.open_positions)} open position(s).")

    def analyze_asset(self, asset):
        print("-" * 60)
        print(f"🔍 Analyzing {asset} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # The self.open_positions DataFrame is now managed directly by the open/close methods
        asset_position_df = self.open_positions[self.open_positions['asset'] == asset]
        
        if not asset_position_df.empty:
            self._manage_open_position(asset_position_df.iloc[0].to_dict())
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
        
        main_timeframe = STRATEGIES[asset]['main']['timeframe']
        df = self.data.get_data(asset, main_timeframe)
        if df is None or df.empty:
            print(f"   - ❌ Skipping position management for {asset} due to data failure.")
            return

        current_price = df['close'].iloc[-1]
        outcome = None
        close_price = None

        # --- 1. TRAILING STOP LOSS LOGIC ---
        new_stop_loss = self.risk_manager.update_trailing_stop(
            entry_price=position['entry_price'],
            current_price=current_price,
            direction=position['direction'],
            current_stop=position['stop_loss']
        )
        
        if new_stop_loss is not None and new_stop_loss != position['stop_loss']:
            print(f"   - 📈 TRAILING STOP UPDATE for {asset}: New Stop Loss at ${new_stop_loss:,.4f}")
            # Persist the change to the database and CSV
            self.db.update_trade_stop_loss(position['trade_id'], new_stop_loss)
            self.csv_logger.update_trade_stop_loss(position['trade_id'], new_stop_loss)
            # Update the in-memory representation
            position['stop_loss'] = new_stop_loss
            self.open_positions.loc[self.open_positions['trade_id'] == position['trade_id'], 'stop_loss'] = new_stop_loss


        # --- 2. CHECK FOR TP/SL HIT ---
        if position['direction'] == 'long' and current_price >= position['take_profit']:
            outcome = 'TAKE_PROFIT'
        elif position['direction'] == 'short' and current_price <= position['take_profit']:
            outcome = 'TAKE_PROFIT'

        if not outcome:
            if position['direction'] == 'long' and current_price <= position['stop_loss']:
                outcome = 'STOP_LOSS'
            elif position['direction'] == 'short' and current_price >= position['stop_loss']:
                outcome = 'STOP_LOSS'

        # --- 3. CHECK FOR REVERSAL SIGNAL ---
        if not outcome:
            prediction = self.ai.predict(asset, df)
            if prediction:
                self._print_prediction(prediction)
                if (position['direction'] == 'long' and prediction['direction'] == 'short') or \
                   (position['direction'] == 'short' and prediction['direction'] == 'long'):
                    if self.risk_manager.should_execute(prediction, asset, 'main'):
                        outcome = 'REVERSAL_CLOSE'

        if outcome:
            close_price = current_price
            print(f"   - ✅ CLOSING TRADE: {asset} position closed due to {outcome} at ${close_price:,.2f}")
            self._close_trade(position, close_price, outcome)

    def _look_for_new_trade(self, asset):
        try:
            # Get the config for the 'main' strategy for the specific asset
            asset_main_strategy = STRATEGIES[asset]['main']
            main_timeframe = asset_main_strategy['timeframe']
            df = self.data.get_data(asset, main_timeframe)

            if df is None:
                print(f"   - ❌ Skipping analysis for {asset} due to data failure.")
                return

            self.data_monitor.log_result(self.data.last_used_source, not df.empty)

            if len(df) < asset_main_strategy['sequence_length']:
                print(f"   - ⚠️ Insufficient data for {asset} on {main_timeframe} timeframe. Skipping analysis.")
                return

            print("   - Getting AI predictions for all strategies...")
            predictions = {}
            strategies_to_run = ['main']
            if asset == 'BTC-USD':
                strategies_to_run.append('btc-swing')
            else:
                strategies_to_run.append('swing')

            for strategy_name in strategies_to_run:
                # Ensure the asset has this strategy defined
                if strategy_name not in STRATEGIES.get(asset, {}): continue
                
                strategy_config = STRATEGIES[asset][strategy_name]
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

                    # =================================================
                    # == NEW: MARKET REGIME & ENTROPY MASTER FILTER  ==
                    # =================================================
                    regime_filter = self.regime_filters[asset]
                    regime = regime_filter.get_regime(
                        df=df_strategy,
                        adx_threshold=REGIME_CONFIG['adx_trending_threshold'],
                        entropy_threshold=REGIME_CONFIG['entropy_chaotic_threshold'],
                        entropy_window=REGIME_CONFIG['entropy_window'],
                        smoothing_alpha=REGIME_CONFIG['entropy_smoothing_alpha']
                    )
                    print(f"   └ 📊 Market Regime Detected: {regime.upper()}")

                    # For now, we assume all our strategies are trend-following.
                    # This is the master gatekeeper.
                    if regime == 'Chaotic':
                        print("   └ ❌ Regime Rejection: Market is too random and unpredictable. No trades allowed.")
                        continue # Skip to the next prediction
                    
                    if regime != 'Trending':
                        print(f"   └ ❌ Regime Rejection: The '{strategy_name}' strategy requires a trending market. No trades allowed.")
                        continue # Skip to the next prediction
                    # =================================================

                    if self.risk_manager.should_execute(pred, asset, strategy_name):
                        print(f"   └ ✅ Signal passed initial risk checks.")

                        # --- MTF CONFIRMATION FILTER ---
                        if strategy_name == 'main':
                            print("   └ 🧠 Applying Multi-Timeframe (MTF) Confirmation Filter...")
                            # Determine the correct swing strategy name for the asset
                            swing_strategy_name = 'btc-swing' if asset == 'BTC-USD' else 'swing'
                            higher_timeframe = STRATEGIES[asset][swing_strategy_name]['timeframe']
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
        asset = prediction['asset']
        trade_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{asset}"
        rules = STRATEGIES[asset][strategy_name]
        levels = self.risk_manager.calculate_levels(prediction, df)

        position_entry = {
            'trade_id': trade_id, 'asset': asset, 'direction': prediction['direction'],
            'entry_price': prediction['current_price'], 'strategy_name': strategy_name,
            'timestamp': prediction['timestamp'], 'stop_loss': levels.get('stop_loss', 0),
            'take_profit': levels.get('take_profit', 0),
            'confidence': prediction.get('confidence'),
            'pct_change': prediction.get('pct_change')
        }

        print("   └ 💾 Writing to database...")
        if self.db.add_trade(position_entry):
            print("   └ ✅ Trade successfully saved to database.")
            # Update the in-memory DataFrame to reflect the new state
            self.open_positions = pd.concat([self.open_positions, pd.DataFrame([position_entry])], ignore_index=True)
            # Also log the trade to the CSV files for easy viewing
            self.csv_logger.add_trade(position_entry)
        else:
            print("   └ ❌ CRITICAL: Failed to write trade to database. Aborting trade to prevent inconsistency.")

    def _close_trade(self, position, close_price, outcome):
        asset = position['asset']
        strategy_name = position['strategy_name']
        trade_id = position['trade_id']

        print(f"   └ 💾 Closing trade {trade_id} in database...")
        if self.db.close_trade(trade_id, close_price, outcome):
            print(f"   └ ✅ Trade successfully closed in database.")
            # Update the in-memory DataFrame to reflect the new state
            self.open_positions = self.open_positions[self.open_positions['trade_id'] != trade_id].copy()
            # Also update the CSV files to reflect the closure
            self.csv_logger.close_trade(trade_id, close_price, outcome)
        else:
            print(f"   └ ❌ CRITICAL: Failed to close trade in database. Manual check required.")

        # --- INITIATE COOLDOWN ---
        try:
            timeframe = STRATEGIES[asset][strategy_name]['timeframe']
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
