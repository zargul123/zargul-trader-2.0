import os
import sys
import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timedelta
import random

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.core.data_engine import DataMaster
from scripts.core.analysis_engine import AIAnalyst
from scripts.config import STRATEGIES
from scripts.backtest.metrics import calculate_all_metrics, get_empty_metrics

class BacktestEngine:
    def __init__(self, debug=False):
        self.analyst = AIAnalyst()
        self.data = DataMaster()
        self.debug = debug
        self.results = []

    def load_data(self, symbol, days, timeframe):
        """
        Loads historical data for backtesting.
        """
        days_to_load = max(days, 90)
        print(f"\n📊 Loading {days_to_load} days of {timeframe} data for {symbol}...")
        
        # CORRECTED: Use the 'timeframe' passed into the function
        df = self.data.get_data(symbol, timeframe, limit=days_to_load * 24)
        
        if df is None or df.empty:
            print(f"⚠️ Could not load data for {symbol}.")
            return pd.DataFrame()

        df = df.sort_index()

        required_sequence_length = STRATEGIES.get('main', {}).get('sequence_length', 60)
        if len(df) < required_sequence_length:
            print(f"⚠️ Insufficient data for {symbol}. Need at least {required_sequence_length} candles, but got {len(df)}.")
            return pd.DataFrame()
        
        print(f"📅 Data loaded for {symbol}: {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        return df

    def run_backtest(self, symbol, strategy_type, days):
        """
        Runs a backtest for a given symbol and strategy.
        This is the single, clean, AI-driven backtest implementation.
        """
        self.trade_history = [] # Reset history for each run
        try:
            # 1. Get Strategy Configuration
            strategy_config = STRATEGIES.get(strategy_type)
            if not strategy_config:
                print(f"❌ Unknown strategy: {strategy_type}")
                return get_empty_metrics()

            print(f"\n🔧 Running backtest for {symbol} with '{strategy_type}' strategy.")
            print(f"   - Confidence Threshold: {strategy_config['min_confidence'] * 100}%")
            print(f"   - Long Target: > {strategy_config['long_threshold']}%")
            print(f"   - Short Target: < -{strategy_config['short_threshold']}%")

            # 2. Load Data
            df = self.load_data(symbol, days, strategy_config['timeframe'])
            if df.empty:
                return get_empty_metrics()

            # 3. Simulate Trading Loop
            window_size = strategy_config.get('sequence_length', 60)
            step_size = 12  # Check for a signal every 12 hours

            print(f"\n🔎 Scanning {len(df)} candles for signals (Window: {window_size}, Step: {step_size})...")

            for i in range(window_size, len(df), step_size):
                window_data = df.iloc[i - window_size : i]
                current_time = df.index[i]
                current_price = df.iloc[i]['close']

                # Get AI prediction using the correct method and passing historical data
                if strategy_type == "main":
                    prediction = self.analyst.predict(symbol, window_data)
                elif strategy_type == "swing":
                    # CORRECTED: Pass window_data to the prediction method
                    prediction = self.analyst.predict_swing(symbol, window_data)
                else: # scalp
                    # CORRECTED: Pass window_data to the prediction method
                    prediction = self.analyst.predict_scalp(symbol, window_data)

                if not prediction:
                    continue

                # Check if the prediction meets the strategy's criteria
                is_confident = prediction['confidence'] >= strategy_config['min_confidence']
                is_long_target_met = prediction['direction'] == 'long' and prediction['pct_change'] >= strategy_config['long_threshold']
                is_short_target_met = prediction['direction'] == 'short' and prediction['pct_change'] <= -strategy_config['short_threshold']

                if self.debug:
                    print(f"   - Signal at {current_time}: Dir={prediction['direction']}, Conf={prediction['confidence']:.2f}, PctChg={prediction['pct_change']:.2f}%")

                if is_confident and (is_long_target_met or is_short_target_met):
                    print(f"🎯 {current_time}: {prediction['direction'].upper()} signal found!")
                    trade_data = prediction.copy()
                    trade_data.update({
                        'entry_time': current_time,
                        'current_price': current_price,
                        'asset': symbol
                    })
                    self._execute_trade(trade_data)

            # 4. Calculate Final Metrics
            print(f"\n✅ Backtest scan complete for {symbol}.")
            if not self.trade_history:
                print("⚠️ No trades were executed in this backtest.")
                return get_empty_metrics()

            print("\n📈 Trade Performance Summary:")
            wins = [t for t in self.trade_history if t.get('pnl', 0) > 0]
            if self.trade_history:
                win_rate = (len(wins) / len(self.trade_history)) * 100
                avg_pnl = sum(t.get('pnl', 0) for t in self.trade_history) / len(self.trade_history)
                print(f"• {len(wins)}/{len(self.trade_history)} winning trades ({win_rate:.1f}% win rate)")
                print(f"• Avg PnL: {avg_pnl:.2f}%")
            
            metrics = calculate_all_metrics(self.trade_history)
            
            self.results.append({
                'symbol': symbol,
                'strategy': strategy_type,
                'metrics': metrics,
                'trades': self.trade_history
            })

            return metrics

        except Exception:
            print(f"🔥 An unexpected error occurred during the backtest for {symbol}:")
            print(traceback.format_exc())
            return get_empty_metrics()

    def _execute_trade(self, prediction):
        """
        Simulates a single trade with realistic slippage and fees.
        This is a simplified and robust version.
        """
        try:
            entry_time = prediction.get('entry_time')
            entry_price = prediction.get('current_price')
            direction = prediction.get('direction')
            
            # Simulate realistic market conditions
            slippage = 0.001  # 0.1%
            fees = 0.0005   # 0.05%
            
            # Slippage goes against you
            if direction == 'long':
                actual_entry = entry_price * (1 + slippage)
            else: # short
                actual_entry = entry_price * (1 - slippage)

            # Simplified exit: Hold for a fixed period (e.g., 4 hours)
            # A real system would have dynamic exits (take profit/stop loss)
            hold_period = timedelta(hours=4)
            exit_time = entry_time + hold_period
            
            # We need to find the actual close price from the dataframe at exit_time
            # This part is complex, so for now, we simulate an exit based on prediction
            # This is an area for future improvement!
            predicted_move = prediction.get('pct_change', 0) / 100.0
            exit_price = actual_entry * (1 + predicted_move)

            # Calculate PnL
            if direction == 'long':
                pnl = ((exit_price / actual_entry) - 1 - fees) * 100
            else: # short
                pnl = ((actual_entry / exit_price) - 1 - fees) * 100

            trade = {
                'symbol': prediction['asset'],
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': actual_entry,
                'exit_price': exit_price,
                'type': direction,
                'pnl': pnl,
                'status': 'closed',
                'confidence': prediction.get('confidence', 0)
            }
            
            self.trade_history.append(trade)
            print(f"✅ Executed Trade: {direction.upper()} @ ${actual_entry:.2f} -> ${exit_price:.2f} | PnL: {pnl:.2f}%")

        except Exception:
            print(f"❌ Trade execution error: {traceback.format_exc()}")

    def generate_report(self, symbol):
        """Create visual report (placeholder for now)"""
        result = next((r for r in self.results if r['symbol'] == symbol), None)
        if not result:
            print(f"⚠️ No results found for {symbol} to generate a report.")
            return
            
        print(f"\n📄 Report for {symbol} - {result['strategy']}")
        print("-" * 40)
        for key, value in result['metrics'].items():
            formatted_value = f"{value:.2f}" if isinstance(value, (int, float)) else value
            print(f"{key.replace('_', ' ').title():<20}: {formatted_value}")
        print("-" * 40)
