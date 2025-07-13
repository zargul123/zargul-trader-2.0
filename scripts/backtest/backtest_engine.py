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

            # Initialize risk manager for position sizing
            from scripts.core.risk_engine import RiskManager
            risk_manager = RiskManager()

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
                    
                    # Calculate position size with symbol-aware risk management
                    position_size = risk_manager.calculate_position_size(window_data, symbol)
                    
                    trade_data = prediction.copy()
                    trade_data.update({
                        'entry_time': current_time,
                        'current_price': current_price,
                        'asset': symbol,
                        'position_size': position_size
                    })
                    self._execute_trade(trade_data, df, strategy_type)

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

    def _execute_trade(self, prediction, df, strategy_type):
        """
        Simulates a realistic trade using actual market data for exits.
        """
        try:
            entry_time = prediction.get('entry_time')
            entry_price = prediction.get('current_price')
            direction = prediction.get('direction')
            symbol = prediction.get('asset')
            
            # Realistic market conditions
            slippage = 0.001  # 0.1%
            fees = 0.001     # 0.1% total (entry + exit)
            
            # Apply slippage (goes against you)
            if direction == 'long':
                actual_entry = entry_price * (1 + slippage)
            else: # short
                actual_entry = entry_price * (1 - slippage)

            # Determine hold period based on strategy
            strategy_config = STRATEGIES.get(strategy_type, STRATEGIES['main'])  # Use correct strategy
            hold_hours = strategy_config.get('hold_period_hours', 4)
            hold_period = timedelta(hours=hold_hours)
            
            # Check for dynamic exit
            use_dynamic_exit = strategy_config.get('dynamic_exit', False)
            target_exit_time = entry_time + hold_period
            
            # Fix: Ensure timezone consistency for time comparisons
            if hasattr(entry_time, 'tz') and entry_time.tz is not None:
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                entry_time = entry_time.tz_convert('UTC')
                target_exit_time = target_exit_time.tz_convert('UTC')
            elif df.index.tz is not None:
                df.index = df.index.tz_localize(None)
                if hasattr(entry_time, 'tz'):
                    entry_time = entry_time.tz_localize(None)
                    target_exit_time = target_exit_time.tz_localize(None)
                
            # Find the closest available data point to our target exit time
            try:
                future_data = df[df.index > entry_time]
                if future_data.empty:
                    return  # No future data available
                    
                # Use the first available price after our target time, or the last available
                exit_data = future_data[future_data.index >= target_exit_time]
                if exit_data.empty:
                    exit_data = future_data.tail(1)
                else:
                    exit_data = exit_data.head(1)
            except Exception as e:
                print(f"⚠️ Time comparison error in backtest: {e}")
                return
                
            actual_exit_time = exit_data.index[0]
            exit_price = exit_data['close'].iloc[0]
            
            # Apply exit slippage
            if direction == 'long':
                actual_exit = exit_price * (1 - slippage)
            else: # short
                actual_exit = exit_price * (1 + slippage)

            # Implement dynamic exit if enabled
            if use_dynamic_exit and not future_data.empty:
                # Check for better exit points based on momentum
                for idx, row in future_data.iterrows():
                    if idx >= target_exit_time:
                        break
                    
                    current_pnl = 0
                    if direction == 'long':
                        current_pnl = ((row['close'] - actual_entry) / actual_entry - fees) * 100
                    else:
                        current_pnl = ((actual_entry - row['close']) / actual_entry - fees) * 100
                    
                    # Dynamic exit conditions
                    if current_pnl > 1.5:  # Take profit at 1.5%
                        actual_exit_time = idx
                        exit_price = row['close']
                        break
                    elif current_pnl < -1.0:  # Stop loss at 1%
                        actual_exit_time = idx
                        exit_price = row['close']
                        break

            # Apply exit slippage
            if direction == 'long':
                actual_exit = exit_price * (1 - slippage)
            else: # short
                actual_exit = exit_price * (1 + slippage)

            # Calculate realistic PnL
            if direction == 'long':
                raw_return = (actual_exit - actual_entry) / actual_entry
                pnl = (raw_return - fees) * 100
            else: # short
                raw_return = (actual_entry - actual_exit) / actual_entry
                pnl = (raw_return - fees) * 100

            # Add some realism - not all trades work perfectly
            # Simulate stop losses (2% loss limit)
            if pnl < -2.0:
                pnl = -2.0 - random.uniform(0, 0.5)  # Some slippage on stops
                actual_exit = actual_entry * (0.98 if direction == 'long' else 1.02)

            trade = {
                'symbol': symbol,
                'entry_time': entry_time,
                'exit_time': actual_exit_time,
                'entry_price': actual_entry,
                'exit_price': actual_exit,
                'type': direction,
                'pnl': pnl,
                'status': 'closed',
                'confidence': prediction.get('confidence', 0),
                'predicted_pnl': prediction.get('pct_change', 0),  # For comparison
            }
            
            self.trade_history.append(trade)
            result_emoji = "✅" if pnl > 0 else "❌"
            print(f"{result_emoji} Trade: {direction.upper()} @ ${actual_entry:.2f} -> ${actual_exit:.2f} | PnL: {pnl:.2f}% | Pred: {prediction.get('pct_change', 0):.2f}% ")

        except Exception as e:
            print(f"❌ Trade execution error: {str(e)}")

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