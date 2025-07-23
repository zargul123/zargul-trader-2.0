import os
import sys
import pandas as pd
import numpy as np
import traceback
from datetime import datetime

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.core.data_engine import DataMaster
from scripts.core.analysis_engine import AIAnalyst
from scripts.config import STRATEGIES
from scripts.backtest.metrics import calculate_all_metrics, get_empty_metrics
from scripts.core.risk_engine import RiskManager

class BacktestEngine:
    def __init__(self, debug=False):
        self.analyst = AIAnalyst()
        self.data = DataMaster()
        self.risk_manager = RiskManager()
        self.debug = debug
        self.results = []

    def load_data(self, symbol, days, timeframe):
        """
        Loads historical data for backtesting.
        """
        print(f"\n📊 Loading {days} days of {timeframe} data for {symbol}...")
        
        # Calculate records needed based on timeframe
        if 'h' in timeframe:
            records_per_day = 24 / int(timeframe.replace('h', ''))
        elif 'm' in timeframe:
            records_per_day = (24 * 60) / int(timeframe.replace('m', ''))
        else:
            records_per_day = 1 # Default for daily
            
        limit = int(days * records_per_day)
        
        df = self.data.get_data(symbol, timeframe, limit=limit)
        
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
        Runs a backtest for a given symbol and strategy, simulating the live trade management logic.
        """
        self.trade_history = []
        try:
            strategy_config = STRATEGIES.get(strategy_type)
            if not strategy_config:
                print(f"❌ Unknown strategy: {strategy_type}")
                return get_empty_metrics()

            print(f"\n🔧 Running backtest for {symbol} with '{strategy_type}' strategy.")
            df = self.load_data(symbol, days, strategy_config['timeframe'])
            if df.empty:
                return get_empty_metrics()

            # --- Backtesting State ---
            open_position = None # This is our "virtual notebook"
            # -------------------------

            sequence_length = strategy_config.get('sequence_length', 60)
            print(f"\n🔎 Scanning {len(df)} candles for signals...")

            for i in range(sequence_length, len(df)):
                current_candle = df.iloc[i]
                current_price = current_candle['close']
                current_time = df.index[i]
                
                # ---- POSITION MANAGEMENT LOGIC ----
                if open_position:
                    outcome = None
                    exit_price = None

                    # 1. Check for Stop Loss or Take Profit
                    if open_position['direction'] == 'long':
                        if current_candle['high'] >= open_position['take_profit']:
                            outcome, exit_price = 'TAKE_PROFIT', open_position['take_profit']
                        elif current_candle['low'] <= open_position['stop_loss']:
                            outcome, exit_price = 'STOP_LOSS', open_position['stop_loss']
                    else: # short
                        if current_candle['low'] <= open_position['take_profit']:
                            outcome, exit_price = 'TAKE_PROFIT', open_position['take_profit']
                        elif current_candle['high'] >= open_position['stop_loss']:
                            outcome, exit_price = 'STOP_LOSS', open_position['stop_loss']
                    
                    # 2. Check for Reversal Signal (simplified for backtesting)
                    if not outcome:
                        # A more complex simulation could re-run the model here.
                        # For now, we use a simple time-based exit if SL/TP is not hit.
                        hold_hours = strategy_config.get('hold_period_hours', 24 * 5) # 5-day hold default
                        if current_time >= open_position['entry_time'] + pd.Timedelta(hours=hold_hours):
                            outcome, exit_price = 'TIME_EXIT', current_price

                    if outcome:
                        self._close_trade(open_position, exit_price, current_time, outcome)
                        open_position = None # Position is now closed
                
                # ---- NEW TRADE LOGIC ----
                if not open_position:
                    window_data = df.iloc[i - sequence_length : i]
                    
                    prediction = self.analyst.predict(symbol, window_data, strategy_name=strategy_type)

                    if prediction and self.risk_manager.should_execute(prediction, strategy_type):
                        print(f"🎯 {current_time}: Opening {prediction['direction'].upper()} trade at ${current_price:.2f}")
                        open_position = self._open_trade(prediction, current_time, strategy_type, window_data)

            # Final Metrics Calculation
            print(f"\n✅ Backtest scan complete for {symbol}.")
            if not self.trade_history:
                print("⚠️ No trades were executed in this backtest.")
                return get_empty_metrics()

            metrics = calculate_all_metrics(self.trade_history)
            self.results.append({'symbol': symbol, 'strategy': strategy_type, 'metrics': metrics, 'trades': self.trade_history})
            return metrics

        except Exception:
            print(f"🔥 An unexpected error occurred during the backtest for {symbol}:")
            print(traceback.format_exc())
            return get_empty_metrics()

    def _open_trade(self, prediction, entry_time, strategy_name, df):
        """Creates a new virtual position."""
        entry_price = prediction['current_price']
        direction = prediction['direction']
        
        rules = STRATEGIES[strategy_name]
        levels = self.risk_manager.calculate_levels(prediction, df)

        position = {
            'asset': prediction['asset'],
            'direction': direction,
            'entry_price': entry_price,
            'entry_time': entry_time,
            'stop_loss': levels['stop_loss'],
            'take_profit': levels['take_profit'],
            'strategy_name': strategy_name,
            'confidence': prediction['confidence']
        }
        return position

    def _close_trade(self, position, exit_price, exit_time, outcome):
        """Closes the virtual position and logs the trade."""
        pnl = 0
        fees = 0.001 # 0.1% fee
        
        if position['direction'] == 'long':
            pnl = ((exit_price - position['entry_price']) / position['entry_price']) - fees
        else: # short
            pnl = ((position['entry_price'] - exit_price) / position['entry_price']) - fees

        trade = {
            'symbol': position['asset'],
            'entry_time': position['entry_time'],
            'exit_time': exit_time,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'type': position['direction'],
            'pnl': pnl * 100, # As percentage
            'status': 'closed',
            'outcome': outcome,
            'confidence': position['confidence']
        }
        self.trade_history.append(trade)
        result_emoji = "✅" if pnl > 0 else "❌"
        print(f"{result_emoji} {exit_time}: Closing {position['direction'].upper()} trade. Outcome: {outcome}. PnL: {pnl*100:.2f}%")

    def generate_report(self, symbol):
        """Create a text-based report."""
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