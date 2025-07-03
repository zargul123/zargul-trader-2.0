import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scripts.core.data_engine import DataMaster
from scripts.config import ASSETS, TECHNICAL_INDICATORS
from .strategies import MainStrategy, SwingStrategy, ScalpStrategy
import matplotlib.pyplot as plt
import seaborn as sns
import os

class BacktestEngine:
    def __init__(self):
        self.data = DataMaster()
        self.results = []
        self.trade_history = []

    def load_data(self, symbol, days=180, timeframe="1h"):
        df = self.data.get_data(symbol, timeframe)
        print(f"✅ Loaded {len(df)} rows for {symbol} ({timeframe}). Columns: {list(df.columns)}")  # DEBUG LINE
        print(f"📅 Date Range: {df.index[0]} to {df.index[-1]}")  # DEBUG LINE
        return df[df['volume'] > 0].copy()

    def run_backtest(self, symbol, strategy_type="main", days=30):
        """Updated to handle strategy parameter"""
        try:
            print(f"\n🔍 Backtesting {symbol} ({strategy_type.upper()} Strategy)...")
            
            # Load strategy config
            config = STRATEGIES[strategy_type]
            
            # Get data with correct timeframe
            df = self.data.get_data(symbol, config['timeframe'])
            df = df.iloc[-days*24:]  # Get requested days (converted to hours)
            
            # Initialize strategy
            strategy = self.get_strategy(strategy_type)
            strategy.set_parameters(config)
            
            self.open_trades = []  # Track open trades
            
            # FORCE TEST TRADES - DELETE AFTER VERIFICATION
            df.at[df.index[50], 'signal'] = 1  # Force long at row 50
        df.at[df.index[100], 'signal'] = -1  # Force short at row 100

        # Prepare empty columns for our signals
        df['signal'] = 0  # 0=no trade, 1=long, -1=short
        df['position'] = 0  # Current position
        df['pnl'] = 0.0  # Profit/loss per trade

        # Get strategy rules
        strategy_rules = self.get_strategy(strategy_type)
        strategy_rules.open_trades = self.open_trades  # Share trade tracking
        
        # Set strategy-specific thresholds
        if hasattr(strategy_rules, 'set_thresholds'):
            strategy_rules.set_thresholds(
                long=config['long_threshold'],
                short=config['short_threshold'],
                confidence=config['min_confidence']
            )

        # Simulate trading
        for i in range(1, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]

            # Get trade signal
            signal = strategy_rules.get_signal(df[:i])  # Only use past data

            # Execute trade with realistic conditions
            if signal != 0 and prev['position'] == 0:
                # Enhanced trade execution with slippage (0.15%)
                entry_price = current['open'] * (1.0015 if signal == 1 else 0.9985)
                df.at[df.index[i], 'position'] = signal
                
                # Track open trades for strategy
                self.open_trades.append({
                    'entry_time': df.index[i],
                    'entry_price': entry_price
                })
                
                # When opening a trade (updated format):
                self.trade_history.append({
                    'symbol': symbol,
                    'entry_time': df.index[i],
                    'entry_price': entry_price,
                    'type': 'long' if signal == 1 else 'short',
                    'exit_time': None,  # Will be set when trade closes
                    'exit_price': None,
                    'pnl': 0.0,
                    'status': 'open'
                })

            # Exit logic
            elif prev['position'] != 0:
                exit_signal = strategy_rules.get_exit_signal(df[:i], prev['position'])
                if exit_signal:
                    # Enhanced trade execution with slippage (0.15%)
                    exit_price = current['open'] * (0.9985 if prev['position'] == 1 else 1.0015)
                    df.at[df.index[i], 'position'] = 0
                    
                    # Clear open trades on exit
                    self.open_trades.clear()
                    
                    # When closing a trade (updated format):
                    for trade in reversed(self.trade_history):
                        if trade['status'] == 'open' and trade['symbol'] == symbol:  # Add symbol check
                            pnl_percent = ((exit_price - trade['entry_price']) / trade['entry_price']) * 100
                            trade.update({
                                'exit_time': df.index[i],
                                'exit_price': exit_price,
                                'pnl': pnl_percent if trade['type'] == 'long' else -pnl_percent,
                                'status': 'closed'
                            })
                            break

        # Debug trade log
        print(f"\n🔍 TRADE LOG ({symbol}):")
        for i, trade in enumerate([t for t in self.trade_history if t['symbol'] == symbol][:5]):
            duration = "N/A" 
            if trade.get('exit_time'):
                duration = f"{(trade['exit_time']-trade['entry_time']).seconds/3600:.1f}h"
            print(f"Trade {i+1}: {trade['type']}@{trade['entry_price']:.2f} | PnL: {trade.get('pnl',0):.2f}% | Duration: {duration}")

        # Calculate metrics
        metrics = self.calculate_metrics(df)
        self.results.append({
            'symbol': symbol,
            'strategy': strategy_type,
            'metrics': metrics,
            'trades': [t for t in self.trade_history if t['symbol'] == symbol]
        })

        return metrics

    def get_strategy(self, name):
        """Get strategy rules"""
        if name == "main":
            return MainStrategy()
        elif name == "swing":
            return SwingStrategy()
        elif name == "scalp":
            return ScalpStrategy()
        else:
            raise ValueError(f"Unknown strategy: {name}")

    def calculate_position_size(self, volatility, df):
        """Better position sizing based on volatility"""
        atr = self._calculate_atr(df)
        account_risk = 0.01  # Risk 1% of account per trade
        return (account_risk * 10000) / (atr * 1.5)  # 1.5x ATR stop

    def _calculate_atr(self, df, period=14):
        """Average True Range calculation"""
        import numpy as np
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(period).mean().iloc[-1]

    def calculate_metrics(self, df):
        """Calculate performance metrics"""
        from .metrics import calculate_all_metrics
        return calculate_all_metrics(self.trade_history)

    def generate_report(self, symbol):
        """Create visual report"""
        from .reports import generate_html_report
        result = next(r for r in self.results if r['symbol'] == symbol)
        generate_html_report(result)

        print(f"✅ Backtest report saved to backtest_reports/{symbol}_report.html")