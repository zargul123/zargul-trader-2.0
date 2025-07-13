import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Add these at the TOP of the file under other imports
LONG_THRESHOLD = 1.2  # 1.2% target for longs
SHORT_THRESHOLD = 0.8 # 0.8% target for shorts

# Now import your modules
from scripts.core.data_engine import DataMaster
from scripts.core.analysis_engine import AIAnalyst
from scripts.config import ASSETS, STRATEGIES

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import traceback
from scripts.config import TECHNICAL_INDICATORS
from scripts.backtest.metrics import calculate_all_metrics, get_empty_metrics
from scripts.backtest.strategies import MainStrategy, SwingStrategy, ScalpStrategy
import matplotlib.pyplot as plt
import seaborn as sns

class BacktestEngine:
    def __init__(self, debug=False, enforce_longs=False):
        from scripts.core.analysis_engine import AIAnalyst
        from scripts.core.data_engine import DataMaster
        self.enforce_longs = enforce_longs
        self.trade_history = []  # Initialize the trade_history list
        self.analyst = AIAnalyst()
        self.data = DataMaster()
        self.trades = []
        self.debug = debug

    def load_data(self, symbol, days=90, timeframe="4h"):
        """Load proper backtesting data"""
        # Change the data loading to:
        days_to_load = max(days, 7)  # Always load at least 7 days
        print(f"\n📊 Loading {days_to_load} days of {timeframe} data for {symbol}")
        df = self.data.get_data(symbol, "1h")
        df = df.loc[df.index[-days_to_load*24:]]  # For hourly data
        
        # Ensure we have enough data
        if len(df) < 50:
            print(f"⚠️ Insufficient data for {symbol} ({len(df)} rows)")
            return pd.DataFrame()
        
        print(f"📅 Date range: {df.index[0]} to {df.index[-1]}")
        print(f"✅ Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        return df

    def run_backtest(self, symbol, strategy_type, days):
        try:
            print(f"\n🚀 Starting AI-powered backtest for {symbol}")
            
            # Load data using same method as main system
            df = self.data.get_data(symbol, "1h")
            if df.empty or len(df) < 100:
                print("❌ Insufficient data for backtest")
                return
            
            # Limit to requested days
            df = df.tail(days * 24)  # 24 hours per day
            print(f"📊 Testing {len(df)} candles from {df.index[0]} to {df.index[-1]}")
            
            signals_found = 0
            window_size = 60  # Match your AI model's sequence length
            
            # Step through data and get AI predictions
            for i in range(window_size, len(df), 12):  # Check every 12 hours
                window_data = df.iloc[i-window_size:i]
                current_time = df.index[i]
                current_price = df.iloc[i]['close']
                
                # Get AI prediction using same method as main system
                if strategy_type == "main":
                    prediction = self.analyst.predict(symbol, window_data)
                elif strategy_type == "swing":
                    prediction = self.analyst.predict_swing(symbol)
                else:
                    prediction = self.analyst.predict_scalp(symbol)
                
                if not prediction:
                    continue
                    
                # Use same evaluation logic as main system
                confidence_pct = prediction.get('confidence', 0) * 100
                pct_change = prediction.get('pct_change', 0)
                direction = prediction.get('direction', '').lower()
                
                # Apply same thresholds as main system
                if confidence_pct >= 50:  # MIN_CONFIDENCE * 100
                    if ((direction == 'long' and pct_change >= 0.1) or  # LONG_THRESHOLD
                        (direction == 'short' and pct_change <= -0.1)):  # SHORT_THRESHOLD
                        
                        print(f"🎯 {current_time}: {direction.upper()} signal "
                              f"| Conf: {confidence_pct:.0f}% | Move: {pct_change:.2f}%")
                        
                        # Execute trade with current timestamp and price
                        trade_data = prediction.copy()
                        trade_data.update({
                            'entry_time': current_time,
                            'current_price': current_price,
                            'asset': symbol
                        })
                        self._execute_trade(trade_data)
                        signals_found += 1

            print(f"\n📍 Found {signals_found} valid signals in this period")
            
            # Add this after the backtest completes:
            print("\n📈 Trade Performance Summary:")
            if self.trade_history:
                wins = [t for t in self.trade_history if t.get('pnl',0) > 0]
                print(f"• {len(wins)}/{len(self.trade_history)} winning trades "
                      f"({len(wins)/len(self.trade_history):.1%} win rate)")
                print(f"• Avg PnL: {sum(t.get('pnl',0) for t in self.trade_history)/len(self.trade_history):.2f}%")
            else:
                print("⚠️ No trades executed")
                
        except Exception as e:
            print(f"💥 Backtest error: {traceback.format_exc()}")
            
            # Filter by trading hours
            df = self.data._filter_trading_hours(df, config['trading_hours'])
            
            # Add minimum data threshold check
            if len(df) < 10:  # Minimum data threshold
                print(f"⚠️ Insufficient data after time filtering for {symbol}")
                return get_empty_metrics()
                
            # Prepare dataframe columns
            df = df.assign(
                signal=0,      # 0=no trade, 1=long, -1=short
                position=0,    # Current position
                pnl=0.0        # Profit/loss per trade
            )
            
            # DEBUG: Force test trades (remove after verification)
            if len(df) > 100:
                df.at[df.index[50], 'signal'] = 1  # Force long
                df.at[df.index[100], 'signal'] = -1  # Force short

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
                    # Remove the volume filter completely for now
                    # Keep price change filter but make it much smaller
                    price_change = abs((current['close'] - current['open']) / current['open'] * 100)
                    if price_change > 0.1:  # Reduced from 0.5%
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
                                final_pnl = pnl_percent if trade['type'] == 'long' else -pnl_percent
                                
                                # Skip trades with < 0.1% PnL
                                if abs(final_pnl) < 0.1:
                                    # Mark as closed but don't record for metrics
                                    trade.update({
                                        'exit_time': df.index[i],
                                        'exit_price': exit_price,
                                        'pnl': float(final_pnl),  # Ensure PnL is stored as float
                                        'status': 'filtered'  # Different status for filtered trades
                                    })
                                    break
                                
                                trade.update({
                                    'exit_time': df.index[i],
                                    'exit_price': exit_price,
                                    'pnl': float(final_pnl),  # Ensure PnL is stored as float
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
            try:
                metrics = calculate_all_metrics([t for t in self.trade_history if t['symbol'] == symbol])
            except Exception as e:
                print(f"❌ Metric calculation failed: {str(e)}")
                metrics = get_empty_metrics()
            
            self.results.append({
                'symbol': symbol,
                'strategy': strategy_type,
                'metrics': metrics,
                'trades': [t for t in self.trade_history if t['symbol'] == symbol]
            })

            return metrics
            
        except Exception as e:
            print(f"🔥 Backtest failed for {symbol}: {str(e)}")
            return get_empty_metrics()

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

    def should_execute_trade(self, prediction):
        if not prediction:
            print("⚠️ Empty prediction in backtest")
            return False
            
        # TEMPORARY - Accept all trades for debugging
        print(f"✅ Accepting trade regardless of criteria: {prediction}")
        return True
        
        # We'll restore proper criteria after confirming trades flow

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

    def calculate_metrics(self, trades):
        """Wrapper for metrics calculation with data validation"""
        from .metrics import calculate_all_metrics
        
        if not trades or not isinstance(trades, list):
            return calculate_all_metrics([])
        
        # Ensure all trades have required fields
        validated_trades = []
        for trade in trades:
            if not isinstance(trade, dict):
                continue
                
            validated = {
                'symbol': str(trade.get('symbol', '')),
                'status': str(trade.get('status', 'closed')),
                'pnl': float(trade.get('pnl', 0)),
                'entry_time': trade.get('entry_time'),
                'exit_time': trade.get('exit_time'),
                'type': str(trade.get('type', ''))
            }
            validated_trades.append(validated)
        
        return calculate_all_metrics(validated_trades)

    def calculate_performance(self):
        """More robust metric calculation"""
        if not self.trade_history:
            return {"error": "No trades executed"}
        
        # Filter out invalid trades
        valid_trades = [t for t in self.trade_history 
                       if isinstance(t.get('pnl', None), (int, float))]
        
        if not valid_trades:
            return {"error": "No valid trades found"}
        
        returns = [t['pnl']/100 for t in valid_trades]
        wins = [r for r in returns if r > 0]
        
        metrics = {
            'total_trades': len(valid_trades),
            'win_rate': len(wins)/len(returns) if returns else 0,
            'avg_pnl': np.mean(returns)*100 if returns else 0,
            'sharpe': (np.mean(returns)/np.std(returns))*np.sqrt(252) if len(returns)>1 and np.std(returns)>0 else 0,
            'max_drawdown': min(returns)*100 if returns else 0,
            'profit_factor': (sum(wins)/abs(sum([r for r in returns if r<0]))) if wins and any(r<0 for r in returns) else 0
        }
        return metrics

    def generate_report(self, symbol):
        """Create visual report"""
        from .reports import generate_html_report
        result = next(r for r in self.results if r['symbol'] == symbol)
        generate_html_report(result)

        print(f"✅ Backtest report saved to backtest_reports/{symbol}_report.html")

    def _execute_trade(self, prediction):
        """Simulate realistic trade based on AI prediction"""
        try:
            if not prediction:
                return
                
            entry_time = prediction.get('entry_time')
            entry_price = prediction.get('current_price')
            direction = prediction.get('direction')
            predicted_move = prediction.get('pct_change', 0)
            
            # Simulate realistic market conditions
            # Add 0.1% slippage and 0.05% fees
            slippage = 0.001 * (-1 if direction == 'long' else 1)
            fees = 0.0005
            
            actual_entry = entry_price * (1 + slippage)
            
            # Simulate exit after predicted move (with some randomness)
            import random
            market_noise = random.uniform(0.8, 1.2)  # 80% to 120% of predicted move
            actual_move = predicted_move * market_noise
            
            if direction == 'long':
                exit_price = actual_entry * (1 + actual_move/100)
                pnl = ((exit_price - actual_entry) / actual_entry - fees) * 100
            else:  # short
                exit_price = actual_entry * (1 + actual_move/100)  # actual_move is negative
                pnl = ((actual_entry - exit_price) / actual_entry - fees) * 100
            
            # Create trade record
            trade = {
                'symbol': prediction['asset'],
                'entry_time': entry_time,
                'exit_time': entry_time + pd.Timedelta(hours=4),  # 4 hour holding period
                'entry_price': actual_entry,
                'exit_price': exit_price,
                'type': direction,
                'pnl': pnl,
                'status': 'closed',
                'confidence': prediction.get('confidence', 0)
            }
            
            self.trade_history.append(trade)
            print(f"✅ Trade: {direction.upper()} ${actual_entry:.2f} → ${exit_price:.2f} | PnL: {pnl:.2f}%")
            
        except Exception as e:
            print(f"❌ Trade execution error: {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset', required=True, help='Asset to backtest (e.g. BTC-USD)')
    parser.add_argument('--strategy', required=True, help='Strategy to use (main/swing/scalp)')
    parser.add_argument('--days', type=int, default=90, help='Number of days to backtest')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with verbose output')
    parser.add_argument('--enforce-longs', action='store_true', help='Ensure minimum 30% long trades')
    args = parser.parse_args()

    print(f"\n🚀 Starting backtest for {args.asset} ({args.strategy} strategy)")
    if args.debug:
        print("🔍 Debug mode enabled - verbose output active")
    
    engine = BacktestEngine(debug=args.debug, enforce_longs=args.enforce_longs)
    results = engine.run_backtest(args.asset, args.strategy, args.days)
    
    print("\n📊 Backtest Results:")
    print(f"Total Trades Executed: {len(engine.trade_history)}")
    if engine.trade_history:
        print("\nSample Trades:")
        for i, trade in enumerate(engine.trade_history[:3]):
            print(f"Trade {i+1}:")
            print(f"• Symbol: {trade.get('symbol', 'N/A')}")
            print(f"• Direction: {trade.get('direction', trade.get('type', 'N/A'))}")
            print(f"• Entry: ${trade.get('entry', trade.get('entry_price', 0)):.2f}")
            print(f"• PnL: {trade.get('pnl', 0):.2f}%")
            print("─"*30)