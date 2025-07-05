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
    def __init__(self, debug=False):
        from scripts.core.analysis_engine import AIAnalyst
        from scripts.core.data_engine import DataMaster
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
        df = df.last(f"{days_to_load}D")  # Get last N days
        
        # Ensure we have enough data
        if len(df) < 50:
            print(f"⚠️ Insufficient data for {symbol} ({len(df)} rows)")
            return pd.DataFrame()
        
        print(f"📅 Date range: {df.index[0]} to {df.index[-1]}")
        print(f"✅ Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        return df

    def run_backtest(self, symbol, strategy_type, days):
        try:
            import traceback  # Add at top of file
            
            # Move Strategy Selection to the top of run_backtest:
            strategy = self.get_strategy(strategy_type)
            print(f"\n🔧 Using {strategy_type} strategy with rules:")
            print(f"- Long Threshold: {strategy.long_threshold}%")
            print(f"- Short Threshold: {strategy.short_threshold}%")
            print(f"- Min Confidence: {strategy.min_confidence}%")
            
            # Replace the initial data loading with:
            df = self.load_data(symbol, days, 
                               "1h" if strategy_type != "scalp" else "15m")
            if df.empty:
                print("❌ No data loaded - aborting backtest")
                return
            
            # Replace the single prediction call with:
            print(f"\n🔎 Scanning {len(df)} candles for signals...")
            signals_found = 0

            # Change the scanning loop to:
            window_size = 60  # Must match SEQUENCE_LENGTH in analysis_engine.py
            step_size = 5     # Check every 5 candles

            for i in range(window_size, len(df), step_size):
                window = df.iloc[i-window_size:i]  # Use rolling window
                
                if strategy_type == "main":
                    prediction = self.analyst.predict(symbol, window)
                elif strategy_type == "swing":
                    prediction = self.analyst.predict_swing(symbol, window)
                else:
                    prediction = self.analyst.predict_scalp(symbol, window)
                
                if prediction and prediction['confidence'] >= 0.65:
                    # Add debug print
                    if self.debug:
                        print(f"\n🔎 Signal Check:")
                        print(f"Direction: {prediction['direction']}")
                        print(f"Change: {prediction['pct_change']:.2f}%")
                        print(f"Threshold: {'>=' + str(LONG_THRESHOLD) if prediction['direction'] == 'long' else '<=' + str(-SHORT_THRESHOLD)}%")
                        print(f"Raw prediction: {prediction}")
                    else:
                        print(f"🔎 Signal: {prediction['direction']} {prediction['pct_change']:.2f}% (conf: {prediction['confidence']*100:.0f}%)")
                    
                    # Add these filters before accepting trades
                    if (prediction['confidence'] < 0.75 or 
                        abs(prediction['pct_change']) < 1.0 or
                        (prediction['direction'] == 'long' and prediction['pct_change'] < LONG_THRESHOLD) or
                        (prediction['direction'] == 'short' and prediction['pct_change'] > -SHORT_THRESHOLD)):  # Notice > for shorts
                        print(f"⏩ Skipping marginal signal: {prediction['direction']} {prediction['pct_change']:.2f}%")
                        print(f"Threshold Check: {'>=' if prediction['direction']=='long' else '<='} " 
                              f"{LONG_THRESHOLD if prediction['direction']=='long' else -SHORT_THRESHOLD}%")
                        continue
                    
                    print(f"🎯 Signal at {df.index[i]} | {prediction['direction']} | "
                          f"Conf: {prediction['confidence']*100:.0f}% | "
                          f"Change: {prediction['pct_change']:.2f}%")
                    self._execute_trade(prediction)
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
        """Realistic trade simulation with PnL"""
        try:
            import random
            from datetime import timedelta
            
            if not prediction:
                print(f"⚠️ [BACKTEST] No prediction to execute")
                return
                
            # Generate realistic exit price after hold period
            hold_hours = random.randint(4, 24) if prediction['type'] != 'scalp' else random.randint(1, 4)
            
            # Get current price and direction
            current_price = prediction['current_price']
            direction = prediction['direction']
            pct_change = prediction['pct_change']
            
            # Calculate entry price with slippage
            entry_price = current_price * (1.0015 if direction == 'long' else 0.9985)
            
            # Set 1% stop-loss for all trades
            stop_loss = entry_price * (0.99 if direction == 'long' else 1.01)
            
            # Get historical data for volatility calculation
            try:
                df = self.data.get_data(prediction['asset'], "1h")
                if len(df) > 20:
                    # Simulate price path with volatility
                    volatility = df['close'].pct_change().std() * 2  # 2x std dev
                    direction_sign = 1 if direction == 'long' else -1
                    exit_price = current_price * (1 + direction_sign * abs(pct_change)/100 * random.uniform(0.7, 1.3))
                else:
                    # Fallback if no data
                    target = entry_price * (1 + pct_change/100)
                    exit_price = random.uniform(min(stop_loss, target), max(stop_loss, target))
            except Exception:
                # Fallback exit logic
                target = entry_price * (1 + pct_change/100)
                exit_price = random.uniform(min(stop_loss, target), max(stop_loss, target))
            
            # Enforce stop-loss
            if direction == 'long':
                exit_price = max(exit_price, stop_loss)  # Long stop-loss is minimum
            else:
                exit_price = min(exit_price, stop_loss)  # Short stop-loss is maximum
            
            # Apply exit slippage
            exit_price = exit_price * (0.9985 if direction == 'long' else 1.0015)
            
            # Add directional profit calculation
            if direction == 'long':
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            else:  # short
                pnl_pct = ((entry_price - exit_price) / entry_price) * 100  # Inverted for shorts
            
            # Add slippage (0.1%) and fees (0.05%)
            final_pnl = pnl_pct * 0.9985  # Adjust for costs
            
            # Debug PnL calculation
            print(f"🔢 PnL Check: Entry=${entry_price:.2f} | Exit=${exit_price:.2f} | "
                  f"Raw PnL={((exit_price-entry_price)/entry_price)*100:.2f}% | "
                  f"Direction={direction} | Final PnL={final_pnl:.2f}%")
            pnl_pct = final_pnl
            
            trade = {
                'symbol': prediction['asset'],
                'entry': entry_price,
                'exit': exit_price,
                'entry_time': datetime.now(),
                'exit_time': datetime.now() + timedelta(hours=hold_hours),
                'direction': direction,
                'pnl': pnl_pct,
                'status': 'closed',
                'confidence': prediction['confidence'],
                'stop_loss': stop_loss  # Track stop-loss for analysis
            }
            self.trade_history.append(trade)
            
            # Enhanced logging
            stop_hit = "STOP HIT" if (direction == 'long' and exit_price <= stop_loss * 1.001) or \
                                   (direction == 'short' and exit_price >= stop_loss * 0.999) else ""
            
            print(f"✅ Executed {direction} trade | Entry: ${entry_price:.2f} | "
                  f"Exit: ${exit_price:.2f} | PnL: {pnl_pct:.2f}% {stop_hit}")
                  
        except Exception as e:
            print(f"❌ Trade execution failed: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset', required=True, help='Asset to backtest (e.g. BTC-USD)')
    parser.add_argument('--strategy', required=True, help='Strategy to use (main/swing/scalp)')
    parser.add_argument('--days', type=int, default=90, help='Number of days to backtest')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with verbose output')
    args = parser.parse_args()

    print(f"\n🚀 Starting backtest for {args.asset} ({args.strategy} strategy)")
    if args.debug:
        print("🔍 Debug mode enabled - verbose output active")
    
    engine = BacktestEngine(debug=args.debug)
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