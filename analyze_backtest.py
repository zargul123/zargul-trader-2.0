#!/usr/bin/env python3
"""
Backtest Analysis for August 20th - September 10th, 2025
Analyzes BTC-USD trading results and various scenarios
"""

import re
import pandas as pd
import numpy as np
from datetime import datetime
import os

def parse_backtest_logs(log_file, start_date="2025-08-20", end_date="2025-09-10"):
    """Extract trades from backtest log files for the specified date range"""
    
    trades = []
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Find all TRADE CLOSED blocks
    trade_blocks = re.findall(r'TRADE CLOSED: BTC-USD - (SHORT|LONG).*?PnL \(Net\):\s*([\d.-]+)%.*?Emoji:\s*✅', 
                             content, re.DOTALL)
    
    # More detailed pattern to extract all trade information
    detailed_pattern = r'TRADE CLOSED: BTC-USD - (SHORT|LONG)\s*.*?Entry Time:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*.*?Exit Time:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*.*?Entry Price:\s*\$?([\d.,]+)\s*.*?Exit Price:\s*\$?([\d.,]+)\s*.*?Stop Loss:\s*\$?([\d.,]+)\s*.*?Take Profit:\s*\$?([\d.,]+)\s*.*?AI Confidence:\s*([\d.]+)%\s*.*?Outcome:\s*(\w+)\s*.*?PnL \(Net\):\s*([\d.-]+)%'
    
    matches = re.findall(detailed_pattern, content, re.DOTALL)
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    for match in matches:
        direction, entry_time_str, exit_time_str, entry_price_str, exit_price_str, stop_loss_str, take_profit_str, confidence_str, outcome, pnl_str = match
        
        # Parse timestamps
        try:
            entry_time = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
            exit_time = datetime.strptime(exit_time_str, "%Y-%m-%d %H:%M:%S")
            
            # Filter by date range
            if start_dt.date() <= entry_time.date() <= end_dt.date():
                
                # Clean and convert prices
                entry_price = float(entry_price_str.replace(',', ''))
                exit_price = float(exit_price_str.replace(',', ''))
                stop_loss = float(stop_loss_str.replace(',', ''))
                take_profit = float(take_profit_str.replace(',', ''))
                confidence = float(confidence_str)
                pnl_log = float(pnl_str)
                
                # Calculate correct PnL based on direction
                if direction == "SHORT":
                    pnl_correct = ((entry_price - exit_price) / entry_price) * 100
                else:  # LONG
                    pnl_correct = ((exit_price - entry_price) / entry_price) * 100
                
                # Determine correct outcome based on price action
                if direction == "SHORT":
                    if exit_price <= take_profit:
                        outcome_correct = "TAKE_PROFIT"
                    elif exit_price >= stop_loss:
                        outcome_correct = "STOP_LOSS"
                    else:
                        outcome_correct = "TIME_EXIT"
                else:  # LONG
                    if exit_price >= take_profit:
                        outcome_correct = "TAKE_PROFIT"
                    elif exit_price <= stop_loss:
                        outcome_correct = "STOP_LOSS"
                    else:
                        outcome_correct = "TIME_EXIT"
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'direction': direction,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': confidence,
                    'outcome_raw': outcome,
                    'outcome_corrected': outcome_correct,
                    'pnl_log_pct': pnl_log,
                    'pnl_correct_pct': round(pnl_correct, 4)
                })
                
        except Exception as e:
            print(f"Error parsing trade: {e}")
            continue
    
    return pd.DataFrame(trades)

def load_market_data():
    """Try to load historical market data for verification"""
    
    # Check if we have historical data file
    historical_files = [
        'historical_data_for_analysis.csv',
        'btc_historical_data.csv',
        'market_data.csv'
    ]
    
    for file_path in historical_files:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"Found historical data file: {file_path}")
                return df
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    print("No historical data files found - verification will be limited")
    return None

def verify_trades_against_market_data(trades_df, market_df):
    """Verify that trade exit prices are within market candle ranges"""
    
    if market_df is None:
        print("No market data available for verification")
        return trades_df
    
    # Add verification columns
    trades_df['verified'] = False
    trades_df['verification_notes'] = ''
    
    print("Market data verification not implemented in this version")
    return trades_df

def analyze_sl_tp_swap_impact(trades_df):
    """Analyze what would happen if we swap stop loss and take profit levels"""
    
    results = []
    
    for _, trade in trades_df.iterrows():
        
        # Original configuration (as logged)
        original_outcome = trade['outcome_corrected']
        original_pnl = trade['pnl_correct_pct']
        
        # Swapped configuration - swap the SL and TP levels
        swapped_sl = trade['take_profit']
        swapped_tp = trade['stop_loss']
        
        # Calculate what would happen with swapped levels
        if trade['direction'] == "SHORT":
            if trade['exit_price'] <= swapped_tp:  # Now TP is where SL was
                swapped_outcome = "TAKE_PROFIT"
                swapped_pnl = ((trade['entry_price'] - swapped_tp) / trade['entry_price']) * 100
            elif trade['exit_price'] >= swapped_sl:  # Now SL is where TP was
                swapped_outcome = "STOP_LOSS"
                swapped_pnl = ((trade['entry_price'] - swapped_sl) / trade['entry_price']) * 100
            else:
                swapped_outcome = "TIME_EXIT"
                swapped_pnl = original_pnl
        else:  # LONG
            if trade['exit_price'] >= swapped_tp:
                swapped_outcome = "TAKE_PROFIT"
                swapped_pnl = ((swapped_tp - trade['entry_price']) / trade['entry_price']) * 100
            elif trade['exit_price'] <= swapped_sl:
                swapped_outcome = "STOP_LOSS"
                swapped_pnl = ((swapped_sl - trade['entry_price']) / trade['entry_price']) * 100
            else:
                swapped_outcome = "TIME_EXIT"
                swapped_pnl = original_pnl
        
        results.append({
            'entry_time': trade['entry_time'],
            'direction': trade['direction'],
            'original_outcome': original_outcome,
            'original_pnl': original_pnl,
            'swapped_outcome': swapped_outcome,
            'swapped_pnl': round(swapped_pnl, 4),
            'pnl_difference': round(swapped_pnl - original_pnl, 4)
        })
    
    return pd.DataFrame(results)

def analyze_time_based_effects(trades_df):
    """Analyze performance by hour of day and different exit timing"""
    
    # Add hour of day
    trades_df['entry_hour'] = trades_df['entry_time'].dt.hour
    
    # Group by hour and calculate metrics
    hourly_stats = trades_df.groupby('entry_hour').agg({
        'pnl_correct_pct': ['mean', 'median', 'std', 'count'],
        'outcome_corrected': lambda x: (x == 'TAKE_PROFIT').sum() / len(x)
    }).round(4)
    
    hourly_stats.columns = ['avg_pnl', 'median_pnl', 'pnl_std', 'trade_count', 'win_rate']
    
    return hourly_stats

def main():
    print("=" * 60)
    print("BTC-USD BACKTEST ANALYSIS: August 20 - September 10, 2025")
    print("=" * 60)
    
    # Parse the backtest data
    print("\n1. Extracting backtest data...")
    trades_df = parse_backtest_logs('backtest_btc_main_log_1.log', "2025-08-20", "2025-09-10")
    
    if len(trades_df) == 0:
        print("No trades found in the specified date range!")
        return
    
    print(f"Found {len(trades_df)} trades in the specified period")
    
    # Save the parsed data
    trades_df.to_csv('backtest_aug20_sep10.csv', index=False)
    print("Saved parsed data to: backtest_aug20_sep10.csv")
    
    # Basic statistics
    print("\n2. Basic Trade Statistics:")
    print(f"Total Trades: {len(trades_df)}")
    print(f"Average PnL: {trades_df['pnl_correct_pct'].mean():.4f}%")
    print(f"Total PnL: {trades_df['pnl_correct_pct'].sum():.4f}%")
    print(f"Win Rate: {(trades_df['pnl_correct_pct'] > 0).mean()*100:.2f}%")
    print(f"Direction Distribution:")
    print(trades_df['direction'].value_counts())
    
    # Outcome analysis
    print(f"\nOriginal vs Corrected Outcomes:")
    print("Original outcomes from log:")
    print(trades_df['outcome_raw'].value_counts())
    print("Corrected outcomes based on price action:")
    print(trades_df['outcome_corrected'].value_counts())
    
    # Load market data for verification
    print("\n3. Loading market data for verification...")
    market_df = load_market_data()
    verified_trades = verify_trades_against_market_data(trades_df, market_df)
    
    # Analyze SL/TP swap impact
    print("\n4. Analyzing Stop Loss / Take Profit Swap Impact...")
    swap_analysis = analyze_sl_tp_swap_impact(trades_df)
    
    # Calculate summary statistics for swap analysis
    original_stats = {
        'win_rate': (swap_analysis['original_pnl'] > 0).mean(),
        'avg_pnl': swap_analysis['original_pnl'].mean(),
        'total_pnl': swap_analysis['original_pnl'].sum()
    }
    
    swapped_stats = {
        'win_rate': (swap_analysis['swapped_pnl'] > 0).mean(), 
        'avg_pnl': swap_analysis['swapped_pnl'].mean(),
        'total_pnl': swap_analysis['swapped_pnl'].sum()
    }
    
    print("SL/TP Swap Analysis Results:")
    print(f"Original - Win Rate: {original_stats['win_rate']*100:.2f}%, Avg PnL: {original_stats['avg_pnl']:.4f}%, Total: {original_stats['total_pnl']:.4f}%")
    print(f"Swapped  - Win Rate: {swapped_stats['win_rate']*100:.2f}%, Avg PnL: {swapped_stats['avg_pnl']:.4f}%, Total: {swapped_stats['total_pnl']:.4f}%")
    print(f"Impact: Win Rate {(swapped_stats['win_rate']-original_stats['win_rate'])*100:+.2f}pp, Total PnL {swapped_stats['total_pnl']-original_stats['total_pnl']:+.4f}%")
    
    swap_analysis.to_csv('sl_tp_swap_analysis.csv', index=False)
    print("Saved SL/TP swap analysis to: sl_tp_swap_analysis.csv")
    
    # Time-based analysis
    print("\n5. Analyzing Time-Based Effects...")
    hourly_performance = analyze_time_based_effects(trades_df)
    
    print("Performance by Hour of Day (UTC):")
    print(hourly_performance.head(10))
    
    # Find best performing hours
    best_hours = hourly_performance.nlargest(3, 'avg_pnl')
    print(f"\nTop 3 performing hours:")
    for hour, stats in best_hours.iterrows():
        print(f"  {hour:02d}:00 - Avg PnL: {stats['avg_pnl']:.4f}%, Win Rate: {stats['win_rate']*100:.1f}%, Trades: {int(stats['trade_count'])}")
    
    hourly_performance.to_csv('time_of_day_analysis.csv')
    print("Saved time analysis to: time_of_day_analysis.csv")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("Files generated:")
    print("- backtest_aug20_sep10.csv (parsed trade data)")
    print("- sl_tp_swap_analysis.csv (stop loss/take profit swap analysis)")
    print("- time_of_day_analysis.csv (hourly performance analysis)")

if __name__ == "__main__":
    main()