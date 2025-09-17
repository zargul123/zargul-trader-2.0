#!/usr/bin/env python3
"""
Proper Stop Loss / Take Profit Verification
Check if market reached TP before SL, and what happens if we swap SL/TP
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def load_market_data():
    """Load historical market data"""
    hist_file = "../historical_data_for_analysis.csv"
    if os.path.exists(hist_file):
        try:
            df = pd.read_csv(hist_file)
            print(f"Found historical data file with {len(df)} records")
            
            # Try to identify datetime column
            datetime_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            if datetime_cols:
                df['datetime'] = pd.to_datetime(df[datetime_cols[0]])
                
                # Try to identify OHLC columns
                price_mapping = {}
                for col in df.columns:
                    col_lower = col.lower()
                    if 'open' in col_lower:
                        price_mapping['open'] = col
                    elif 'high' in col_lower:
                        price_mapping['high'] = col
                    elif 'low' in col_lower:
                        price_mapping['low'] = col
                    elif 'close' in col_lower:
                        price_mapping['close'] = col
                
                # Rename columns to standard format
                for standard, original in price_mapping.items():
                    df[standard] = pd.to_numeric(df[original], errors='coerce')
                
                # Keep only relevant columns
                df = df[['datetime', 'open', 'high', 'low', 'close']].copy()
                df = df.dropna()
                df = df.sort_values('datetime').reset_index(drop=True)
                
                return df
        except Exception as e:
            print(f"Error reading historical file: {e}")
    
    return None

def check_sl_tp_hit_order(trade, market_data):
    """
    Check which was hit first: Stop Loss or Take Profit
    For SHORT trades:
    - Take Profit should be BELOW entry price
    - Stop Loss should be ABOVE entry price
    """
    entry_time = pd.to_datetime(trade['entry_time'])
    
    # Get trade parameters
    entry_price = trade['entry_price']
    stop_loss = trade['stop_loss']
    take_profit = trade['take_profit']
    direction = trade['direction']
    
    # Filter market data from entry time onwards (look ahead for several hours)
    market_future = market_data[market_data['datetime'] >= entry_time].copy()
    
    if market_future.empty:
        return {
            'status': 'NO_DATA',
            'message': 'No market data available after entry time'
        }
    
    # For each subsequent candle, check if SL or TP was hit
    sl_hit_time = None
    tp_hit_time = None
    sl_hit_candle = None
    tp_hit_candle = None
    
    for idx, candle in market_future.iterrows():
        candle_high = candle['high']
        candle_low = candle['low']
        candle_time = candle['datetime']
        
        # Skip the entry candle if it's exactly at entry time
        if candle_time == entry_time:
            continue
        
        if direction == 'SHORT':
            # For SHORT: SL is above entry, TP is below entry
            # SL hit if price goes UP to stop_loss level
            # TP hit if price goes DOWN to take_profit level
            
            if sl_hit_time is None and candle_high >= stop_loss:
                sl_hit_time = candle_time
                sl_hit_candle = candle
            
            if tp_hit_time is None and candle_low <= take_profit:
                tp_hit_time = candle_time
                tp_hit_candle = candle
        
        elif direction == 'LONG':
            # For LONG: SL is below entry, TP is above entry
            # SL hit if price goes DOWN to stop_loss level
            # TP hit if price goes UP to take_profit level
            
            if sl_hit_time is None and candle_low <= stop_loss:
                sl_hit_time = candle_time
                sl_hit_candle = candle
            
            if tp_hit_time is None and candle_high >= take_profit:
                tp_hit_time = candle_time
                tp_hit_candle = candle
        
        # If both are hit, break (we found the first one)
        if sl_hit_time is not None and tp_hit_time is not None:
            break
        
        # Look only 24 hours ahead maximum
        if (candle_time - entry_time).total_seconds() > 24 * 3600:
            break
    
    # Determine which was hit first
    result = {
        'status': 'ANALYZED',
        'sl_hit_time': sl_hit_time,
        'tp_hit_time': tp_hit_time,
        'sl_hit_candle': sl_hit_candle,
        'tp_hit_candle': tp_hit_candle
    }
    
    if sl_hit_time is None and tp_hit_time is None:
        result['outcome'] = 'NEITHER_HIT'
        result['winner'] = 'NEITHER'
        result['message'] = 'Neither SL nor TP was hit in the time period'
    elif tp_hit_time is None:
        result['outcome'] = 'SL_HIT_FIRST'
        result['winner'] = 'STOP_LOSS'
        result['message'] = 'Stop Loss was hit first - LOSS'
    elif sl_hit_time is None:
        result['outcome'] = 'TP_HIT_FIRST'
        result['winner'] = 'TAKE_PROFIT'
        result['message'] = 'Take Profit was hit first - WIN'
    elif tp_hit_time < sl_hit_time:
        result['outcome'] = 'TP_HIT_FIRST'
        result['winner'] = 'TAKE_PROFIT'
        result['message'] = 'Take Profit was hit first - WIN'
    elif sl_hit_time < tp_hit_time:
        result['outcome'] = 'SL_HIT_FIRST'
        result['winner'] = 'STOP_LOSS'
        result['message'] = 'Stop Loss was hit first - LOSS'
    else:
        result['outcome'] = 'SAME_TIME'
        result['winner'] = 'TIE'
        result['message'] = 'SL and TP hit at same time'
    
    return result

def check_swapped_sl_tp_outcome(trade, market_data):
    """
    Check what would happen if we swapped SL and TP levels
    """
    # Create a swapped trade
    swapped_trade = trade.copy()
    swapped_trade['stop_loss'] = trade['take_profit']
    swapped_trade['take_profit'] = trade['stop_loss']
    
    # Analyze the swapped trade
    result = check_sl_tp_hit_order(swapped_trade, market_data)
    
    return result

def analyze_all_trades():
    """Analyze all trades for proper SL/TP verification"""
    
    print("PROPER STOP LOSS / TAKE PROFIT VERIFICATION")
    print("=" * 60)
    
    # Load backtest data
    backtest_file = 'backtest_aug20_sep10.csv'
    if not os.path.exists(backtest_file):
        print(f"ERROR: Backtest file {backtest_file} not found!")
        return
    
    df = pd.read_csv(backtest_file)
    print(f"Loaded {len(df)} trades for verification")
    
    # Load market data
    market_data = load_market_data()
    if market_data is None:
        print("ERROR: Could not load market data!")
        return
    
    print(f"Market data: {len(market_data)} candles from {market_data['datetime'].min()} to {market_data['datetime'].max()}")
    
    # Analyze each trade
    results = []
    swapped_results = []
    
    print(f"\nAnalyzing {len(df)} trades...")
    
    for idx, (_, trade) in enumerate(df.iterrows()):
        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(df)} trades...")
        
        # Original trade analysis
        original_result = check_sl_tp_hit_order(trade, market_data)
        results.append(original_result)
        
        # Swapped SL/TP analysis
        swapped_result = check_swapped_sl_tp_outcome(trade, market_data)
        swapped_results.append(swapped_result)
    
    # Create detailed report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("DETAILED SL/TP VERIFICATION RESULTS")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    original_wins = 0
    swapped_wins = 0
    
    for idx, (_, trade) in enumerate(df.iterrows()):
        original = results[idx]
        swapped = swapped_results[idx]
        
        report_lines.append(f"TRADE #{idx + 1}")
        report_lines.append("-" * 50)
        report_lines.append(f"Entry Time: {trade['entry_time']}")
        report_lines.append(f"Direction: {trade['direction']}")
        report_lines.append(f"Entry Price: ${trade['entry_price']:,.2f}")
        report_lines.append(f"Stop Loss: ${trade['stop_loss']:,.2f}")
        report_lines.append(f"Take Profit: ${trade['take_profit']:,.2f}")
        report_lines.append("")
        
        # Original strategy result
        report_lines.append("ORIGINAL STRATEGY (Current SL/TP):")
        if original['status'] == 'ANALYZED':
            report_lines.append(f"  Result: {original['message']}")
            if original['winner'] == 'TAKE_PROFIT':
                report_lines.append(f"  ✅ WIN - TP hit at {original['tp_hit_time']}")
                original_wins += 1
            elif original['winner'] == 'STOP_LOSS':
                report_lines.append(f"  ❌ LOSS - SL hit at {original['sl_hit_time']}")
            else:
                report_lines.append(f"  ⚪ UNCLEAR - {original['outcome']}")
        else:
            report_lines.append(f"  ❓ UNKNOWN - {original.get('message', 'No analysis')}")
        
        report_lines.append("")
        
        # Swapped strategy result
        report_lines.append("SWAPPED STRATEGY (If SL became TP and TP became SL):")
        if swapped['status'] == 'ANALYZED':
            report_lines.append(f"  Result: {swapped['message']}")
            if swapped['winner'] == 'TAKE_PROFIT':
                report_lines.append(f"  ✅ WIN - Swapped TP hit at {swapped['tp_hit_time']}")
                swapped_wins += 1
            elif swapped['winner'] == 'STOP_LOSS':
                report_lines.append(f"  ❌ LOSS - Swapped SL hit at {swapped['sl_hit_time']}")
            else:
                report_lines.append(f"  ⚪ UNCLEAR - {swapped['outcome']}")
        else:
            report_lines.append(f"  ❓ UNKNOWN - {swapped.get('message', 'No analysis')}")
        
        report_lines.append("")
        
        # Recommendation
        if original['status'] == 'ANALYZED' and swapped['status'] == 'ANALYZED':
            if original['winner'] == 'STOP_LOSS' and swapped['winner'] == 'TAKE_PROFIT':
                report_lines.append("💡 RECOMMENDATION: Swapping SL/TP would have turned this LOSS into a WIN!")
            elif original['winner'] == 'TAKE_PROFIT' and swapped['winner'] == 'STOP_LOSS':
                report_lines.append("⚠️  WARNING: Swapping SL/TP would have turned this WIN into a LOSS!")
            elif original['winner'] == swapped['winner']:
                report_lines.append("🔄 NEUTRAL: Swapping SL/TP would not change the outcome")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("")
    
    # Summary
    total_analyzed = len([r for r in results if r['status'] == 'ANALYZED'])
    original_win_rate = (original_wins / total_analyzed * 100) if total_analyzed > 0 else 0
    swapped_win_rate = (swapped_wins / total_analyzed * 100) if total_analyzed > 0 else 0
    
    report_lines.append("SUMMARY RESULTS:")
    report_lines.append("=" * 40)
    report_lines.append(f"Total Trades Analyzed: {total_analyzed}")
    report_lines.append(f"")
    report_lines.append(f"ORIGINAL STRATEGY:")
    report_lines.append(f"  Wins: {original_wins}")
    report_lines.append(f"  Win Rate: {original_win_rate:.1f}%")
    report_lines.append(f"")
    report_lines.append(f"SWAPPED STRATEGY (SL↔TP):")
    report_lines.append(f"  Wins: {swapped_wins}")
    report_lines.append(f"  Win Rate: {swapped_win_rate:.1f}%")
    report_lines.append(f"")
    
    if swapped_win_rate > original_win_rate:
        improvement = swapped_win_rate - original_win_rate
        report_lines.append(f"🚀 SWAPPING SL/TP IMPROVES WIN RATE BY {improvement:.1f} PERCENTAGE POINTS!")
    elif original_win_rate > swapped_win_rate:
        decrease = original_win_rate - swapped_win_rate
        report_lines.append(f"⚠️  SWAPPING SL/TP DECREASES WIN RATE BY {decrease:.1f} PERCENTAGE POINTS!")
    else:
        report_lines.append("🔄 SWAPPING SL/TP HAS NO NET EFFECT ON WIN RATE")
    
    # Save detailed report
    report_content = "\n".join(report_lines)
    with open('proper_sl_tp_verification_report.txt', 'w') as f:
        f.write(report_content)
    
    print(f"\n✅ Analysis complete!")
    print(f"Original strategy win rate: {original_win_rate:.1f}%")
    print(f"Swapped strategy win rate: {swapped_win_rate:.1f}%")
    print(f"Detailed report saved to: proper_sl_tp_verification_report.txt")
    
    # Create summary CSV
    summary_data = []
    for idx, (_, trade) in enumerate(df.iterrows()):
        original = results[idx]
        swapped = swapped_results[idx]
        
        summary_data.append({
            'trade_idx': idx + 1,
            'entry_time': trade['entry_time'],
            'direction': trade['direction'],
            'entry_price': trade['entry_price'],
            'stop_loss': trade['stop_loss'],
            'take_profit': trade['take_profit'],
            'original_outcome': original.get('winner', 'UNKNOWN'),
            'original_message': original.get('message', 'No analysis'),
            'swapped_outcome': swapped.get('winner', 'UNKNOWN'),
            'swapped_message': swapped.get('message', 'No analysis'),
            'swapping_beneficial': (
                'YES' if (original.get('winner') == 'STOP_LOSS' and swapped.get('winner') == 'TAKE_PROFIT')
                else 'NO' if (original.get('winner') == 'TAKE_PROFIT' and swapped.get('winner') == 'STOP_LOSS')
                else 'NEUTRAL'
            )
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('sl_tp_verification_summary.csv', index=False)
    print(f"Summary CSV saved to: sl_tp_verification_summary.csv")

if __name__ == "__main__":
    analyze_all_trades()