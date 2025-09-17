#!/usr/bin/env python3
"""
Backtest Pattern Analysis & Verification
Analyzes patterns in backtest data to identify potential accuracy issues
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

def analyze_price_movements(df):
    """Analyze price movement patterns for anomalies"""
    print("\n" + "="*60)
    print("PRICE MOVEMENT PATTERN ANALYSIS")
    print("="*60)
    
    # Calculate price movement statistics
    df['entry_exit_diff'] = df['exit_price'] - df['entry_price']
    df['movement_pct'] = (df['entry_exit_diff'] / df['entry_price']) * 100
    
    # For SHORT trades, positive movement_pct means price went down (profit)
    # Negative movement_pct means price went up (loss)
    
    print(f"Price Movement Statistics:")
    print(f"  Average movement: {df['movement_pct'].mean():.4f}%")
    print(f"  Std deviation: {df['movement_pct'].std():.4f}%")
    print(f"  Min movement: {df['movement_pct'].min():.4f}%")
    print(f"  Max movement: {df['movement_pct'].max():.4f}%")
    
    # Check for suspicious patterns
    suspicious_patterns = []
    
    # 1. Check if all movements are in same direction (unusual for crypto)
    positive_moves = (df['movement_pct'] > 0).sum()
    negative_moves = (df['movement_pct'] < 0).sum()
    
    print(f"\nMovement Direction Analysis:")
    print(f"  Positive movements (price down for SHORT): {positive_moves}")
    print(f"  Negative movements (price up for SHORT): {negative_moves}")
    
    if negative_moves == 0:
        suspicious_patterns.append("ALL movements profitable - highly unusual for crypto trading")
    
    # 2. Check for repeated identical prices
    entry_duplicates = df['entry_price'].duplicated().sum()
    exit_duplicates = df['exit_price'].duplicated().sum()
    
    print(f"\nPrice Duplication Check:")
    print(f"  Duplicate entry prices: {entry_duplicates}")
    print(f"  Duplicate exit prices: {exit_duplicates}")
    
    if entry_duplicates > len(df) * 0.1:  # More than 10% duplicates
        suspicious_patterns.append(f"High number of duplicate entry prices: {entry_duplicates}")
    
    # 3. Check for unrealistic price movements
    large_moves = df[abs(df['movement_pct']) > 5]  # >5% movement in 1 hour
    print(f"\nLarge Movements (>5% in 1 hour): {len(large_moves)}")
    
    if len(large_moves) > len(df) * 0.05:  # More than 5% of trades have >5% moves
        suspicious_patterns.append(f"Unusually high number of large price movements: {len(large_moves)}")
    
    return suspicious_patterns

def analyze_temporal_patterns(df):
    """Analyze temporal patterns for anomalies"""
    print("\n" + "="*60)
    print("TEMPORAL PATTERN ANALYSIS")
    print("="*60)
    
    suspicious_patterns = []
    
    # Check trade timing
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df['trade_duration'] = df['exit_time'] - df['entry_time']
    
    print(f"Trade Duration Analysis:")
    unique_durations = df['trade_duration'].unique()
    print(f"  Unique trade durations: {len(unique_durations)}")
    print(f"  Most common duration: {df['trade_duration'].mode().iloc[0]}")
    
    # Check if all trades are exactly 1 hour (suspicious if ALL are exactly same)
    one_hour_trades = (df['trade_duration'] == timedelta(hours=1)).sum()
    print(f"  Exactly 1-hour trades: {one_hour_trades}/{len(df)}")
    
    if one_hour_trades == len(df):
        suspicious_patterns.append("ALL trades are exactly 1 hour - may indicate simplified backtesting")
    
    # Check for gaps in trading
    df_sorted = df.sort_values('entry_time')
    time_gaps = df_sorted['entry_time'].diff()
    large_gaps = time_gaps[time_gaps > timedelta(hours=12)]
    
    print(f"\nTrading Gap Analysis:")
    print(f"  Gaps > 12 hours: {len(large_gaps)}")
    
    if len(large_gaps) > 0:
        print("  Large gaps found at:")
        for gap_time in large_gaps.index[:5]:  # Show first 5
            print(f"    {df_sorted.loc[gap_time, 'entry_time']}")
    
    # Check for weekend trading patterns
    df['entry_weekday'] = df['entry_time'].dt.day_name()
    weekend_trades = df[df['entry_weekday'].isin(['Saturday', 'Sunday'])]
    
    print(f"\nWeekend Trading:")
    print(f"  Weekend trades: {len(weekend_trades)}/{len(df)}")
    print(f"  Weekday distribution:")
    print(df['entry_weekday'].value_counts())
    
    return suspicious_patterns

def analyze_sl_tp_ratios(df):
    """Analyze stop loss and take profit ratios"""
    print("\n" + "="*60)
    print("STOP LOSS / TAKE PROFIT ANALYSIS")
    print("="*60)
    
    suspicious_patterns = []
    
    # Calculate SL/TP distances from entry
    df['sl_distance'] = abs(df['stop_loss'] - df['entry_price']) / df['entry_price'] * 100
    df['tp_distance'] = abs(df['take_profit'] - df['entry_price']) / df['entry_price'] * 100
    df['risk_reward_ratio'] = df['tp_distance'] / df['sl_distance']
    
    print(f"SL/TP Distance Analysis:")
    print(f"  Average SL distance: {df['sl_distance'].mean():.3f}%")
    print(f"  Average TP distance: {df['tp_distance'].mean():.3f}%")
    print(f"  Average Risk/Reward ratio: {df['risk_reward_ratio'].mean():.3f}")
    
    # Check for unrealistic SL/TP ratios
    high_rr = df[df['risk_reward_ratio'] > 10]
    low_rr = df[df['risk_reward_ratio'] < 0.1]
    
    print(f"\nRisk/Reward Extremes:")
    print(f"  Very high R/R (>10): {len(high_rr)}")
    print(f"  Very low R/R (<0.1): {len(low_rr)}")
    
    if len(high_rr) > len(df) * 0.1:
        suspicious_patterns.append(f"High number of extreme risk/reward ratios: {len(high_rr)}")
    
    # Check for SL/TP inversion (for SHORT trades)
    # For SHORT: SL should be > entry, TP should be < entry
    incorrect_sl = df[df['stop_loss'] < df['entry_price']]
    incorrect_tp = df[df['take_profit'] > df['entry_price']]
    
    print(f"\nSL/TP Direction Check (for SHORT trades):")
    print(f"  SL below entry (incorrect): {len(incorrect_sl)}")
    print(f"  TP above entry (incorrect): {len(incorrect_tp)}")
    
    if len(incorrect_sl) > 0 or len(incorrect_tp) > 0:
        suspicious_patterns.append("SL/TP directions appear inverted for SHORT trades")
    
    return suspicious_patterns

def analyze_pnl_consistency(df):
    """Analyze PnL consistency and patterns"""
    print("\n" + "="*60)
    print("PnL CONSISTENCY ANALYSIS")
    print("="*60)
    
    suspicious_patterns = []
    
    # Check both logged PnL and calculated PnL
    if 'pnl_log_pct' in df.columns and 'pnl_correct_pct' in df.columns:
        pnl_diff = abs(df['pnl_log_pct'] - df['pnl_correct_pct'])
        significant_diff = pnl_diff[pnl_diff > 0.01]  # >1% difference
        
        print(f"PnL Calculation Consistency:")
        print(f"  Trades with significant PnL differences: {len(significant_diff)}")
        print(f"  Average PnL difference: {pnl_diff.mean():.4f}%")
        print(f"  Max PnL difference: {pnl_diff.max():.4f}%")
        
        if len(significant_diff) > 0:
            suspicious_patterns.append(f"PnL calculation inconsistencies found: {len(significant_diff)} trades")
    
    # Analyze PnL distribution
    pnl_col = 'pnl_correct_pct' if 'pnl_correct_pct' in df.columns else 'pnl_log_pct'
    
    print(f"\nPnL Distribution Analysis:")
    print(f"  Mean PnL: {df[pnl_col].mean():.4f}%")
    print(f"  Std Dev: {df[pnl_col].std():.4f}%")
    print(f"  Min PnL: {df[pnl_col].min():.4f}%")
    print(f"  Max PnL: {df[pnl_col].max():.4f}%")
    
    # Check for suspiciously consistent profits
    positive_trades = (df[pnl_col] > 0).sum()
    win_rate = positive_trades / len(df) * 100
    
    print(f"  Win rate: {win_rate:.2f}%")
    
    if win_rate > 95:
        suspicious_patterns.append(f"Extremely high win rate: {win_rate:.2f}% - may indicate overfitting or data issues")
    
    # Check for unrealistic PnL clustering
    pnl_rounded = df[pnl_col].round(2)
    most_common_pnl = pnl_rounded.mode().iloc[0]
    most_common_count = (pnl_rounded == most_common_pnl).sum()
    
    print(f"  Most common PnL: {most_common_pnl:.2f}% ({most_common_count} trades)")
    
    if most_common_count > len(df) * 0.1:
        suspicious_patterns.append(f"High clustering around single PnL value: {most_common_pnl:.2f}%")
    
    return suspicious_patterns

def create_analysis_plots(df):
    """Create visualizations for pattern analysis"""
    print("\n" + "="*60)
    print("CREATING ANALYSIS PLOTS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Backtest Pattern Analysis', fontsize=16)
    
    # 1. PnL distribution
    pnl_col = 'pnl_correct_pct' if 'pnl_correct_pct' in df.columns else 'pnl_log_pct'
    axes[0,0].hist(df[pnl_col], bins=30, alpha=0.7, edgecolor='black')
    axes[0,0].set_title('PnL Distribution')
    axes[0,0].set_xlabel('PnL (%)')
    axes[0,0].set_ylabel('Frequency')
    
    # 2. Price movement over time
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    axes[0,1].plot(df['entry_time'], df['entry_price'], 'b-', alpha=0.7, label='Entry Price')
    axes[0,1].plot(df['entry_time'], df['exit_price'], 'r-', alpha=0.7, label='Exit Price')
    axes[0,1].set_title('Price Movement Over Time')
    axes[0,1].set_xlabel('Date')
    axes[0,1].set_ylabel('Price (USD)')
    axes[0,1].legend()
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Risk/Reward ratio distribution
    if 'risk_reward_ratio' in df.columns:
        axes[0,2].hist(df['risk_reward_ratio'], bins=30, alpha=0.7, edgecolor='black')
        axes[0,2].set_title('Risk/Reward Ratio Distribution')
        axes[0,2].set_xlabel('Risk/Reward Ratio')
        axes[0,2].set_ylabel('Frequency')
    
    # 4. PnL vs Confidence
    if 'confidence' in df.columns:
        axes[1,0].scatter(df['confidence'], df[pnl_col], alpha=0.6)
        axes[1,0].set_title('PnL vs Confidence')
        axes[1,0].set_xlabel('Confidence')
        axes[1,0].set_ylabel('PnL (%)')
    
    # 5. Trading frequency by hour
    df['entry_hour'] = df['entry_time'].dt.hour
    hour_counts = df['entry_hour'].value_counts().sort_index()
    axes[1,1].bar(hour_counts.index, hour_counts.values, alpha=0.7)
    axes[1,1].set_title('Trading Frequency by Hour')
    axes[1,1].set_xlabel('Hour (UTC)')
    axes[1,1].set_ylabel('Number of Trades')
    
    # 6. Cumulative PnL
    df_sorted = df.sort_values('entry_time')
    cumulative_pnl = df_sorted[pnl_col].cumsum()
    axes[1,2].plot(df_sorted['entry_time'], cumulative_pnl)
    axes[1,2].set_title('Cumulative PnL Over Time')
    axes[1,2].set_xlabel('Date')
    axes[1,2].set_ylabel('Cumulative PnL (%)')
    axes[1,2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('backtest_pattern_analysis.png', dpi=300, bbox_inches='tight')
    print("Analysis plots saved to: backtest_pattern_analysis.png")
    
    return 'backtest_pattern_analysis.png'

def main():
    print("BACKTEST PATTERN VERIFICATION")
    print("="*50)
    
    # Load backtest data
    backtest_file = 'backtest_aug20_sep10.csv'
    if not os.path.exists(backtest_file):
        print(f"ERROR: Backtest file {backtest_file} not found!")
        return
    
    try:
        df = pd.read_csv(backtest_file)
        print(f"Loaded {len(df)} trades from backtest file")
    except Exception as e:
        print(f"Error loading backtest file: {e}")
        return
    
    # Run all analyses
    all_suspicious_patterns = []
    
    all_suspicious_patterns.extend(analyze_price_movements(df))
    all_suspicious_patterns.extend(analyze_temporal_patterns(df))
    all_suspicious_patterns.extend(analyze_sl_tp_ratios(df))
    all_suspicious_patterns.extend(analyze_pnl_consistency(df))
    
    # Create visualizations
    plot_file = create_analysis_plots(df)
    
    # Summary report
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    if all_suspicious_patterns:
        print(f"⚠️  FOUND {len(all_suspicious_patterns)} SUSPICIOUS PATTERNS:")
        for i, pattern in enumerate(all_suspicious_patterns, 1):
            print(f"  {i}. {pattern}")
        
        # Save suspicious patterns
        with open('suspicious_patterns_report.txt', 'w') as f:
            f.write("BACKTEST VERIFICATION - SUSPICIOUS PATTERNS\n")
            f.write("=" * 50 + "\n\n")
            for i, pattern in enumerate(all_suspicious_patterns, 1):
                f.write(f"{i}. {pattern}\n")
        
        print(f"\nDetailed report saved to: suspicious_patterns_report.txt")
    else:
        print("✅ No obvious suspicious patterns detected in backtest data")
    
    print(f"\nFiles generated:")
    print(f"- {plot_file}")
    if all_suspicious_patterns:
        print(f"- suspicious_patterns_report.txt")

if __name__ == "__main__":
    main()