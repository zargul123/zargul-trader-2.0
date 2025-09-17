#!/usr/bin/env python3
"""
Individual Trade Market Verification
For each backtest trade, fetch real market data and show what actually happened
"""

import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import os

def get_market_data_for_period(start_time, end_time, retries=3):
    """Try to get market data for a specific time period from free APIs"""
    
    # Convert to timestamps
    start_ts = int(start_time.timestamp())
    end_ts = int(end_time.timestamp())
    
    # Try CoinGecko API first (free, no key required)
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
        params = {
            'vs_currency': 'usd',
            'from': start_ts,
            'to': end_ts
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'prices' in data and data['prices']:
                prices = data['prices']
                # Convert to DataFrame
                df = pd.DataFrame(prices, columns=['timestamp', 'price'])
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df, "CoinGecko"
    except Exception as e:
        print(f"CoinGecko failed: {e}")
    
    # Try CoinCap API as backup
    try:
        url = "https://api.coincap.io/v2/assets/bitcoin/history"
        params = {
            'interval': 'h1',
            'start': start_ts * 1000,  # milliseconds
            'end': end_ts * 1000
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and data['data']:
                df = pd.DataFrame(data['data'])
                df['datetime'] = pd.to_datetime(df['time'])
                df['price'] = pd.to_numeric(df['priceUsd'])
                return df[['datetime', 'price']], "CoinCap"
    except Exception as e:
        print(f"CoinCap failed: {e}")
    
    return None, "No source available"

def check_historical_data_file():
    """Check if we have historical data file available"""
    hist_file = "../historical_data_for_analysis.csv"
    if os.path.exists(hist_file):
        try:
            df = pd.read_csv(hist_file)
            print(f"Found historical data file with {len(df)} records")
            
            # Try to identify datetime column
            datetime_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            if datetime_cols:
                df['datetime'] = pd.to_datetime(df[datetime_cols[0]])
                
                # Try to identify price column
                price_cols = [col for col in df.columns if 'close' in col.lower() or 'price' in col.lower()]
                if price_cols:
                    df['price'] = pd.to_numeric(df[price_cols[0]], errors='coerce')
                    return df[['datetime', 'price']], "Historical File"
        except Exception as e:
            print(f"Error reading historical file: {e}")
    
    return None, "Historical file not available"

def verify_trade_against_market(trade, market_data, source):
    """Verify a single trade against market data"""
    entry_time = pd.to_datetime(trade['entry_time'])
    exit_time = pd.to_datetime(trade['exit_time'])
    
    # Find closest market data points
    if market_data is None or market_data.empty:
        return {
            'status': 'NO_DATA',
            'source': source,
            'message': 'No market data available for verification'
        }
    
    # Convert market data datetime to UTC if needed
    market_data['datetime'] = pd.to_datetime(market_data['datetime']).dt.tz_localize(None)
    
    # Find entry price in market data
    entry_diff = abs(market_data['datetime'] - entry_time)
    entry_idx = entry_diff.idxmin()
    entry_market_time = market_data.loc[entry_idx, 'datetime']
    entry_market_price = market_data.loc[entry_idx, 'price']
    
    # Find exit price in market data
    exit_diff = abs(market_data['datetime'] - exit_time)
    exit_idx = exit_diff.idxmin()
    exit_market_time = market_data.loc[exit_idx, 'datetime']
    exit_market_price = market_data.loc[exit_idx, 'price']
    
    # Calculate differences
    entry_time_diff = abs((entry_market_time - entry_time).total_seconds() / 60)  # minutes
    exit_time_diff = abs((exit_market_time - exit_time).total_seconds() / 60)  # minutes
    
    entry_price_diff = abs(trade['entry_price'] - entry_market_price) / trade['entry_price'] * 100
    exit_price_diff = abs(trade['exit_price'] - exit_market_price) / trade['exit_price'] * 100
    
    # Calculate what actually happened in market
    actual_price_change = exit_market_price - entry_market_price
    actual_pnl_pct = (actual_price_change / entry_market_price) * 100
    
    # For SHORT trades, flip the PnL calculation
    if trade['direction'] == 'SHORT':
        actual_pnl_pct = -actual_pnl_pct
    
    return {
        'status': 'VERIFIED',
        'source': source,
        'entry_market_time': entry_market_time,
        'entry_market_price': entry_market_price,
        'entry_time_diff_min': entry_time_diff,
        'entry_price_diff_pct': entry_price_diff,
        'exit_market_time': exit_market_time,
        'exit_market_price': exit_market_price,
        'exit_time_diff_min': exit_time_diff,
        'exit_price_diff_pct': exit_price_diff,
        'actual_price_change': actual_price_change,
        'actual_pnl_pct': actual_pnl_pct,
        'backtest_pnl': trade.get('pnl_correct_pct', trade.get('pnl_log_pct', 0)),
        'pnl_difference': actual_pnl_pct - trade.get('pnl_correct_pct', trade.get('pnl_log_pct', 0))
    }

def create_detailed_verification_report(df, verification_results):
    """Create detailed report showing what actually happened for each trade"""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("DETAILED TRADE-BY-TRADE MARKET VERIFICATION")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    successful_verifications = 0
    total_trades = len(df)
    
    for idx, (_, trade) in enumerate(df.iterrows()):
        verification = verification_results[idx]
        
        report_lines.append(f"TRADE #{idx + 1}")
        report_lines.append("-" * 40)
        report_lines.append(f"Entry Time: {trade['entry_time']}")
        report_lines.append(f"Exit Time:  {trade['exit_time']}")
        report_lines.append(f"Direction:  {trade['direction']}")
        report_lines.append("")
        
        if verification['status'] == 'VERIFIED':
            successful_verifications += 1
            
            report_lines.append("BACKTEST DATA:")
            report_lines.append(f"  Entry Price: ${trade['entry_price']:,.2f}")
            report_lines.append(f"  Exit Price:  ${trade['exit_price']:,.2f}")
            report_lines.append(f"  PnL:         {trade.get('pnl_correct_pct', trade.get('pnl_log_pct', 0)):.3f}%")
            report_lines.append("")
            
            report_lines.append(f"ACTUAL MARKET DATA ({verification['source']}):")
            report_lines.append(f"  Entry Price: ${verification['entry_market_price']:,.2f}")
            report_lines.append(f"  Exit Price:  ${verification['exit_market_price']:,.2f}")
            report_lines.append(f"  Actual PnL:  {verification['actual_pnl_pct']:.3f}%")
            report_lines.append("")
            
            report_lines.append("DIFFERENCES:")
            report_lines.append(f"  Entry Price Diff: {verification['entry_price_diff_pct']:.3f}%")
            report_lines.append(f"  Exit Price Diff:  {verification['exit_price_diff_pct']:.3f}%")
            report_lines.append(f"  PnL Difference:   {verification['pnl_difference']:.3f}%")
            report_lines.append("")
            
            # Flag significant discrepancies
            if verification['entry_price_diff_pct'] > 1.0:
                report_lines.append("⚠️  LARGE ENTRY PRICE DISCREPANCY!")
            if verification['exit_price_diff_pct'] > 1.0:
                report_lines.append("⚠️  LARGE EXIT PRICE DISCREPANCY!")
            if abs(verification['pnl_difference']) > 0.5:
                report_lines.append("⚠️  SIGNIFICANT PnL DIFFERENCE!")
            
        else:
            report_lines.append(f"❌ VERIFICATION FAILED: {verification['message']}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("")
    
    # Summary
    verification_rate = (successful_verifications / total_trades) * 100
    report_lines.append(f"SUMMARY:")
    report_lines.append(f"Total Trades: {total_trades}")
    report_lines.append(f"Successfully Verified: {successful_verifications}")
    report_lines.append(f"Verification Rate: {verification_rate:.1f}%")
    
    return "\n".join(report_lines)

def main():
    print("INDIVIDUAL TRADE MARKET VERIFICATION")
    print("=" * 50)
    
    # Load backtest data
    backtest_file = 'backtest_aug20_sep10.csv'
    if not os.path.exists(backtest_file):
        print(f"ERROR: Backtest file {backtest_file} not found!")
        return
    
    df = pd.read_csv(backtest_file)
    print(f"Loaded {len(df)} trades for verification")
    
    # Try to get market data
    print("\nAttempting to fetch market data...")
    
    # First try historical data file
    market_data, source = check_historical_data_file()
    
    # If no historical data, try APIs for a sample period
    if market_data is None:
        print("No historical data file found, trying live APIs for sample...")
        start_time = pd.to_datetime(df['entry_time'].min())
        end_time = pd.to_datetime(df['exit_time'].max())
        market_data, source = get_market_data_for_period(start_time, end_time)
    
    if market_data is None:
        print("❌ Could not obtain market data from any source!")
        print("Available options:")
        print("1. Place a historical_data_for_analysis.csv file in the parent directory")
        print("2. Wait for API rate limits to reset")
        return
    
    print(f"✅ Market data obtained from: {source}")
    print(f"Market data range: {market_data['datetime'].min()} to {market_data['datetime'].max()}")
    print(f"Market data points: {len(market_data)}")
    
    # Verify each trade
    print(f"\nVerifying {len(df)} trades against market data...")
    verification_results = []
    
    for idx, (_, trade) in enumerate(df.iterrows()):
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(df)} trades...")
        
        verification = verify_trade_against_market(trade, market_data, source)
        verification_results.append(verification)
        
        # Small delay to be nice to APIs
        time.sleep(0.1)
    
    # Create detailed report
    print("\nGenerating detailed verification report...")
    report = create_detailed_verification_report(df, verification_results)
    
    # Save report
    report_file = 'detailed_trade_verification_report.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"✅ Detailed report saved to: {report_file}")
    
    # Create summary CSV
    summary_data = []
    for idx, verification in enumerate(verification_results):
        trade = df.iloc[idx]
        summary_data.append({
            'trade_idx': idx + 1,
            'entry_time': trade['entry_time'],
            'backtest_entry_price': trade['entry_price'],
            'market_entry_price': verification.get('entry_market_price', 'N/A'),
            'backtest_exit_price': trade['exit_price'],
            'market_exit_price': verification.get('exit_market_price', 'N/A'),
            'backtest_pnl': trade.get('pnl_correct_pct', trade.get('pnl_log_pct', 0)),
            'actual_market_pnl': verification.get('actual_pnl_pct', 'N/A'),
            'pnl_difference': verification.get('pnl_difference', 'N/A'),
            'verification_status': verification['status'],
            'data_source': verification['source']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = 'trade_market_comparison_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    
    print(f"✅ Summary CSV saved to: {summary_file}")
    print(f"\nFiles generated:")
    print(f"- {report_file} (detailed text report)")
    print(f"- {summary_file} (summary CSV)")

if __name__ == "__main__":
    main()