#!/usr/bin/env python3
"""
Fresh Market Data Fetcher and Backtest Verifier
Fetches fresh market data and compares with backtest results
"""

import pandas as pd
import yfinance as yf
import requests
import json
from datetime import datetime, timedelta
import time
import os

def fetch_fresh_data_yfinance(symbol="BTC-USD", start_date="2025-08-20", end_date="2025-09-11"):
    """Fetch fresh data from Yahoo Finance"""
    print(f"Fetching fresh data from Yahoo Finance for {symbol}...")
    
    try:
        ticker = yf.Ticker(symbol)
        # Fetch hourly data
        data = ticker.history(start=start_date, end=end_date, interval="1h")
        
        if data.empty:
            print("No data received from Yahoo Finance")
            return None
            
        # Reset index to get datetime as column
        data = data.reset_index()
        data.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        
        print(f"Fetched {len(data)} hourly candles from Yahoo Finance")
        return data
        
    except Exception as e:
        print(f"Error fetching from Yahoo Finance: {e}")
        return None

def fetch_fresh_data_coinbase(symbol="BTC-USD", start_date="2025-08-20", end_date="2025-09-11"):
    """Fetch fresh data from Coinbase API (public, no auth required)"""
    print(f"Fetching fresh data from Coinbase for {symbol}...")
    
    try:
        # Convert dates to ISO format
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        base_url = "https://api.exchange.coinbase.com"
        endpoint = f"/products/{symbol}/candles"
        
        # Coinbase uses different granularity (3600 = 1 hour)
        params = {
            'start': start_dt.isoformat(),
            'end': end_dt.isoformat(),
            'granularity': 3600  # 1 hour
        }
        
        response = requests.get(base_url + endpoint, params=params, timeout=30)
        
        if response.status_code != 200:
            print(f"Coinbase API error: {response.status_code}")
            return None
        
        data = response.json()
        
        if not data:
            print("No data received from Coinbase")
            return None
        
        # Convert to DataFrame
        # Coinbase format: [timestamp, low, high, open, close, volume]
        df = pd.DataFrame(data, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Reorder columns to match standard format
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
        
        print(f"Fetched {len(df)} hourly candles from Coinbase")
        return df
        
    except Exception as e:
        print(f"Error fetching from Coinbase: {e}")
        return None

def fetch_fresh_data_binance(symbol="BTCUSDT"):
    """Fetch fresh data from Binance API (public, no auth required)"""
    print(f"Fetching fresh data from Binance for {symbol}...")
    
    try:
        base_url = "https://api.binance.com"
        endpoint = "/api/v3/klines"
        
        # Convert to timestamps (milliseconds)
        start_dt = datetime(2025, 8, 20)
        end_dt = datetime(2025, 9, 11)
        
        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)
        
        params = {
            'symbol': symbol,
            'interval': '1h',
            'startTime': start_ts,
            'endTime': end_ts,
            'limit': 1000
        }
        
        response = requests.get(base_url + endpoint, params=params, timeout=30)
        
        if response.status_code != 200:
            print(f"Binance API error: {response.status_code}")
            return None
        
        data = response.json()
        
        if not data:
            print("No data received from Binance")
            return None
        
        # Convert to DataFrame
        # Binance format: [timestamp, open, high, low, close, volume, close_time, ...]
        df = pd.DataFrame(data)
        df = df.iloc[:, :6]  # Only take first 6 columns
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Convert timestamp and prices
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
        df = df.sort_values('datetime').reset_index(drop=True)
        
        print(f"Fetched {len(df)} hourly candles from Binance")
        return df
        
    except Exception as e:
        print(f"Error fetching from Binance: {e}")
        return None

def fetch_fresh_data_twelvedata(symbol="BTC/USD", start_date="2025-08-20 00:00:00", end_date="2025-09-10 23:59:59"):
    """Fetch fresh data from TwelveData API (free tier)"""
    print(f"Fetching fresh data from TwelveData for {symbol}...")
    
    # TwelveData free API endpoint (no key required for basic access)
    url = "https://api.twelvedata.com/time_series"
    
    params = {
        'symbol': symbol,
        'interval': '1h',
        'start_date': start_date,
        'end_date': end_date,
        'format': 'JSON'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code != 200:
            print(f"TwelveData API error: {response.status_code}")
            return None
            
        data = response.json()
        
        if 'values' not in data:
            print(f"No values in TwelveData response: {data}")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Convert price columns to float
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        print(f"Fetched {len(df)} hourly candles from TwelveData")
        return df
        
    except Exception as e:
        print(f"Error fetching from TwelveData: {e}")
        return None

def compare_backtest_with_fresh_data(backtest_file, fresh_data):
    """Compare backtest results with fresh market data"""
    print("\n" + "="*60)
    print("COMPARING BACKTEST WITH FRESH MARKET DATA")
    print("="*60)
    
    # Load backtest data
    try:
        backtest_df = pd.read_csv(backtest_file)
        print(f"Loaded {len(backtest_df)} trades from backtest file")
    except Exception as e:
        print(f"Error loading backtest file: {e}")
        return
    
    if fresh_data is None or fresh_data.empty:
        print("No fresh data available for comparison")
        return
    
    # Convert datetime columns
    backtest_df['entry_time'] = pd.to_datetime(backtest_df['entry_time'])
    backtest_df['exit_time'] = pd.to_datetime(backtest_df['exit_time'])
    fresh_data['datetime'] = pd.to_datetime(fresh_data['datetime'])
    
    # Set timezone to UTC for comparison
    fresh_data['datetime'] = fresh_data['datetime'].dt.tz_localize('UTC')
    
    print(f"Fresh data range: {fresh_data['datetime'].min()} to {fresh_data['datetime'].max()}")
    print(f"Backtest range: {backtest_df['entry_time'].min()} to {backtest_df['exit_time'].max()}")
    
    discrepancies = []
    verified_trades = 0
    
    for idx, trade in backtest_df.iterrows():
        entry_time = trade['entry_time']
        exit_time = trade['exit_time']
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        direction = trade['direction']
        
        # Find corresponding candles in fresh data
        entry_candle = fresh_data[fresh_data['datetime'] == entry_time]
        exit_candle = fresh_data[fresh_data['datetime'] == exit_time]
        
        if entry_candle.empty:
            # Try to find closest candle within 5 minutes
            time_diff = abs(fresh_data['datetime'] - entry_time)
            closest_idx = time_diff.idxmin()
            if time_diff.iloc[closest_idx] <= timedelta(minutes=5):
                entry_candle = fresh_data.iloc[[closest_idx]]
            else:
                discrepancies.append({
                    'trade_idx': idx,
                    'issue': 'entry_time_not_found',
                    'entry_time': entry_time,
                    'closest_time': fresh_data.iloc[closest_idx]['datetime'] if not fresh_data.empty else None
                })
                continue
        
        if exit_candle.empty:
            # Try to find closest candle within 5 minutes
            time_diff = abs(fresh_data['datetime'] - exit_time)
            closest_idx = time_diff.idxmin()
            if time_diff.iloc[closest_idx] <= timedelta(minutes=5):
                exit_candle = fresh_data.iloc[[closest_idx]]
            else:
                discrepancies.append({
                    'trade_idx': idx,
                    'issue': 'exit_time_not_found',
                    'exit_time': exit_time,
                    'closest_time': fresh_data.iloc[closest_idx]['datetime'] if not fresh_data.empty else None
                })
                continue
        
        # Compare prices
        entry_candle_row = entry_candle.iloc[0]
        exit_candle_row = exit_candle.iloc[0]
        
        # Check if backtest prices are within the candle ranges
        entry_within_range = (entry_candle_row['low'] <= entry_price <= entry_candle_row['high'])
        exit_within_range = (exit_candle_row['low'] <= exit_price <= exit_candle_row['high'])
        
        if not entry_within_range:
            discrepancies.append({
                'trade_idx': idx,
                'issue': 'entry_price_out_of_range',
                'backtest_price': entry_price,
                'candle_low': entry_candle_row['low'],
                'candle_high': entry_candle_row['high'],
                'entry_time': entry_time
            })
        
        if not exit_within_range:
            discrepancies.append({
                'trade_idx': idx,
                'issue': 'exit_price_out_of_range',
                'backtest_price': exit_price,
                'candle_low': exit_candle_row['low'],
                'candle_high': exit_candle_row['high'],
                'exit_time': exit_time
            })
        
        if entry_within_range and exit_within_range:
            verified_trades += 1
    
    # Print results
    print(f"\nVERIFICATION RESULTS:")
    print(f"Total trades analyzed: {len(backtest_df)}")
    print(f"Trades verified against fresh data: {verified_trades}")
    print(f"Discrepancies found: {len(discrepancies)}")
    print(f"Verification rate: {verified_trades/len(backtest_df)*100:.2f}%")
    
    if discrepancies:
        print(f"\nDISCREPANCIES BREAKDOWN:")
        issue_counts = {}
        for disc in discrepancies:
            issue_type = disc['issue']
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        for issue_type, count in issue_counts.items():
            print(f"  {issue_type}: {count} trades")
        
        # Save detailed discrepancies
        disc_df = pd.DataFrame(discrepancies)
        disc_file = 'fresh_data_discrepancies.csv'
        disc_df.to_csv(disc_file, index=False)
        print(f"\nDetailed discrepancies saved to: {disc_file}")
        
        # Show first few discrepancies
        print(f"\nFIRST 5 DISCREPANCIES:")
        for i, disc in enumerate(discrepancies[:5]):
            print(f"  {i+1}. Trade {disc['trade_idx']}: {disc['issue']}")
    
    return {
        'total_trades': len(backtest_df),
        'verified_trades': verified_trades,
        'discrepancies': discrepancies,
        'verification_rate': verified_trades/len(backtest_df)*100
    }

def main():
    print("FRESH MARKET DATA VERIFICATION")
    print("="*50)
    
    # Try to fetch fresh data from multiple sources
    fresh_data = None
    data_source = "unknown"
    
    # Try Binance first (often most reliable)
    fresh_data = fetch_fresh_data_binance()
    if fresh_data is not None and not fresh_data.empty:
        data_source = "Binance"
    
    # If Binance fails, try Coinbase
    if fresh_data is None or fresh_data.empty:
        fresh_data = fetch_fresh_data_coinbase()
        if fresh_data is not None and not fresh_data.empty:
            data_source = "Coinbase"
    
    # If Coinbase fails, try Yahoo Finance
    if fresh_data is None or fresh_data.empty:
        fresh_data = fetch_fresh_data_yfinance()
        if fresh_data is not None and not fresh_data.empty:
            data_source = "Yahoo Finance"
    
    if fresh_data is None or fresh_data.empty:
        print("ERROR: Could not fetch fresh market data from any source!")
        return
    
    # Save fresh data
    fresh_data_file = 'fresh_market_data_btc_aug20_sep10.csv'
    fresh_data.to_csv(fresh_data_file, index=False)
    print(f"Fresh data saved to: {fresh_data_file}")
    
    # Compare with backtest
    backtest_file = 'backtest_aug20_sep10.csv'
    if os.path.exists(backtest_file):
        results = compare_backtest_with_fresh_data(backtest_file, fresh_data)
        
        # Save summary
        summary = {
            'fresh_data_source': data_source,
            'fresh_data_candles': len(fresh_data),
            'fresh_data_range': f"{fresh_data['datetime'].min()} to {fresh_data['datetime'].max()}",
            'verification_summary': results
        }
        
        with open('fresh_data_verification_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nVerification summary saved to: fresh_data_verification_summary.json")
    else:
        print(f"ERROR: Backtest file {backtest_file} not found!")

if __name__ == "__main__":
    main()