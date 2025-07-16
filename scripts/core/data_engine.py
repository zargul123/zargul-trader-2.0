import os
import time
import random
import pandas as pd
import numpy as np
import requests
import json
import yfinance as yf
from datetime import datetime, timedelta
from scripts.config import (
    TWELVEDATA_API_KEY,
    TWELVEDATA_CONFIG,
    TWELVEDATA_MAPPING,
    TECHNICAL_INDICATORS,
    STRATEGIES,
    TRAINING_CONFIG,
    ASSETS
)

class DataMaster:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        })
        self.last_used_source = None
        self.sequence_length = STRATEGIES.get('main', {}).get('sequence_length', 60)
        self.asset_parameters = {
            "BTC-USD": {"base_price": 95000, "base_volume": 50000},
            "ETH-USD": {"base_price": 1800, "base_volume": 300000},
            "SOL-USD": {"base_price": 150, "base_volume": 1000000},
        }

    def _convert_timeframe(self, tf):
        timeframe_map = {'1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min', '1h': '1h', '4h': '4h', '1d': '1day'}
        return timeframe_map.get(tf, '1h')

    def _twelvedata_request(self, symbol, timeframe, params=None):
        mapped_symbol = TWELVEDATA_MAPPING.get(symbol, symbol)
        base_params = {
            'symbol': mapped_symbol,
            'interval': self._convert_timeframe(timeframe),
            'apikey': TWELVEDATA_API_KEY,
            'outputsize': 5000,
            'format': 'JSON'
        }
        if params:
            base_params.update(params)
        
        print(f"🔍 TwelveData API Request: {mapped_symbol} ({timeframe})")
        print(f"   URL: {TWELVEDATA_CONFIG['base_url']}/time_series")
        print(f"   Params: {base_params}")
        
        try:
            response = self.session.get(f"{TWELVEDATA_CONFIG['base_url']}/time_series", params=base_params, timeout=TWELVEDATA_CONFIG['timeout'])
            print(f"   Status Code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ HTTP Error {response.status_code}: {response.text}")
                return None
                
            data = response.json()
            print(f"   Response Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            
            # Log the actual response for debugging
            if 'message' in data:
                print(f"   API Message: {data['message']}")
            if 'status' in data:
                print(f"   API Status: {data['status']}")
            if 'code' in data:
                print(f"   API Code: {data['code']}")
                
            return data
            
        except requests.exceptions.Timeout:
            print(f"⚠️ TwelveData request timeout for {symbol}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"⚠️ TwelveData request failed for {symbol}: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"⚠️ TwelveData JSON decode error for {symbol}: {e}")
            print(f"   Raw response: {response.text[:500]}...")
            return None

    def _parse_twelvedata_response(self, data, symbol):
        if not data:
            print(f"❌ No data returned from TwelveData for {symbol}")
            return pd.DataFrame()
            
        # Check for API errors
        if 'message' in data and 'error' in data.get('message', '').lower():
            print(f"❌ TwelveData API Error for {symbol}: {data['message']}")
            return pd.DataFrame()
            
        if 'code' in data and data['code'] != 200:
            print(f"❌ TwelveData API Code {data['code']} for {symbol}: {data.get('message', 'Unknown error')}")
            return pd.DataFrame()
            
        # TwelveData doesn't always return status='ok', check for values instead
        if 'values' not in data:
            print(f"❌ No 'values' key in TwelveData response for {symbol}")
            print(f"   Available keys: {list(data.keys())}")
            return pd.DataFrame()
            
        values = data['values']
        if not values or len(values) == 0:
            print(f"❌ Empty values array from TwelveData for {symbol}")
            return pd.DataFrame()
            
        print(f"✅ TwelveData returned {len(values)} data points for {symbol}")
        
        try:
            df = pd.DataFrame(values).iloc[::-1]  # Reverse to get chronological order
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime').rename(columns=str.lower)
            
            # Convert price columns to numeric
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    print(f"⚠️ Missing column '{col}' in TwelveData response for {symbol}")
                    
            # Handle volume - TwelveData sometimes doesn't include volume for crypto
            if 'volume' in df.columns:
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            else:
                print(f"⚠️ No volume data from TwelveData for {symbol}, setting to 0")
                df['volume'] = 0
                
            # Remove any rows with NaN values in essential columns
            essential_cols = ['open', 'high', 'low', 'close']
            df = df.dropna(subset=essential_cols)
            
            print(f"✅ Parsed {len(df)} valid candles for {symbol}")
            return df
            
        except Exception as e:
            print(f"❌ Error parsing TwelveData response for {symbol}: {e}")
            print(f"   Raw data sample: {values[0] if values else 'No data'}")
            return pd.DataFrame()

    def _yahoo_fallback(self, symbol, timeframe):
        period_map = {'1h': '60d', '4h': '120d', '15m': '30d'}
        # Fix: Explicitly set auto_adjust=False to avoid warnings and ensure consistency
        df = yf.download(
            symbol, 
            period=period_map.get(timeframe, '30d'), 
            interval=self._convert_timeframe(timeframe), 
            progress=False,
            auto_adjust=False
        )
        if not df.empty:
            df = df.rename(columns=str.lower)
            # Fix: Remove timezone info to match TwelveData format
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
        return df

    def _generate_synthetic_data(self, symbol):
        params = self.asset_parameters.get(symbol, {"base_price": 100, "base_volume": 100000})
        dates = pd.date_range(end=pd.Timestamp.now(), periods=self.sequence_length * 3, freq='1h')
        returns = np.random.normal(0, 0.002, len(dates))
        for i in range(1, len(returns)):
            returns[i] = 0.7 * returns[i-1] + 0.3 * returns[i]
        close_prices = params["base_price"] * (1 + returns.cumsum())
        volume = np.random.lognormal(mean=np.log(params["base_volume"]), sigma=0.5, size=len(dates))
        df = pd.DataFrame({
            'open': close_prices * np.random.uniform(0.998, 1.002, len(dates)),
            'high': close_prices * np.random.uniform(1.001, 1.005, len(dates)),
            'low': close_prices * np.random.uniform(0.995, 0.999, len(dates)),
            'close': close_prices,
            'volume': volume
        }, index=dates)
        self.last_used_source = 'synthetic'
        return df

    def _add_technical_indicators(self, df):
        if df.empty:
            return df
            
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        try:
            df['rsi'] = self._calculate_rsi(df)
            df = self._calculate_macd(df)
            df = self._calculate_bollinger_bands(df)
            df = self._calculate_cmf(df)
            df = self._calculate_obv(df)
            df = self._calculate_vol_spike(df)
            df = self._calculate_vwap(df)
            df = self._calculate_emas(df)
            df = self._calculate_atr(df)
            df = self._calculate_stochastic(df)
            df = self._calculate_adx(df)
            df = self._calculate_volume_ma(df)
            
            # Only drop rows where ALL indicator columns are NaN
            # Keep the original OHLCV data intact
            essential_cols = ['open', 'high', 'low', 'close', 'volume']
            before_count = len(df)
            df = df.dropna(subset=essential_cols, how='any')
            after_count = len(df)
            
            if before_count != after_count:
                print(f"⚠️ Dropped {before_count - after_count} rows with missing OHLCV data")

            # --- ROBUSTNESS FIX: Fill NaN values from indicators ---
            # Forward-fill first to propagate last valid values
            # Backward-fill to handle NaNs at the start of the series
            indicator_cols = [indi for indi in TECHNICAL_INDICATORS if indi in df.columns]
            df[indicator_cols] = df[indicator_cols].fillna(method='ffill').fillna(method='bfill')
            
            # As a final safety net, fill any remaining NaNs (if any) with 0
            df[indicator_cols] = df[indicator_cols].fillna(0)
            
            print(f"✅ Cleaned NaN values from technical indicators.")
            # ---------------------------------------------------------
                
            return df
            
        except Exception as e:
            print(f"❌ Error in technical indicators calculation: {e}")
            # Return original data without indicators if calculation fails
            return df

    def _calculate_rsi(self, df, window=14):
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window, min_periods=1).mean()
        loss = -delta.where(delta < 0, 0).rolling(window, min_periods=1).mean()
        rs = gain / loss.replace(0, 0.001)
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, df, fast=12, slow=26, signal=9):
        df['macd'] = df['close'].ewm(span=fast).mean() - df['close'].ewm(span=slow).mean()
        df['macd_signal'] = df['macd'].ewm(span=signal).mean()
        return df

    def _calculate_bollinger_bands(self, df, window=20):
        sma = df['close'].rolling(window).mean()
        std = df['close'].rolling(window).std()
        df['bollinger_upper'] = sma + (std * 2)
        df['bollinger_lower'] = sma - (std * 2)
        return df

    def _calculate_cmf(self, df, window=20):
        mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, 0.001) * df['volume']
        cmf = mfv.rolling(window).sum() / df['volume'].rolling(window).sum().replace(0, 0.001)
        df['cmf'] = cmf
        return df

    def _calculate_obv(self, df):
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv'] = obv
        return df

    def _calculate_vol_spike(self, df, window=10):
        df['vol_spike'] = df['volume'] / df['volume'].rolling(window).mean().replace(0, 0.001)
        return df

    def _calculate_vwap(self, df):
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        return df

    def _calculate_emas(self, df):
        """Calculate Exponential Moving Averages"""
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['ema_200'] = df['close'].ewm(span=200).mean()
        return df

    def _calculate_atr(self, df, window=14):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = true_range.rolling(window).mean()
        return df

    def _calculate_stochastic(self, df, k_window=14, d_window=3):
        """Calculate Stochastic Oscillator"""
        low_min = df['low'].rolling(k_window).min()
        high_max = df['high'].rolling(k_window).max()
        df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min).replace(0, 0.001))
        df['stoch_d'] = df['stoch_k'].rolling(d_window).mean()
        return df

    def _calculate_adx(self, df, window=14):
        """Calculate Average Directional Index"""
        high_diff = df['high'].diff()
        low_diff = df['low'].diff()

        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

        # Calculate True Range for ADX
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))

        # Smooth the values
        atr = pd.Series(true_range).rolling(window).mean()
        plus_di = 100 * (pd.Series(plus_dm).rolling(window).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window).mean() / atr)

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 0.001)
        df['adx'] = dx.rolling(window).mean()
        return df

    def _calculate_volume_ma(self, df, window=20):
        """Calculate Volume Moving Average"""
        df['volume_ma'] = df['volume'].rolling(window).mean()
        return df

    def get_data(self, symbol, timeframe='4h', limit=None):
        print(f"\n📊 Fetching {timeframe} data for {symbol}...")
        
        # Try TwelveData first
        data = self._twelvedata_request(symbol, timeframe)
        df = self._parse_twelvedata_response(data, symbol)
        
        if not df.empty:
            self.last_used_source = 'TwelveData'
            print(f"✅ TwelveData successful for {symbol}")
        else:
            print(f"📉 TwelveData failed for {symbol}. Falling back to Yahoo Finance.")
            df = self._yahoo_fallback(symbol, timeframe)
            if not df.empty:
                self.last_used_source = 'Yahoo'
                print(f"✅ Yahoo Finance successful for {symbol}")
            else:
                print(f"❌ CRITICAL: All real data sources failed for {symbol}. No data available.")
                return None

        # Add technical indicators
        try:
            df = self._add_technical_indicators(df)
            print(f"✅ Technical indicators added for {symbol}")
        except Exception as e:
            print(f"⚠️ Error adding technical indicators for {symbol}: {e}")
            return None

        # Apply limit if specified
        if limit and not df.empty:
            original_len = len(df)
            df = df.tail(int(limit))
            print(f"📊 Limited data from {original_len} to {len(df)} candles")

        print(f"✅ Final result: {len(df)} {timeframe} candles for {symbol} from {self.last_used_source}")
        return df

    def get_training_data(self, symbol, days=None):
        if days is None:
            days = TRAINING_CONFIG['training_days']

        df = self.get_data(symbol, '1h')

        if df is None or df.empty:
            print("="*60)
            print(f"⚠️ WARNING: Real data failed for {symbol}. Training will proceed with FAKE data.")
            print("         Models trained on this data should NOT be used for live trading.")
            print("="*60)
            df = self._generate_synthetic_data(symbol)

        df = df.last(f'{days}D')
        return self._add_technical_indicators(df)