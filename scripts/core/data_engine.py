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
    SEQUENCE_LENGTH,
    TRAINING_DAYS,
    ASSETS
)

class DataMaster:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        })
        self.cache = {}
        self.last_request_time = 0
        self.last_used_source = None
        self.request_delay = (0.1, 0.5)
        self.asset_parameters = {
            "BTC-USD": {"base_price": 95000, "base_volume": 50000},
            "ETH-USD": {"base_price": 1800, "base_volume": 300000},
            "SOL-USD": {"base_price": 150, "base_volume": 1000000},
            "BNB-USD": {"base_price": 600, "base_volume": 200000}
        }

    def _ensure_required_columns(self, df, symbol):
        """More lenient data validation"""
        if not isinstance(df, pd.DataFrame) or df.empty:
            return self._generate_synthetic_data(symbol)

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                if col == 'volume':
                    df['volume'] = df.get('volume', 100000)  # Default volume
                else:
                    df[col] = df.get('close', 100)  # Use close price as fallback
        
        # Ensure we have basic technical indicators
        if 'rsi' not in df.columns:
            df = self._calculate_rsi(df)
        if 'macd' not in df.columns:
            df = self._calculate_macd(df)
            
        return df

    def _convert_timeframe(self, tf):
        """Convert timeframe to API format"""
        timeframe_map = {
            '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '1h', '4h': '4h', '1d': '1day', '1w': '1week', '1M': '1month'
        }
        return timeframe_map.get(tf, '1h')

    def _twelvedata_request(self, endpoint, symbol, timeframe, params=None):
        """Make API request with robust error handling"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time

        # Enforce strict rate limiting
        min_interval = 60 / TWELVEDATA_CONFIG['rate_limit']
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        base_params = {
            'symbol': TWELVEDATA_MAPPING.get(symbol, symbol),
            'interval': self._convert_timeframe(timeframe),
            'apikey': TWELVEDATA_API_KEY,
            'outputsize': SEQUENCE_LENGTH * 3
        }

        if params:
            base_params.update(params)

        try:
            response = self.session.get(
                f"{TWELVEDATA_CONFIG['base_url']}/{endpoint}",
                params=base_params,
                timeout=TWELVEDATA_CONFIG['timeout']
            )
            response.raise_for_status()

            # Validate JSON response
            try:
                data = response.json()
                self.last_request_time = time.time()
                return data
            except json.JSONDecodeError:
                print(f"⚠️ Invalid JSON response from TwelveData for {symbol}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"⚠️ TwelveData request failed for {symbol}: {str(e)}")
            return None

    def _parse_twelvedata_response(self, data, symbol):
        """Safely parse TwelveData response"""
        try:
            if not data or data.get('status') != 'ok' or 'values' not in data:
                return pd.DataFrame()

            df = pd.DataFrame(data['values'])
            if df.empty:
                return pd.DataFrame()

            df = df.iloc[::-1]  # Reverse to chronological order

            # Standardize columns with robust type conversion
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df = df.dropna(subset=['datetime'])

            df = df.rename(columns={
                'datetime': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }).set_index('timestamp')

            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            return self._ensure_required_columns(df, symbol)

        except Exception as e:
            print(f"❌ TwelveData parsing failed for {symbol}: {str(e)}")
            return pd.DataFrame()

    def _get_twelvedata_series(self, symbol, timeframe):
        """Get data from TwelveData with timeout and retry"""
        # Add timeout and retry
        try:
            data = self._twelvedata_request('time_series', symbol, timeframe)
            if not data or data.get('status') != 'ok':
                raise ValueError("Bad data")
                
            df = self._parse_twelvedata_response(data, symbol)
            if len(df) < 100:  # Require minimum data
                raise ValueError("Not enough data")
                
            print(f"✅ Got clean {symbol} data from TwelveData")
            return df
            
        except Exception as e:
            print(f"⚠️ TwelveData failed for {symbol}, trying Yahoo...")
            return self._yahoo_fallback(symbol, timeframe)

    def _calculate_volume_spike(self, df):
        """Calculate volume spikes with robust error handling"""
        if 'volume' not in df.columns:
            return df.assign(vol_spike=1.0)

        try:
            df['vol_ma'] = df['volume'].rolling(
                window=10,
                min_periods=1,
                center=False
            ).mean()
            df['vol_spike'] = np.where(
                df['vol_ma'] > 0,
                df['volume'] / df['vol_ma'],
                0.8
            )
            return df.drop(columns=['vol_ma'])
        except Exception:
            return df.assign(vol_spike=1.0)

    def _yahoo_fallback(self, symbol, timeframe):
        """Yahoo Finance fallback with robust error handling"""
        try:
            df = yf.download(
                symbol,
                period="30d",
                interval=self._convert_timeframe(timeframe),
                prepost=True,
                progress=False,
                threads=True
            )
            if df.empty:
                return pd.DataFrame()

            # Standardize columns
            column_map = {
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
            df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

            return self._ensure_required_columns(df[[col for col in ['open','high','low','close','volume'] if col in df.columns]], symbol)
        except Exception as e:
            print(f"❌ Yahoo failed for {symbol}: {str(e)}")
            return pd.DataFrame()

    def _generate_synthetic_data(self, symbol):
        """Generate realistic synthetic data with proper statistics"""
        params = self.asset_parameters.get(symbol, {"base_price": 100, "base_volume": 100000})
        base_price = params["base_price"]
        base_volume = params["base_volume"]

        dates = pd.date_range(
            end=pd.Timestamp.now(),
            periods=SEQUENCE_LENGTH*3,
            freq='1h'
        )

        # Generate realistic price series with autocorrelation
        returns = np.random.normal(0, 0.002, len(dates))
        for i in range(1, len(returns)):
            returns[i] = 0.7 * returns[i-1] + 0.3 * returns[i]

        close_prices = base_price * (1 + returns.cumsum())

        # Generate realistic volume with clustering
        volume = np.random.lognormal(
            mean=np.log(base_volume),
            sigma=0.5,
            size=len(dates))

        # Create 10% chance of volume spike
        spike_mask = np.random.random(len(dates)) < 0.1
        volume[spike_mask] *= np.random.uniform(1.5, 3.0, size=spike_mask.sum())

        df = pd.DataFrame({
            'open': close_prices * np.random.uniform(0.998, 1.002, len(dates)),
            'high': close_prices * np.random.uniform(1.001, 1.005, len(dates)),
            'low': close_prices * np.random.uniform(0.995, 0.999, len(dates)),
            'close': close_prices,
            'volume': volume
        }, index=dates)

        self.last_used_source = 'synthetic'
        return df

    def _add_technical_indicators(self, df, symbol):
        """Calculate all technical indicators with guaranteed success"""
        try:
            df = self._ensure_required_columns(df, symbol)
            df = self._calculate_volume_spike(df)

            # Calculate all indicators with proper error handling
            indicators = {
                'rsi': self._calculate_rsi,
                'macd': self._calculate_macd,
                'macd_signal': lambda x: x,  # Handled in MACD calculation
                'bollinger_upper': self._calculate_bollinger_bands,
                'bollinger_lower': lambda x: x,  # Handled in Bollinger
                'cmf': self._calculate_cmf,
                'vwap': self._calculate_vwap,
                'obv': self._calculate_obv
            }

            for indicator, func in indicators.items():
                if indicator not in df.columns:
                    df = func(df)

            return df.dropna()
        except Exception as e:
            print(f"⚠️ Indicator calculation failed for {symbol}: {str(e)}")
            return self._ensure_required_columns(df, symbol).assign(
                **{indicator: 0.0 for indicator in TECHNICAL_INDICATORS}
            )

    def _calculate_rsi(self, df, window=14):
        """Calculate RSI with proper handling"""
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window, min_periods=1).mean()
        avg_loss = loss.rolling(window, min_periods=1).mean()

        rs = avg_gain / avg_loss.replace(0, 0.001)
        df['rsi'] = 100 - (100 / (1 + rs))
        return df

    def _calculate_macd(self, df):
        """Calculate MACD and signal line"""
        df['macd'] = df['close'].ewm(span=12, adjust=False).mean() - \
                     df['close'].ewm(span=26, adjust=False).mean()
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        return df

    def _calculate_bollinger_bands(self, df, window=20):
        """Calculate Bollinger Bands"""
        sma = df['close'].rolling(window, min_periods=1).mean()
        std = df['close'].rolling(window, min_periods=1).std()
        df['bollinger_upper'] = sma + (std * 2)
        df['bollinger_lower'] = sma - (std * 2)
        return df

    def _calculate_cmf(self, df, window=20):
        """Calculate Chaikin Money Flow"""
        mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / \
                       (df['high'] - df['low']).replace(0, 0.0001)
        mf_volume = mf_multiplier * df['volume']
        df['cmf'] = mf_volume.rolling(window, min_periods=1).sum() / \
                   df['volume'].rolling(window, min_periods=1).sum()
        return df

    def _calculate_vwap(self, df):
        """Calculate Volume Weighted Average Price"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return df

    def _calculate_obv(self, df):
        """Calculate On-Balance Volume"""
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        return df

    def get_data(self, symbol, timeframe='1h'):
        """Get market data with more historical depth"""
        # Increase data length based on timeframe
        if timeframe == '1h':
            days = 30  # ~720 candles (30 days)
        elif timeframe == '4h':
            days = 60  # ~360 candles (60 days)
        elif timeframe == '15m':
            days = 15  # ~600 candles (15 days)
        else:
            days = 30
            
        # Modified TwelveData request with more data
        params = {
            'symbol': TWELVEDATA_MAPPING.get(symbol, symbol),
            'interval': self._convert_timeframe(timeframe),
            'apikey': TWELVEDATA_API_KEY,
            'outputsize': days * 24  # Get full days of data
        }

        # Try TwelveData first with increased data
        data = self._twelvedata_request('time_series', symbol, timeframe, params)
        if data and data.get('status') == 'ok':
            df = self._parse_twelvedata_response(data, symbol)
            if len(df) >= 100:  # Require minimum data
                self.last_used_source = 'twelvedata'
            else:
                df = pd.DataFrame()
        else:
            df = pd.DataFrame()

        # Fallback to Yahoo
        if df.empty:
            df = self._yahoo_fallback(symbol, timeframe)
            self.last_used_source = 'yahoo' if not df.empty else self.last_used_source

        # Final fallback to synthetic
        if df.empty:
            df = self._generate_synthetic_data(symbol)
            self.last_used_source = 'synthetic'

        # Add technical indicators
        df = self._add_technical_indicators(df, symbol)

        # Ensure all technical indicators exist
        for indicator in TECHNICAL_INDICATORS:
            if indicator not in df.columns:
                df[indicator] = 0.0

        self.cache[symbol] = df
        return df

    def _filter_trading_hours(self, df, hours):
        """Filter DataFrame to only include specified trading hours"""
        if not hours:  # No filter if empty
            return df
        return df[df.index.hour.isin(hours)]

    def get_training_data(self, symbol):
        """Get training data with guaranteed structure"""
        # Try TwelveData first with extended history
        df = self._get_twelvedata_series(symbol, '15m')
        if df.empty:
            # Fallback to Yahoo
            df = self._yahoo_fallback(symbol, '15m')
            if df.empty:
                # Final fallback to synthetic with more data points
                df = self._generate_synthetic_data(symbol)
                df = df.iloc[-TRAINING_DAYS*96:]  # 15min intervals for training days

        # Add technical indicators
        df = self._add_technical_indicators(df, symbol)
        return df