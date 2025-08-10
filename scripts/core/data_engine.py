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
    ASSETS,
    LUNARCRUSH_CONFIG
)
from scripts.core.social_analyzer import SocialAnalyzer, get_ttl_hash

import pandas_ta as ta

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
        self.social_analyzer = SocialAnalyzer(
            api_key="gtqpx5sjq3t11j982d0vow0c0rme5s7ms5iy6jnzm",
            base_url=LUNARCRUSH_CONFIG['api_url']
        )

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
        
        try:
            response = self.session.get(f"{TWELVEDATA_CONFIG['base_url']}/time_series", params=base_params, timeout=TWELVEDATA_CONFIG['timeout'])
            response.raise_for_status()
            data = response.json()
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"⚠️ TwelveData request failed for {symbol}: {e}")
            return None
        except json.JSONDecodeError:
            print(f"⚠️ TwelveData JSON decode error for {symbol}")
            return None

    def _parse_twelvedata_response(self, data, symbol):
        if not data or 'values' not in data:
            return pd.DataFrame()
            
        try:
            df = pd.DataFrame(data['values']).iloc[::-1]
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime').rename(columns=str.lower)
            
            # Remove duplicate indices that cause reindexing errors
            df = df[~df.index.duplicated(keep='first')]
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
            return df
            
        except Exception as e:
            print(f"❌ Error parsing TwelveData response for {symbol}: {e}")
            return pd.DataFrame()

    def _yahoo_fallback(self, symbol, timeframe):
        period_map = {'1h': '60d', '4h': '120d', '15m': '30d'}
        df = yf.download(
            symbol, 
            period=period_map.get(timeframe, '30d'), 
            interval=self._convert_timeframe(timeframe), 
            progress=False,
            auto_adjust=False
        )
        if not df.empty:
            df = df.rename(columns=str.lower)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            # Remove duplicate indices that cause reindexing errors
            df = df[~df.index.duplicated(keep='first')]
        return df

    def _add_technical_indicators(self, df):
        """
        Calculate technical indicators using the pandas-ta library.
        This is more robust and cleaner than manual calculations.
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Ensure 'volume' column exists before calling pandas-ta, as it's required by some indicators.
        if 'volume' not in df.columns:
            print("⚠️ 'volume' column not found in data. Creating a placeholder column of zeros.")
            df['volume'] = 0
        
        # Use the built-in strategy from pandas-ta to get a wide range of indicators
        custom_strategy = ta.Strategy(
            name="ZargulStrategy",
            description="A comprehensive set of technical indicators for the Zargul trading bot.",
            ta=[
                {"kind": "rsi"},
                {"kind": "macd"},
                {"kind": "bbands", "length": 20, "std": 2},
                {"kind": "cmf"},
                {"kind": "obv"},
                {"kind": "vwap"},
                {"kind": "ema", "length": 20},
                {"kind": "ema", "length": 50},
                {"kind": "ema", "length": 200},
                {"kind": "atr"},
                {"kind": "stoch"},
                {"kind": "adx"},
                {"kind": "sma", "length": 20, "close": df['volume'], "col_names": ("volume_ma",)},
            ]
        )
        
        df.ta.strategy(custom_strategy)

        # --- Rename columns to match the old format BEFORE calculating elite features ---
        df.rename(columns={
            'RSI_14': 'rsi',
            'MACD_12_26_9': 'macd',
            'MACDs_12_26_9': 'macd_signal',
            'BBU_20_2.0': 'bollinger_upper',
            'BBL_20_2.0': 'bollinger_lower',
            'CMF_20': 'cmf',
            'OBV': 'obv',
            'VWAP_D': 'vwap',
            'EMA_20': 'ema_20',
            'EMA_50': 'ema_50',
            'EMA_200': 'ema_200',
            'ATRr_14': 'atr',
            'STOCHk_14_3_3': 'stoch_k',
            'STOCHd_14_3_3': 'stoch_d',
            'ADX_14': 'adx'
        }, inplace=True)

        # --- ELITE NORMALIZED & DERIVATIVE FEATURES ---
        # Ensure volume_ma is not zero to avoid division by zero errors
        df['vol_spike'] = df['volume'] / df['volume_ma'].replace(0, 1)
        df['atr_norm'] = (df['atr'] / df['close']) * 100
        df['macd_norm'] = (df['macd'] / df['close']) * 100
        df['bollinger_width'] = ((df['bollinger_upper'] - df['bollinger_lower']) / df['close']) * 100
        df['ema_spread'] = ((df['ema_20'] - df['ema_200']) / df['close']) * 100
        df['pct_change'] = df['close'].pct_change() * 100
        df['log_return'] = np.log(df['close'] / df['close'].shift())

        # --- FINAL CLEANUP ---
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        df.fillna(0, inplace=True)
        
        print(f"✅ Engineered features using pandas-ta and cleaned NaN values.")
        return df

    def get_data(self, symbol, timeframe='4h', limit=None):
        print(f"\n📊 Fetching {timeframe} data for {symbol}...")
        
        data = self._twelvedata_request(symbol, timeframe)
        df = self._parse_twelvedata_response(data, symbol)
        
        if df.empty:
            print(f"📉 TwelveData failed for {symbol}. Falling back to Yahoo Finance.")
            df = self._yahoo_fallback(symbol, timeframe)
            if df.empty:
                print(f"❌ CRITICAL: All real data sources failed for {symbol}. No data available.")
                return None
        
        df = self._add_technical_indicators(df)

        # --- Inject Social Metrics from LunarCrush ---
        try:
            social_metrics = self.social_analyzer.get_social_metrics(symbol, ttl_hash=get_ttl_hash())
            if social_metrics:
                print(f"✅ Got social metrics for {symbol} from LunarCrush.")
                for metric in LUNARCRUSH_CONFIG['metrics_to_use']:
                    if metric in social_metrics:
                        df[f'lc_{metric}'] = social_metrics[metric]
                
                social_cols = [f'lc_{m}' for m in LUNARCRUSH_CONFIG['metrics_to_use'] if f'lc_{m}' in df.columns]
                df[social_cols] = df[social_cols].ffill().bfill().fillna(0)
                print(f"   - Injected and cleaned features: {social_cols}")
        except Exception as e:
            print(f"⚠️ Could not get or inject social metrics for {symbol}: {e}")
            for metric in LUNARCRUSH_CONFIG['metrics_to_use']:
                df[f'lc_{metric}'] = 0

        if limit and not df.empty:
            df = df.tail(int(limit))

        print(f"✅ Final result: {len(df)} {timeframe} candles for {symbol}")
        return df

    def get_training_data(self, symbol, timeframe='1h', days=None):
        if days is None:
            days = TRAINING_CONFIG['training_days']

        df = self.get_data(symbol, timeframe)

        if df is None or df.empty:
            print("="*60)
            print(f"⚠️ WARNING: Real data failed for {symbol}. Training will proceed with FAKE data.")
            print("="*60)
            # This part needs to be implemented if you want synthetic data generation
            return pd.DataFrame()
        
        df = df.last(f'{days}D')
        return self._add_technical_indicators(df)