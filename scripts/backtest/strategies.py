import numpy as np
import pandas as pd  # ← Add this line
from scripts.config import (
    LONG_THRESHOLD, SHORT_THRESHOLD, MIN_CONFIDENCE,
    SWING_THRESHOLD, SCALP_THRESHOLD
)

class BaseStrategy:
    def __init__(self):
        self.long_threshold = 0.5
        self.short_threshold = 0.5
        self.min_confidence = 0.65
        
    def set_thresholds(self, long=None, short=None, confidence=None):
        """Set strategy-specific thresholds"""
        if long is not None:
            self.long_threshold = long
        if short is not None:
            self.short_threshold = short
        if confidence is not None:
            self.min_confidence = confidence
    
    def get_signal(self, df):
        """Should return 1 (long), -1 (short), or 0 (no trade)"""
        raise NotImplementedError

    def get_exit_signal(self, df, current_position):
        """Should return True if should exit, False otherwise"""
        raise NotImplementedError

class MainStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.open_trades = []
        
    def get_signal(self, df):
        if len(df) < 20: 
            return 0
        
        # Get current values
        current = df.iloc[-1]
        rsi = current['rsi']
        macd_bullish = current['macd'] > current['macd_signal']
        above_vwap = current['close'] > current['vwap']
        
        # Long conditions:
        # - RSI not overbought (<60)
        # - MACD bullish crossover
        # - Price above VWAP (institutional buying)
        if all([rsi < 60, macd_bullish, above_vwap]):
            return 1
            
        # Short conditions:
        # - RSI overbought (>70)
        # - Price below Bollinger Upper Band
        # - High volume (1.5x average)
        if (current['rsi'] > 70 and 
            current['close'] < current['bollinger_upper'] and
            df['volume'].iloc[-1] > df['volume'].mean() * 1.5):
            return -1  # Short signal
            
        return 0

    def _get_ai_signal(self, df):
        """Replace this with your actual AI model call"""
        try:
            # Mock - use your real model here
            last_60 = df.iloc[-60:][['open','high','low','close','volume'] + TECHNICAL_INDICATORS]
            return 1 if df['close'].iloc[-1] > df['close'].iloc[-2] else -1
        except:
            return None

    def _get_ta_signal(self, df):
        """Original technical analysis signal logic"""
        if len(df) < 50:  # Need more data
            return 0
        
        # Strong filters
        ema20 = df['close'].ewm(span=20).mean()
        price_above_ema = df['close'].iloc[-1] > ema20.iloc[-1]
        volume_ok = df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1] * 1.2
        rsi_ok = 30 < df['rsi'].iloc[-1] < 70
        macd_ok = df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]
        recent_trend = df['close'].iloc[-1] > df['close'].iloc[-5]  # Price higher than 5 periods ago
        trend_strength = df['close'].iloc[-1] > df['close'].rolling(10).mean().iloc[-1]  # 10-period trend filter
        
        # Long signal conditions
        if price_above_ema and volume_ok and rsi_ok and macd_ok and recent_trend and trend_strength:
            if df['close'].iloc[-1] > df['close'].iloc[-2] * (1 + LONG_THRESHOLD/100):
                return 1  # Long
        
        # Short signal conditions (opposite logic)
        price_below_ema = df['close'].iloc[-1] < ema20.iloc[-1]
        macd_bearish = df['macd'].iloc[-1] < df['macd_signal'].iloc[-1]
        recent_downtrend = df['close'].iloc[-1] < df['close'].iloc[-5]
        trend_weakness = df['close'].iloc[-1] < df['close'].rolling(10).mean().iloc[-1]
        
        if price_below_ema and volume_ok and rsi_ok and macd_bearish and recent_downtrend and trend_weakness:
            if df['close'].iloc[-1] < df['close'].iloc[-2] * (1 - LONG_THRESHOLD/100):
                return -1  # Short
        
        return 0
        
    def get_exit_signal(self, df, current_position):
        if len(self.open_trades) == 0:
            return False
            
        entry = self.open_trades[0]
        current = df.iloc[-1]
        hours_held = (current.name - entry['entry_time']).total_seconds() / 3600
        
        # Dynamic exits
        if current_position == 1:  # Long
            # Take profit at 1.5% or after 6 hours
            take_profit = entry['entry_price'] * 1.015
            if current['close'] >= take_profit or hours_held >= 6:
                return True
            # Stop loss at 1% or if RSI > 70
            if current['close'] <= entry['entry_price'] * 0.99 or df['rsi'].iloc[-1] > 70:
                return True
                
        elif current_position == -1:  # Short
            # Take profit at 1% or after 4 hours
            take_profit = entry['entry_price'] * 0.99
            if current['close'] <= take_profit or hours_held >= 4:
                return True
            # Stop loss at 1.5% or if RSI < 30
            if current['close'] >= entry['entry_price'] * 1.015 or df['rsi'].iloc[-1] < 30:
                return True
        
        return False
    
    def get_current_entry_price(self, df, current_position):
        """Get entry price for current position"""
        if len(self.open_trades) > 0:
            return self.open_trades[0]['entry_price']
        return df['close'].iloc[-1]  # Fallback to current price

class SwingStrategy(MainStrategy):
    def get_signal(self, df):
        if len(df) < 50:  # Need more data for swing
            return 0

        signal = super().get_signal(df)
        if signal == 0:
            return 0

        # Additional swing filters
        current = df.iloc[-1]
        if current['close'] > df['close'].rolling(20).mean().iloc[-1]:  # Above 20MA
            return 1 if signal == 1 else 0
        elif current['close'] < df['close'].rolling(20).mean().iloc[-1]:  # Below 20MA
            return -1 if signal == -1 else 0
        return 0

    def get_exit_signal(self, df, current_position):
        # Hold longer than main strategy
        if super().get_exit_signal(df, current_position):
            return True

        # Additional exit conditions
        current = df.iloc[-1]
        if current_position == 1 and current['close'] < df['close'].rolling(10).mean().iloc[-1]:
            return True
        elif current_position == -1 and current['close'] > df['close'].rolling(10).mean().iloc[-1]:
            return True

        return False

class ScalpStrategy(MainStrategy):
    def get_signal(self, df):
        if len(df) < 5:  # Need less data for scalp
            return 0

        current = df.iloc[-1]
        prev = df.iloc[-2]

        # More sensitive thresholds
        long_thresh = SCALP_THRESHOLD / 100
        short_thresh = SCALP_THRESHOLD / 100

        # Long signal
        if (current['close'] > prev['close'] * (1 + long_thresh) and
            current['macd'] > current['macd_signal']):
            return 1

        # Short signal
        elif (current['close'] < prev['close'] * (1 - short_thresh) and
              current['macd'] < current['macd_signal']):
            return -1

        return 0

    def get_exit_signal(self, df, current_position):
        # Quick exits for scalp
        current = df.iloc[-1]
        if current_position == 1:
            return (current['close'] < df['close'].iloc[-2] or  # Price drops
                    current['macd'] < current['macd_signal'])  # MACD crosses
        elif current_position == -1:
            return (current['close'] > df['close'].iloc[-2] or  # Price rises
                    current['macd'] > current['macd_signal'])  # MACD crosses
        return False