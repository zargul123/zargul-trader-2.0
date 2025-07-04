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
        if len(df) < 10:  # Reduced minimum data requirement
            return 0
            
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # More lenient volume check
        volume_ok = current['volume'] > df['volume'].rolling(10, min_periods=5).mean().iloc[-1] * 0.8
        
        # Simplified indicator checks - just need price momentum
        price_up = current['close'] > prev['close']
        rsi_ok = 25 < current['rsi'] < 75  # Wider RSI range
        
        # Simple momentum-based signals
        if price_up and rsi_ok and volume_ok:
            return 1  # Long signal
        elif not price_up and rsi_ok and volume_ok:
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
        current = df.iloc[-1]
        if len(self.open_trades) == 0:
            return False
            
        entry = self.open_trades[0]
        hours_held = (current.name - entry['entry_time']).total_seconds() / 3600
        
        # More generous exit conditions
        if current_position == 1:  # Long
            take_profit = entry['entry_price'] * 1.02  # 2% TP (was 1.5%)
            stop_loss = entry['entry_price'] * 0.985   # 1.5% SL (was 1%)
            return (current['close'] >= take_profit or 
                    current['close'] <= stop_loss or 
                    hours_held >= 8)  # Increased from 6h
                    
        elif current_position == -1:  # Short
            take_profit = entry['entry_price'] * 0.98  # 2% TP (was 1%)
            stop_loss = entry['entry_price'] * 1.015   # 1.5% SL (was 1.5%)
            return (current['close'] <= take_profit or 
                    current['close'] >= stop_loss or 
                    hours_held >= 6)  # Increased from 4h
        
        return False
    
    def get_current_entry_price(self, df, current_position):
        """Get entry price for current position"""
        if len(self.open_trades) > 0:
            return self.open_trades[0]['entry_price']
        return df['close'].iloc[-1]  # Fallback to current price

class SwingStrategy(MainStrategy):
    def get_signal(self, df):
        if len(df) < 50:
            return 0
            
        current = df.iloc[-1]
        
        # Your suggested "medium trend or Bollinger" condition
        trend_ok = (df['close'].iloc[-1] > df['close'].rolling(50).mean().iloc[-1])
        bollinger_ok = (current['close'] < current['bollinger_lower'] or 
                       current['close'] > current['bollinger_upper'])
        
        if trend_ok or bollinger_ok:
            return super().get_signal(df)
        return 0

    def get_exit_signal(self, df, current_position):
        current = df.iloc[-1]
        if len(self.open_trades) == 0:
            return False
            
        entry = self.open_trades[0]
        hours_held = (current.name - entry['entry_time']).total_seconds() / 3600
        
        # More generous swing exit conditions
        if current_position == 1:  # Long
            take_profit = entry['entry_price'] * 1.03  # 3% TP for swings
            stop_loss = entry['entry_price'] * 0.975   # 2.5% SL
            return (current['close'] >= take_profit or 
                    current['close'] <= stop_loss or 
                    hours_held >= 12)  # Longer hold for swings
                    
        elif current_position == -1:  # Short
            take_profit = entry['entry_price'] * 0.97  # 3% TP for swings
            stop_loss = entry['entry_price'] * 1.025   # 2.5% SL
            return (current['close'] <= take_profit or 
                    current['close'] >= stop_loss or 
                    hours_held >= 10)  # Longer hold for shorts
        
        return False

class ScalpStrategy(MainStrategy):
    def get_signal(self, df):
        if len(df) < 5:
            return 0
            
        current = df.iloc[-1]
        
        # Your RSI/volume spike suggestion
        rsi_ok = (current['rsi'] < 35 or current['rsi'] > 65)
        volume_spike = current['vol_spike'] > 1.5
        volume_ok = current['volume'] > df['volume'].rolling(20).mean().iloc[-1]
        
        if (rsi_ok or volume_spike) and volume_ok:
            return super().get_signal(df)
        return 0

    def get_exit_signal(self, df, current_position):
        current = df.iloc[-1]
        if len(self.open_trades) == 0:
            return False
            
        entry = self.open_trades[0]
        hours_held = (current.name - entry['entry_time']).total_seconds() / 3600
        
        # More generous scalp exit conditions
        if current_position == 1:  # Long
            take_profit = entry['entry_price'] * 1.008  # 0.8% TP (was tighter)
            stop_loss = entry['entry_price'] * 0.994    # 0.6% SL
            return (current['close'] >= take_profit or 
                    current['close'] <= stop_loss or 
                    hours_held >= 1)  # Max 1 hour hold
                    
        elif current_position == -1:  # Short
            take_profit = entry['entry_price'] * 0.992  # 0.8% TP
            stop_loss = entry['entry_price'] * 1.006    # 0.6% SL
            return (current['close'] <= take_profit or 
                    current['close'] >= stop_loss or 
                    hours_held >= 0.75)  # Max 45 minutes hold
        
        return False