
import numpy as np
import pandas as pd
from scripts.config import RISK_REWARD_RATIO, MIN_CONFIDENCE, LONG_THRESHOLD, SHORT_THRESHOLD
from scripts.core.safety import armor_get

def super_safe_get(obj, key, default=None):
    """Legacy wrapper for armor_get"""
    return armor_get(obj, key, default)

class RiskManager:
    def _calculate_atr(self, df, period=14):
        """Average True Range calculation"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(period).mean().iloc[-1]

    def calculate_levels(self, df, prediction):
        """ATR-based risk calculation with 2:1 reward ratio"""
        # Add this at the start:
        print(f"🔍 Analyzing {prediction['asset']} at {df.index[-1]}")
        print(f"   Current Price: {df['close'].iloc[-1]}")
        print(f"   Volume: {df['volume'].iloc[-1]} vs avg {df['volume'].mean()}")
        
        atr = self._calculate_atr(df)
        current_price = df['close'].iloc[-1]
        
        return {
            'sl': current_price - (atr * 2),  # 2x ATR stop
            'tp': current_price + (atr * 4),   # 2:1 reward
            'size': min(0.2, 0.05 * (1 + (df['volume'].iloc[-1]/df['volume'].mean())))  # Sizes based on volume
        }

    def _get_position_size(self, df):
        vol_trend = armor_get(df, 'vol_trend', 0)  # Use 0 if missing
        """Volume-trend based sizing"""
        if vol_trend > 0.1: return 'Large'
        if vol_trend < -0.1: return 'Small'
        return 'Medium'


class RiskManagerPro(RiskManager):
    def __init__(self):
        super().__init__()
        self.portfolio_heat = 1.0  # Start neutral

    def dynamic_position_sizing(self, volatility):
        """Kelly Criterion-inspired sizing"""
        optimal = (volatility ** 2) / (self.portfolio_heat * 0.1)  # 0.1 = estimated Sharpe
        return min(max(optimal, 0.01), 0.1)  # Cap between 1-10%

    def update_portfolio_heat(self, recent_performance):
        """Adjust exposure based on performance"""
        self.portfolio_heat *= 0.9 if recent_performance > 0 else 1.1


    def should_execute(self, prediction):
        """Validate if a trade should be executed based on confidence and price movement thresholds"""
        if armor_get(prediction, 'confidence', 0) < MIN_CONFIDENCE:
            return False
        if prediction['direction'] == 'long' and armor_get(prediction, 'pct_change', 0) < LONG_THRESHOLD:
            return False
        if prediction['direction'] == 'short' and armor_get(prediction, 'pct_change', 0) > -SHORT_THRESHOLD:
            return False
        return True
