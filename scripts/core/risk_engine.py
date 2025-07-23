
import numpy as np
import pandas as pd
from scripts.config import STRATEGIES, RISK_CONFIG, RISK_PER_TRADE
from scripts.core.safety import armor_get

class RiskManager:
    def _calculate_atr(self, df, period=14):
        """Average True Range calculation"""
        if df.empty or len(df) < period:
            return 0.01 # Return a default non-zero value
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(period).mean().iloc[-1]

    def calculate_levels(self, prediction, df):
        """
        Calculates Stop Loss and Take Profit levels.
        - Stop Loss is ATR or percentage-based.
        - Take Profit is dynamic, targeting swing points with an ATR offset,
          with a fallback to a fixed risk/reward ratio.
        """
        sl_config = RISK_CONFIG.get('stop_loss', {})
        sl_type = sl_config.get('type', 'percentage')
        current_price = prediction['current_price']
        direction = prediction['direction']
        rr_ratio = RISK_CONFIG.get('risk_reward_ratio', 2.0)
        atr = prediction.get('atr', 0)

        # --- 1. CALCULATE STOP LOSS ---
        stop_loss_distance = 0
        if sl_type == 'atr' and atr > 0:
            atr_multiplier = sl_config.get('atr_multiplier', 2.0)
            stop_loss_distance = atr * atr_multiplier
        else:
            # Fallback to percentage if ATR is missing, zero, or type is 'percentage'
            percentage = sl_config.get('percentage', 1.5)
            stop_loss_distance = current_price * (percentage / 100)

        if direction == 'long':
            stop_loss = current_price - stop_loss_distance
        else:  # short
            stop_loss = current_price + stop_loss_distance

        # --- 2. CALCULATE DYNAMIC TAKE PROFIT ---
        take_profit = None
        lookback_period = 20
        if len(df) >= lookback_period and atr > 0:
            lookback_df = df.tail(lookback_period)
            atr_offset = 0.3 * atr

            if direction == 'long':
                swing_high = lookback_df['high'].max()
                dynamic_tp = swing_high - atr_offset
                # Use dynamic TP only if it's actually profitable
                if dynamic_tp > current_price:
                    take_profit = dynamic_tp
            else:  # short
                swing_low = lookback_df['low'].min()
                dynamic_tp = swing_low + atr_offset
                # Use dynamic TP only if it's actually profitable
                if dynamic_tp < current_price:
                    take_profit = dynamic_tp
        
        # --- 3. FALLBACK TO FIXED RISK/REWARD RATIO ---
        if take_profit is None:
            if direction == 'long':
                take_profit = current_price + (stop_loss_distance * rr_ratio)
            else: # short
                take_profit = current_price - (stop_loss_distance * rr_ratio)
            
        return {'stop_loss': stop_loss, 'take_profit': take_profit}
    
    def update_trailing_stop(self, entry_price, current_price, direction, current_stop=None):
        """
        Updates trailing stop loss based on RISK_CONFIG settings.
        """
        if not RISK_CONFIG.get('trailing_stop', {}).get('enabled', False):
            return current_stop
            
        trail_config = RISK_CONFIG['trailing_stop']
        activation_pct = trail_config['activation_pct'] / 100
        trail_pct = trail_config['trail_pct'] / 100
        
        if direction == 'long':
            # Check if we've hit activation threshold
            profit_pct = (current_price - entry_price) / entry_price
            if profit_pct >= activation_pct:
                new_stop = current_price * (1 - trail_pct)
                return max(current_stop or 0, new_stop)  # Only move stop up
        else: # short
            # For shorts, profit when price goes down
            profit_pct = (entry_price - current_price) / entry_price
            if profit_pct >= activation_pct:
                new_stop = current_price * (1 + trail_pct)
                return min(current_stop or float('inf'), new_stop)  # Only move stop down
                
        return current_stop

    def calculate_position_size(self, df, symbol=None):
        """
        Calculates position size based on volatility (ATR) and asset weights.
        """
        base_risk = RISK_PER_TRADE
        
        # Apply asset-specific weights
        if symbol and symbol in RISK_CONFIG.get('asset_weights', {}):
            asset_weight = RISK_CONFIG['asset_weights'][symbol]
            base_risk = base_risk * asset_weight
        
        # Apply volatility adjustment if enabled
        if RISK_CONFIG.get('volatility_adjusted', False):
            atr = self._calculate_atr(df)
            if atr == 0: # Avoid division by zero
                return base_risk
                
            volatility = atr / df['close'].iloc[-1]
            
            # Dynamic sizing: take less risk when volatility is high
            position_size = base_risk / max(volatility, 0.01) # Ensure volatility is not zero
        else:
            position_size = base_risk
        
        # Cap position size to a max of 10% of portfolio
        return min(position_size, 0.10)

    def should_execute(self, prediction, strategy_name):
        """
        Validates if a trade should be executed based on the rules
        from the STRATEGIES dictionary in config.py.
        """
        rules = STRATEGIES.get(strategy_name)
        if not rules:
            return False # Don't trade if strategy doesn't exist

        confidence = armor_get(prediction, 'confidence', 0)
        pct_change = armor_get(prediction, 'pct_change', 0)
        direction = armor_get(prediction, 'direction', 'hold')

        if confidence < rules['min_confidence']:
            return False
        if direction == 'long' and pct_change < rules['long_threshold']:
            return False
        if direction == 'short' and pct_change > -rules['short_threshold']:
            return False
            
        return True
