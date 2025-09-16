
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

    def calculate_levels(self, prediction, df, strategy_config, tp_atr_mult_override=None, sl_atr_mult_override=None):
        """
        Calculates Stop Loss and Take Profit levels based on ATR.
        - Stop Loss is based on config 'sl_atr_multiplier' from the strategy.
        - Take Profit is based on config 'tp_atr_multiplier' from the strategy.
        - Falls back to percentage/RR ratio if ATR is unavailable.
        - Allows for override values for optimization purposes.
        """
        sl_config = RISK_CONFIG.get('stop_loss', {})
        sl_type = sl_config.get('type', 'atr') # Prioritize ATR
        current_price = prediction['current_price']
        direction = prediction['direction']
        atr = prediction.get('atr', 0)

        stop_loss_distance = 0
        take_profit_distance = 0

        # --- 1. CALCULATE SL/TP DISTANCES ---
        if sl_type == 'atr' and atr > 0:
            # Primary Logic: ATR-based distances from the specific strategy config
            sl_atr_multiplier = sl_atr_mult_override if sl_atr_mult_override is not None else strategy_config.get('sl_atr_multiplier', 1.5)
            tp_atr_multiplier = tp_atr_mult_override if tp_atr_mult_override is not None else strategy_config.get('tp_atr_multiplier', 2.0)
            
            stop_loss_distance = atr * sl_atr_multiplier
            take_profit_distance = atr * tp_atr_multiplier
            
        else:
            # Fallback Logic: Percentage-based SL and R/R-based TP
            percentage = sl_config.get('percentage', 1.5)
            rr_ratio = RISK_CONFIG.get('risk_reward_ratio', 1.33)
            
            stop_loss_distance = current_price * (percentage / 100)
            take_profit_distance = stop_loss_distance * rr_ratio

        # --- 2. CALCULATE FINAL SL/TP PRICES ---
        if direction == 'long':
            stop_loss = current_price - stop_loss_distance
            take_profit = current_price + take_profit_distance
        else:  # short
            stop_loss = current_price + stop_loss_distance
            take_profit = current_price - take_profit_distance
            
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

    def should_execute(self, prediction, asset, strategy_name, debug=False):
        """
        Validates if a trade should be executed based on the rules
        from the STRATEGIES dictionary in config.py.
        This now includes a dynamic ATR-based threshold check.
        """
        rules = STRATEGIES.get(asset, {}).get(strategy_name)
        if not rules:
            if debug: print(f"   -  RiskManager: No rules found for strategy '{strategy_name}' on asset '{asset}'.")
            return False

        confidence = armor_get(prediction, 'confidence', 0)
        pct_change = armor_get(prediction, 'pct_change', 0)
        direction = armor_get(prediction, 'direction', 'hold')
        current_price = armor_get(prediction, 'current_price', 0)
        atr = armor_get(prediction, 'atr', 0)
        atr_multiplier = rules.get('atr_threshold_multiplier')

        # 1. Confidence Check (applies to all strategies)
        if confidence < rules['min_confidence']:
            if debug: print(f"   - RiskManager: Confidence ({confidence:.2f}) is below threshold ({rules['min_confidence']}).")
            return False

        # 2. Direction Check
        if direction == 'hold':
            if debug: print("   - RiskManager: Signal is 'hold'.")
            return False

        # 3. Dynamic ATR Threshold Check (Primary Logic)
        if atr_multiplier and atr > 0 and current_price > 0:
            required_move_abs = atr * atr_multiplier
            predicted_move_abs = abs(current_price * (pct_change / 100))
            
            if debug: print(f"   - RiskManager (ATR Check): Required Move: ${required_move_abs:.4f}, Predicted Move: ${predicted_move_abs:.4f}")

            if predicted_move_abs < required_move_abs:
                if debug: print(f"   - RiskManager: Predicted move does not meet ATR-based threshold for '{strategy_name}'.")
                return False
        
        # 4. Static Percentage Threshold Check (Fallback Logic)
        else:
            if debug: print("   - RiskManager (Static Check): Using fallback percentage thresholds.")
            if direction == 'long' and pct_change < rules['long_threshold']:
                if debug: print(f"   - RiskManager: Predicted change ({pct_change:.2f}%) is below long threshold ({rules['long_threshold']}%).")
                return False
            if direction == 'short' and pct_change > -rules['short_threshold']:
                if debug: print(f"   - RiskManager: Predicted change ({pct_change:.2f}%) is above short threshold ({-rules['short_threshold']}%).")
                return False
            
        # If all checks pass
        if debug: print(f"   - RiskManager: Signal for {direction.upper()} {prediction['asset']} passed all checks.")
        return True
