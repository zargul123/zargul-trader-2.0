import pytz
import os

# ==============================================================================
# == SINGLE SOURCE OF TRUTH: STRATEGY CONFIGURATION                           ==
# ==============================================================================
# This dictionary is the ONLY place where strategy rules are defined.
# Both the live bot (main.py) and the simulator (backtest_engine.py)
# will read their rules from here to ensure 100% consistency.

STRATEGIES = {
    'main': {
        'timeframe': '1h',
        'long_threshold': 2.5,       # Aim for 2.5% up moves
        'short_threshold': 2.5,      # Aim for 2.5% down moves
        'min_confidence': 0.90,      # AI must be 90% sure
        'sequence_length': 45,       # Shorter memory, faster reaction
        'hold_period_hours': 8,      
        'dynamic_exit': True         
    },
    'swing': {
        'timeframe': '4h',
        'long_threshold': 1.2,       # Lowered threshold for more swing opportunities
        'short_threshold': 1.2,      # Lowered threshold for more swing opportunities
        'min_confidence': 0.85,      # Increased confidence for higher quality trades
        'sequence_length': 90,
        'hold_period_hours': 24,     
        'dynamic_exit': True
    },
    'scalp': {
        'timeframe': '15m',
        'long_threshold': 999,       # Impossible threshold - effectively disables scalping
        'short_threshold': 999,      # Impossible threshold - effectively disables scalping
        'min_confidence': 0.99,      # Nearly impossible confidence requirement
        'sequence_length': 90,
        'hold_period_hours': 0.5,    
        'dynamic_exit': True
    }
}


# ==============================================================================
# == UNIVERSAL SETTINGS                                                       ==
# ==============================================================================

# Assets to be traded
ASSETS = ["BTC-USD", "ETH-USD", "SOL-USD"]

# General risk management
RISK_PER_TRADE = 0.02  # Risk 2% of (pretend) capital on any single trade

# General Risk Configuration
RISK_CONFIG = {
    'risk_reward_ratio': 2.0,  # Aim for 2:1 risk-reward
    'max_daily_drawdown': 0.10, # Max 10% loss in a day
    'volatility_adjusted': True,      # New - adjust position size based on volatility
    'trailing_stop': {
        'enabled': True,
        'activation_pct': 0.5,       # Activate after 0.5% profit
        'trail_pct': 0.3             # Trail by 0.3%
    },
    'asset_weights': {               # Different risk per asset
        'BTC-USD': 1.0,
        'ETH-USD': 0.8,
        'SOL-USD': 0.6
    }
}

# Schedule for automatically retraining the AI models
AUTO_TRAIN_SCHEDULE = {
    'enabled': True,
    'day_of_week': "sunday", # Lowercase day name
    'time_utc': "03:00"      # 3 AM UTC
}


# ==============================================================================
# == AI & TRAINING CONFIGURATION                                              ==
# ==============================================================================

# These settings control how the AI model is built and trained
TRAINING_CONFIG = {
    'epochs': 100,                  # Max training rounds
    'batch_size': 32,               # How many data samples to process at once
    'early_stop_patience': 10,      # Stop training if no improvement after 10 epochs
    'training_days': 365            # Use 1 year of historical data for training
}


# ==============================================================================
# == API & ENVIRONMENT CONFIGURATION                                          ==
# ==============================================================================

# TwelveData API settings
TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY")
TWELVEDATA_CONFIG = {
    'base_url': 'https://api.twelvedata.com',
    'timeout': 10,
    'rate_limit': 8,  # requests per minute
    'max_retries': 3
}
TWELVEDATA_MAPPING = {
    "BTC-USD": "BTC/USD",
    "ETH-USD": "ETH/USD",
    "SOL-USD": "SOL/USD",
}

# Technical indicators to be calculated and fed to the AI
TECHNICAL_INDICATORS = [
    'rsi', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower', 'obv',
    'vol_spike', 'cmf', 'vwap', 'ema_20', 'ema_50', 'ema_200', 
    'atr', 'stoch_k', 'stoch_d', 'adx', 'volume_ma'
]

# Suppress excessive TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set the system's timezone for consistent date/time handling
TIMEZONE = pytz.timezone('UTC')