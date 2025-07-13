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
        'long_threshold': 0.8,       # Minimum % increase for a long signal
        'short_threshold': 0.6,      # Minimum % decrease for a short signal
        'min_confidence': 0.65,      # AI's confidence must be at least this value (0.0 to 1.0)
        'sequence_length': 60,       # How many past candles the AI looks at (must match model)
        'hold_period_hours': 12,     # How long to hold a trade in backtesting
    },
    'swing': {
        'timeframe': '4h',
        'long_threshold': 1.5,
        'short_threshold': 1.2,
        'min_confidence': 0.70,
        'sequence_length': 60,
        'hold_period_hours': 48,
    },
    'scalp': {
        'timeframe': '15m',
        'long_threshold': 0.5,
        'short_threshold': 0.5,
        'min_confidence': 0.75,
        'sequence_length': 60,
        'hold_period_hours': 1,
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
TWELVEDATA_API_KEY = "2c86eee94557424ea431537d0d59a5b1"
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
    'vol_spike', 'cmf', 'vwap'
]

# Suppress excessive TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set the system's timezone for consistent date/time handling
TIMEZONE = pytz.timezone('UTC')
