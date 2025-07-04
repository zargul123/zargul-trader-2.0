
import pytz  # Add to requirements.txt if needed
import os

TIMEZONE = pytz.timezone('UTC')  # Or 'America/New_York' etc.

# ===== QUANTUM TRADER PRO STRATEGY CONFIG =====
# Optimized for BTC/ETH/SOL - Tested Parameters

# Enhanced Strategy Configs
STRATEGIES = {
    'main': {
        'timeframe': '4h',
        'long_threshold': 0.3,       # Reduced from 0.85
        'short_threshold': 0.25,     # Reduced from 0.65
        'min_confidence': 0.55,      # Reduced from 0.68
        'required_indicators': 1,    # Only need 1/3 indicators to confirm
        'min_hold': 21600,           # 6 hours (reduced from 8)
        'max_hold': 86400,           # 24 hours (reduced from 48)
        'sl_multiplier': 2.0,        # Reduced from 2.5
        'tp_multiplier': 3.0,        # Reduced from 4.0
        'rsi_limits': (30, 70),      # Widened from (28,72)
        'trading_hours': [2,5,8,11,14,17,20,23],  # More frequent
        'timezone': 'UTC'
    },
    'swing': {
        'timeframe': '4h',
        'long_threshold': 0.5,       # Reduced from 1.25
        'short_threshold': 0.45,     # Reduced from 1.1
        'min_confidence': 0.6,       # Reduced from 0.72
        'trend_strength': 'medium',  # More flexible trend requirement
        'min_hold': 36000,           # 10 hours (reduced from 12)
        'max_hold': 172800,          # 48 hours (reduced from 72)
        'sl_multiplier': 2.5,        # Reduced from 3.0
        'tp_multiplier': 4.0,        # Reduced from 5.0
        'rsi_limits': (28, 72),      # Slightly wider
        'trading_hours': [4,10,16,22], # More opportunities
        'timezone': 'UTC'
    },
    'scalp': {
        'timeframe': '30m',
        'long_threshold': 0.15,      # Reduced from 0.35
        'short_threshold': 0.1,      # Reduced from 0.3
        'min_confidence': 0.65,      # Reduced from 0.78
        'volume_multiplier': 1.2,    # Reduced from 1.5
        'min_hold': 600,             # 10 minutes (reduced from 15)
        'max_hold': 1800,            # 30 minutes (reduced from 60)
        'sl_multiplier': 0.4,        # Tighter from 0.5
        'tp_multiplier': 1.2,        # Reduced from 1.5
        'rsi_limits': (35, 65),      # More sensitive
        'trading_hours': [0,2,4,6,8,10,12,14,16,18,20,22],  # All hours
        'timezone': 'UTC'
    }
}

# Universal Settings
ASSETS = ["BTC-USD", "ETH-USD", "SOL-USD"]
RISK_PER_TRADE = 0.02               # Slightly increased risk for better position sizing
AUTO_TRAIN = True                   # Keep models fresh

# ===== TRAINING SCHEDULE =====
RETRAIN_DAY = "sunday"     # Lowercase day name
RETRAIN_TIME = "03:00"     # 3 AM UTC (adjust if needed)
MIN_CONFIDENCE = 0.55      # Retrain if confidence < 55%

# Updated Trading Parameters
LONG_THRESHOLD = 0.25   # Reduced from 0.5
SHORT_THRESHOLD = 0.25  
MIN_CONFIDENCE = 0.55   # Reduced from 0.65
RISK_PER_TRADE = 0.02   # Slightly increased risk for better position sizing
RISK_REWARD_RATIO = 2.0  # Classic institutional risk model

# Legacy Swing Trading Strategy
SWING_THRESHOLD = 0.85  # 0.85% move needed for swing entries
SWING_MIN_HOLD = 64800  # 18 hours minimum hold (crypto is faster-paced)
SWING_MAX_HOLD = 172800  # Max 2 days hold to avoid weekend slippage
SWING_MIN_CONFIDENCE = 0.7  # Slight bump for stronger signals

# Legacy Scalp Trading Strategy
SCALP_THRESHOLD = 0.25  # 0.25% move = crypto scalp sweet spot
SCALP_MIN_HOLD = 300  # 5 minutes minimum (unchanged)
SCALP_MAX_HOLD = 2700  # Max 45 minutes (scalping should be fast)
SCALP_MIN_CONFIDENCE = 0.82  # High conviction scalp plays only

# Training Configuration
TRAINING_EPOCHS = 200
BATCH_SIZE = 32
EARLY_STOP_PATIENCE = 7
SEQUENCE_LENGTH = 60
TRAINING_DAYS = 180  # 6 months of training data

# TwelveData Configuration
TWELVEDATA_API_KEY = "2c86eee94557424ea431537d0d59a5b1"  # Your actual key
TWELVEDATA_CONFIG = {
    'base_url': 'https://api.twelvedata.com',
    'timeout': 5,
    'rate_limit': 8,  # requests per minute
    'max_retries': 3
}
TWELVEDATA_MAPPING = {
    "BTC-USD": "BTC/USD",
    "ETH-USD": "ETH/USD",
    "SOL-USD": "SOL/USD",
    "BNB-USD": "BNB/USD"
}

# Technical Indicators
TECHNICAL_INDICATORS = [
    'rsi', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower', 'obv',
    'vol_spike', 'cmf', 'vwap'
]

# Asset-Specific Optimization
BTC_SETTINGS = {
    'long_threshold': 0.9,  # BTC needs bigger moves
    'rsi_limits': (30, 70)
}

ETH_SETTINGS = {
    'long_threshold': 0.7,  # ETH reacts quicker
    'rsi_limits': (35, 65)
}

# Scalp Trading Settings
SCALP_SETTINGS = {
    'min_confidence': 0.82,  # Increase from 0.75
    'min_hold': 300,         # 5 minutes minimum
    'max_hold': 900          # 15 minutes maximum (reduced from 30)
}

# System Optimization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs
os.environ['OMP_NUM_THREADS'] = '6'  # Match CPU cores
