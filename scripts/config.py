
import os

# ===== QUANTUM TRADER PRO STRATEGY CONFIG =====
# Optimized for BTC/ETH/SOL - Tested Parameters

STRATEGIES = {
    'main': {
        'timeframe': '1h',
        'long_threshold': 0.85,      # 0.85% target upside
        'short_threshold': 0.65,     # 0.65% target downside
        'min_confidence': 0.68,      # 68%+ AI confidence
        'min_hold': 14400,           # 4 hours minimum (in seconds)
        'max_hold': 86400,           # Max 24 hours hold
        'sl_multiplier': 1.8,        # 1.8x ATR for stop loss
        'tp_multiplier': 3.0,        # 3.0x ATR for take profit
        'rsi_limits': (28, 72)       # Oversold/overbought levels
    },
    'swing': {
        'timeframe': '4h',
        'long_threshold': 1.25,      # 1.25% target upside  
        'short_threshold': 1.1,       # 1.1% target downside
        'min_confidence': 0.72,      # 72%+ AI confidence
        'min_hold': 43200,           # 12 hours minimum
        'max_hold': 259200,          # Max 3 days hold
        'sl_multiplier': 1.5,        # 1.5x ATR
        'tp_multiplier': 4.0,        # 4.0x ATR 
        'rsi_limits': (25, 75)       # Wider bands for swings
    },
    'scalp': {
        'timeframe': '15m',          # 15m gives better signals than 5m
        'long_threshold': 0.35,      # 0.35% quick gain
        'short_threshold': 0.3,      # 0.3% quick drop
        'min_confidence': 0.78,      # 78%+ AI confidence
        'min_hold': 300,             # 5 minutes minimum
        'max_hold': 1800,            # Max 30 minutes hold
        'sl_multiplier': 0.5,        # Tight 0.5x ATR stop
        'tp_multiplier': 1.5,        # 1.5x ATR target
        'rsi_limits': (32, 68)       # Tight bands for scalping
    }
}

# Universal Settings
ASSETS = ["BTC-USD", "ETH-USD", "SOL-USD"]
RISK_PER_TRADE = 0.01               # Risk 1% of capital per trade
AUTO_TRAIN = True                   # Keep models fresh

# ===== TRAINING SCHEDULE =====
RETRAIN_DAY = "sunday"     # Lowercase day name
RETRAIN_TIME = "03:00"     # 3 AM UTC (adjust if needed)
MIN_CONFIDENCE = 0.65      # Retrain if confidence < 65%

# Legacy Trading Strategy Parameters (backwards compatibility)
# ---------------------------
LONG_THRESHOLD = 0.5   # 0.5% move needed
SHORT_THRESHOLD = 0.5  
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

# System Optimization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs
os.environ['OMP_NUM_THREADS'] = '6'  # Match CPU cores
