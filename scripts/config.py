import os

# ===== TRAINING SCHEDULE =====
AUTO_TRAIN = True          # Enable auto-training
RETRAIN_DAY = "sunday"     # Lowercase day name
RETRAIN_TIME = "03:00"     # 3 AM UTC (adjust if needed)
MIN_CONFIDENCE = 0.65      # Retrain if confidence < 65%

# Trading Strategy Parameters
# ---------------------------
# Main Strategy (updated for better sensitivity in crypto)
LONG_THRESHOLD = 0.5   # 0.5% move needed
SHORT_THRESHOLD = 0.5  
MIN_CONFIDENCE = 0.60  # Slightly lower confidence threshold
RISK_REWARD_RATIO = 2.0  # Classic institutional risk model

# Swing Trading Strategy (tighter trigger, faster entry, more flexible hold)
SWING_THRESHOLD = 0.85  # 0.85% move needed for swing entries
SWING_MIN_HOLD = 64800  # 18 hours minimum hold (crypto is faster-paced)
SWING_MAX_HOLD = 172800  # Max 2 days hold to avoid weekend slippage
SWING_MIN_CONFIDENCE = 0.7  # Slight bump for stronger signals

# Scalp Trading Strategy (faster reaction + tighter hold range)
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

# Assets
ASSETS = ["BTC-USD", "ETH-USD", "SOL-USD"]

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
