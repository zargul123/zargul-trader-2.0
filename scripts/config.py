import pytz
import os

# ==============================================================================
# == SINGLE SOURCE OF TRUTH: STRATEGY CONFIGURATION                           ==
# ==============================================================================
# This dictionary is the ONLY place where strategy rules are defined.
# Both the live bot (main.py) and the simulator (backtest_engine.py)
# will read their rules from here to ensure 100% consistency.
STRATEGIES = {
    "BTC-USD": {
        'main': {
            'enabled': True,
            'timeframe': '1h',
            'atr_threshold_multiplier': 1.5,
            'long_threshold': 2.5,
            'short_threshold': 2.5,
            'min_confidence': 0.60, # Lowered from 0.75 for diagnostics
            'sequence_length': 20, # Set to 20 as requested
            'hold_period_hours': 8,
            'dynamic_exit': True
        },
        'btc-swing': {
            'enabled': True,
            'timeframe': '4h',
            'atr_threshold_multiplier': 1.0,
            'long_threshold': 2.0,
            'short_threshold': 2.0,
            'min_confidence': 0.90,
            'sequence_length': 90,
            'hold_period_hours': 24,
            'dynamic_exit': True
        },
        'scalp': {
            'enabled': True,
            'timeframe': '5m',
            'atr_threshold_multiplier': 1.0,
            'long_threshold': 0.5,
            'short_threshold': 0.5,
            'min_confidence': 0.75,
            'sequence_length': 22,
            'hold_period_hours': 1,
            'dynamic_exit': True
        }
    },
    "ETH-USD": {
        'main': {
            'enabled': True,
            'timeframe': '1h',
            'atr_threshold_multiplier': 1.5,
            'long_threshold': 2.5,
            'short_threshold': 2.5,
            'min_confidence': 0.90,
            'sequence_length': 20,
            'hold_period_hours': 8,
            'dynamic_exit': True
        },
        'swing': {
            'enabled': True,
            'timeframe': '4h',
            'atr_threshold_multiplier': 1.5,
            'long_threshold': 1.2,
            'short_threshold': 1.2,
            'min_confidence': 0.85,
            'sequence_length': 90,
            'hold_period_hours': 24,
            'dynamic_exit': True
        },
        'scalp': {
            'enabled': True,
            'timeframe': '5m',
            'atr_threshold_multiplier': 1.0,
            'long_threshold': 0.5,
            'short_threshold': 0.5,
            'min_confidence': 0.75,
            'sequence_length': 22,
            'hold_period_hours': 1,
            'dynamic_exit': True
        }
    },
    "SOL-USD": {
        'main': {
            'enabled': True,
            'timeframe': '1h',
            'atr_threshold_multiplier': 1.5,
            'long_threshold': 2.5,
            'short_threshold': 2.5,
            'min_confidence': 0.90,
            'sequence_length': 20,
            'hold_period_hours': 8,
            'dynamic_exit': True
        },
        'swing': {
            'enabled': True,
            'timeframe': '4h',
            'atr_threshold_multiplier': 1.5,
            'long_threshold': 1.2,
            'short_threshold': 1.2,
            'min_confidence': 0.85,
            'sequence_length': 90,
            'hold_period_hours': 24,
            'dynamic_exit': True
        },
        'scalp': {
            'enabled': True,
            'timeframe': '5m',
            'atr_threshold_multiplier': 1.0,
            'long_threshold': 0.5,
            'short_threshold': 0.5,
            'min_confidence': 0.75,
            'sequence_length': 22,
            'hold_period_hours': 1,
            'dynamic_exit': True
        }
    }
}


# ==============================================================================
# == UNIVERSAL SETTINGS                                                       ==
# ==============================================================================

# Assets to be traded
ASSETS = ["BTC-USD", "ETH-USD", "SOL-USD"]

# General risk management
RISK_PER_TRADE = 0.02  # Risk 2% of (pretend) capital on any single trade

# Backtesting-specific settings for realistic simulation
BACKTEST_CONFIG = {
    'fees_pct': 0.001,           # 0.1% fee per trade (buy/sell)
    'slippage_pct': 0.0005       # 0.05% slippage per trade
}

# General Risk Configuration
RISK_CONFIG = {
    'risk_reward_ratio': 1.33, # Fallback R/R, derived from 2.0 / 1.5
    'tp_atr_multiplier': 2.0,    # New: Take Profit is 2.0 * ATR
    'stop_loss': {
        'type': 'atr',               # 'atr' or 'percentage'
        'atr_multiplier': 1.5,       # Stop Loss is 1.5 * ATR
        'percentage': 1.5            # Fallback percentage if ATR fails
    },
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
    'training_days': 1095           # Use 3 years of historical data for training
}

# Asset-specific, optimized model architectures.
# Each asset has a dictionary of strategies, allowing for fine-tuned models per use case.
MODEL_HYPERPARAMS = {
    'BTC-USD': {
        'main': {
            'n_layers': 1,
            'learning_rate': 0.001394224585333133,
            'units_layer_0': 89,
            'dropout_layer_0': 0.24254440958679202
        },
        'scalp': {
            'n_layers': 3,
            'learning_rate': 0.009982953155109069,
            'units_layer_0': 238,
            'dropout_layer_0': 0.35081521888366163,
            'units_layer_1': 92,
            'dropout_layer_1': 0.49685851840612133,
            'units_layer_2': 248,
            'dropout_layer_2': 0.4932229889671592
        },
        'btc-swing': { # Placeholder: Using main's params as a default
            'n_layers': 1,
            'learning_rate': 0.009886643874520377,
            'units_layer_0': 251,
            'dropout_layer_0': 0.35025264979998394
        }
    },
    'ETH-USD': {
        'main': {
            'n_layers': 3,
            'learning_rate': 0.009847271267492321,
            'units_layer_0': 52,
            'dropout_layer_0': 0.4547476109205212,
            'units_layer_1': 36,
            'dropout_layer_1': 0.26953065823496686,
            'units_layer_2': 206,
            'dropout_layer_2': 0.15479272548181033
        },
        'scalp': { # Placeholder: Using main's params as a default
            'n_layers': 3,
            'learning_rate': 0.009847271267492321,
            'units_layer_0': 52,
            'dropout_layer_0': 0.4547476109205212,
            'units_layer_1': 36,
            'dropout_layer_1': 0.26953065823496686,
            'units_layer_2': 206,
            'dropout_layer_2': 0.15479272548181033
        },
        'swing': { # Placeholder: Using main's params as a default
            'n_layers': 3,
            'learning_rate': 0.009847271267492321,
            'units_layer_0': 52,
            'dropout_layer_0': 0.4547476109205212,
            'units_layer_1': 36,
            'dropout_layer_1': 0.26953065823496686,
            'units_layer_2': 206,
            'dropout_layer_2': 0.15479272548181033
        }
    },
    'SOL-USD': {
        'main': {
            'n_layers': 1,
            'learning_rate': 0.009906324714275988,
            'units_layer_0': 151,
            'dropout_layer_0': 0.13198466453039642
        },
        'scalp': { # Placeholder: Using main's params as a default
            'n_layers': 1,
            'learning_rate': 0.009906324714275988,
            'units_layer_0': 151,
            'dropout_layer_0': 0.13198466453039642
        },
        'swing': { # Placeholder: Using main's params as a default
            'n_layers': 1,
            'learning_rate': 0.009906324714275988,
            'units_layer_0': 151,
            'dropout_layer_0': 0.13198466453039642
        }
    },
    'default': { # Fallback for any other assets
        'n_layers': 1,
        'learning_rate': 0.005,
        'units_layer_0': 128,
        'dropout_layer_0': 0.3
    }
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

# LunarCrush API Configuration for Social Sentiment Analysis
LUNARCRUSH_CONFIG = {
    'api_key': os.environ.get("LUNARCRUSH_API_KEY"),
    'api_url': 'https://lunarcrush.com/api4',
    # The specific, high-value metrics we will feed to our AI
    'metrics_to_use': [
        'galaxy_score',
        'alt_rank',
        'social_volume_24h',
        'social_dominance',
        'sentiment'
    ]
}

# Technical indicators to be calculated and fed to the AI
TECHNICAL_INDICATORS = [
    # --- Classic Indicators ---
    'rsi', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower', 'obv',
    'vol_spike', 'cmf', 'vwap', 'ema_20', 'ema_50', 'ema_200', 
    'atr', 'stoch_k', 'stoch_d', 'adx', 'volume_ma',
    # --- Elite Engineered Features (Normalized & Derivative) ---
    'atr_norm', 'macd_norm', 'bollinger_width', 'ema_spread', 
    'pct_change', 'log_return' # Ensure ADX is here
] + [f'lc_{metric}' for metric in LUNARCRUSH_CONFIG['metrics_to_use']]

# Market Regime Filter Configuration
REGIME_CONFIG = {
    'adx_trending_threshold': 25,       # ADX value above which the market is considered "trending"
    'entropy_chaotic_threshold': 1.5,   # Shannon Entropy value above which the market is "chaotic"
    'entropy_window': 50,               # Number of candles to use for entropy calculation
    'entropy_smoothing_alpha': 0.1      # Smoothing factor for the entropy EMA (lower is smoother)
}

# Suppress excessive TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set the system's timezone for consistent date/time handling
TIMEZONE = pytz.timezone('UTC')