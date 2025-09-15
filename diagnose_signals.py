
import os
import sys
import pandas as pd
import numpy as np
import joblib

# --- Setup Project Environment ---
# This ensures the script can find the other project files
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from scripts.core.analysis_engine import AIAnalyst
from scripts.core.data_engine import DataMaster
from scripts.core.risk_engine import RiskManager
from scripts.config import STRATEGIES, REGIME_CONFIG

# --- Configuration for the Diagnostic Test ---
ASSET_TO_DIAGNOSE = "BTC-USD"
STRATEGY_TO_DIAGNOSE = "main"
REGIME_TO_DIAGNOSE = "Trending"
CANDLES_TO_ANALYZE = 500 # Analyze the first 500 candles of the dataset
CACHE_DIR = 'cache'

# --- Helper functions copied from optimize_strategy.py for consistency ---

def _calculate_shannon_entropy(series, window):
    """Helper function to calculate entropy for a rolling window."""
    bins = pd.cut(series, bins=[-np.inf, -0.005, 0, 0.005, np.inf], labels=False, right=False)
    counts = np.bincount(bins, minlength=4)
    probabilities = counts / len(series)
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))

def calculate_historical_regimes(df: pd.DataFrame) -> pd.Series:
    """Calculates the market regime for each row in a historical DataFrame."""
    print("--> Calculating historical market regimes...")
    price_returns = df['close'].pct_change().fillna(0)
    rolling_entropy = price_returns.rolling(window=REGIME_CONFIG['entropy_window']).apply(
        _calculate_shannon_entropy, raw=True, kwargs={'window': REGIME_CONFIG['entropy_window']}
    )
    smoothed_entropy = rolling_entropy.ewm(alpha=REGIME_CONFIG['entropy_smoothing_alpha']).mean()
    is_chaotic = smoothed_entropy > REGIME_CONFIG['entropy_chaotic_threshold']
    is_trending = df['adx'] > REGIME_CONFIG['adx_trending_threshold']
    regimes = pd.Series("Ranging", index=df.index)
    regimes[is_trending] = "Trending"
    regimes[is_chaotic] = "Chaotic"
    print("--> Regime calculation complete.")
    return regimes

def load_regime_data(asset, timeframe, regime_type):
    """
    Loads a large dataset and filters it for a specific market regime.
    Uses the same caching logic as the optimizer.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_filename = f"{asset}_{timeframe}_{regime_type}_data.pkl"
    cache_filepath = os.path.join(CACHE_DIR, cache_filename)

    if os.path.exists(cache_filepath):
        print(f"--> Loading pre-processed '{regime_type}' data from cache: {cache_filepath}")
        filtered_df = joblib.load(cache_filepath)
        print(f"--> Cached data loaded successfully ({len(filtered_df)} candles).")
        return filtered_df

    print(f"--> No cache found. Performing one-time data preparation for {asset} in {regime_type} regime...")
    data_master = DataMaster()
    df = data_master.get_training_data(asset, timeframe, days=1095)
    if df is None or df.empty:
        raise ValueError(f"Could not load historical data for {asset}.")

    df['regime'] = calculate_historical_regimes(df)
    df.dropna(inplace=True)
    filtered_df = df[df['regime'] == regime_type].copy()

    if len(filtered_df) < 200:
        print(f"WARNING: Only {len(filtered_df)} data points found for the {regime_type} regime.")
        if len(filtered_df) < 50:
             raise ValueError("Insufficient data for analysis after regime filtering.")

    print(f"--> Saving prepared data to cache: {cache_filepath}")
    joblib.dump(filtered_df, cache_filepath)
    
    print(f"--> Found {len(filtered_df)} candles for the '{regime_type}' regime.")
    return filtered_df

def run_diagnostics():
    """
    Main function to run the signal diagnostics.
    """
    print("="*80)
    print("🤖 RUNNING AI SIGNAL DIAGNOSTIC SCRIPT 🤖")
    print("="*80)

    try:
        # 1. Load the exact same data the optimizer uses
        strategy_config = STRATEGIES[STRATEGY_TO_DIAGNOSE]
        timeframe = strategy_config['timeframe']
        df = load_regime_data(ASSET_TO_DIAGNOSE, timeframe, REGIME_TO_DIAGNOSE)

        # 2. Initialize the AI and Risk Manager
        print("\n--> Initializing AI Analyst and Risk Manager...")
        analyst = AIAnalyst(symbol=ASSET_TO_DIAGNOSE, strategy_type=STRATEGY_TO_DIAGNOSE)
        risk_manager = RiskManager()
        print("--> Engines are ready.")

        # 3. Loop through the data and analyze signals
        sequence_length = strategy_config['sequence_length']
        
        # Use the strategy config from the main config file as our baseline
        min_conf = strategy_config['min_confidence']
        atr_mult = strategy_config['atr_threshold_multiplier']

        print(f"\n--> Analyzing the first {CANDLES_TO_ANALYZE} candles for signals...")
        print(f"--> Using baseline rules: min_confidence={min_conf}, atr_multiplier={atr_mult}")
        print("-" * 110)
        print("{:<22} {:<7} {:<6} {:<7} {:<7} {:<15} {:<12} {:<12} {:<15}".format(
            "Timestamp", "Direction", "Conf", "Move%", "ATR", "Required Move%", "Conf Pass?", "ATR Pass?", "Simulated Outcome"
        ))
        print("-" * 110)

        for i in range(sequence_length, min(len(df), sequence_length + CANDLES_TO_ANALYZE)):
            window_data = df.iloc[i - sequence_length : i]
            current_time = df.index[i]
            
            prediction = analyst.predict(ASSET_TO_DIAGNOSE, window_data, strategy_name=STRATEGY_TO_DIAGNOSE)

            if prediction and prediction['direction'] != 'hold':
                # --- Perform the checks ---
                conf_pass = prediction['confidence'] >= min_conf
                
                required_move_abs = prediction['atr'] * atr_mult
                predicted_move_abs = abs(prediction['current_price'] * (prediction['pct_change'] / 100))
                atr_pass = predicted_move_abs >= required_move_abs
                
                # --- Convert required move to percentage for easier comparison ---
                required_move_pct = (required_move_abs / prediction['current_price']) * 100 if prediction['current_price'] > 0 else 0

                # --- NEW: SIMULATE TRADE OUTCOME ---
                outcome = "-"
                if conf_pass and atr_pass:
                    levels = risk_manager.calculate_levels(prediction, window_data)
                    sl_price = levels['stop_loss']
                    tp_price = levels['take_profit']
                    
                    sim_outcome = "TIMEOUT" # Default if neither SL nor TP is hit
                    # Look ahead up to 50 candles
                    future_candles = df.iloc[i + 1 : i + 1 + 50]

                    for _, candle in future_candles.iterrows():
                        if prediction['direction'] == 'long':
                            if candle['low'] <= sl_price:
                                sim_outcome = "LOSS"
                                break
                            if candle['high'] >= tp_price:
                                sim_outcome = "WIN"
                                break
                        elif prediction['direction'] == 'short':
                            if candle['high'] >= sl_price:
                                sim_outcome = "LOSS"
                                break
                            if candle['low'] <= tp_price:
                                sim_outcome = "WIN"
                                break
                    outcome = sim_outcome
                # --- END OF NEW SIMULATION LOGIC ---

                # --- Print the detailed report line ---
                print(
                    f"{str(current_time):<22} "
                    f"{prediction['direction'].upper():<7} "
                    f"{prediction['confidence']:.2f}  "
                    f"{prediction['pct_change']:.2f}%  "
                    f"{prediction['atr']:.4f} "
                    f"{required_move_pct:.2f}%           "
                    f"{('✅ PASS' if conf_pass else '❌ FAIL'):<12} "
                    f"{('✅ PASS' if atr_pass else '❌ FAIL'):<12} "
                    f"{outcome:<15}"
                )

        print("-"*80)
        print("✅ Diagnostic complete.")
        print("="*80)

    except Exception as e:
        print(f"\n🔥 An error occurred during diagnostics: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_diagnostics()