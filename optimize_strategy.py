import os
import sys
import json
import argparse

from dotenv import load_dotenv
load_dotenv()

import optuna
import pandas as pd
import numpy as np
from copy import deepcopy
import joblib # Using joblib for efficient caching of DataFrames

# --- Setup Project Environment ---
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from scripts.backtest.backtest_engine import BacktestEngine
from scripts.core.analysis_engine import AIAnalyst
from scripts.core.data_engine import DataMaster
from scripts.config import ASSETS, STRATEGIES, REGIME_CONFIG

# --- Constants ---
OPTIMIZED_RESULTS_FILE = 'optimized_strategies.json'
CACHE_DIR = 'cache' # Directory to store pre-processed data

def _calculate_shannon_entropy(series, window):
    """Helper function to calculate entropy for a rolling window."""
    bins = pd.cut(series, bins=[-np.inf, -0.005, 0, 0.005, np.inf], labels=False, right=False)
    counts = np.bincount(bins, minlength=4)
    probabilities = counts / len(series)
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))

def calculate_historical_regimes(df: pd.DataFrame) -> pd.Series:
    """Calculates the market regime for each row in a historical DataFrame."""
    print("-- Calculating historical market regimes...")
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
    print("-- Regime calculation complete.")
    return regimes

def load_regime_data(asset, timeframe, regime_type):
    """
    Loads a large dataset and filters it for a specific market regime.
    Now includes caching to dramatically speed up subsequent runs.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_filename = f"{asset}_{timeframe}_{regime_type}_data.pkl"
    cache_filepath = os.path.join(CACHE_DIR, cache_filename)

    if os.path.exists(cache_filepath):
        print(f"\n-- Loading pre-processed data from cache: {cache_filepath} --")
        filtered_df = joblib.load(cache_filepath)
        print("-- Cached data loaded successfully. --")
        return filtered_df

    print(f"\n-- No cache found. Performing one-time data preparation for {asset} in {regime_type} regime --")
    data_master = DataMaster()
    df = data_master.get_training_data(asset, timeframe, days=1095)
    if df is None or df.empty:
        raise ValueError(f"Could not load historical data for {asset}.")

    df['regime'] = calculate_historical_regimes(df)
    df.dropna(inplace=True)
    filtered_df = df[df['regime'] == regime_type].copy()

    if len(filtered_df) < 200:
        print(f"⚠️ Warning: Only {len(filtered_df)} data points found for the {regime_type} regime.")
        if len(filtered_df) < 50:
             raise ValueError("Insufficient data for backtesting after regime filtering.")

    print(f"-- Saving prepared data to cache: {cache_filepath} --")
    joblib.dump(filtered_df, cache_filepath)
    
    print(f"-- Found {len(filtered_df)} candles for the '{regime_type}' regime --")
    return filtered_df

def objective(trial, asset, strategy_name, regime, regime_df, backtest_engine):
    """The core Optuna objective function."""
    try:
        # --- 1. Create a temporary config for this trial ---
        # Load the base strategy config
        temp_strategy_config = deepcopy(STRATEGIES[asset][strategy_name])
        
        # --- 2. Define the search space and apply it to the correct regime ---
        # These parameters will be optimized ONLY for the specified regime

        # This is the "Smartly-Tightened" search space based on backtest log analysis.
        regime_params = {
            'min_confidence': trial.suggest_float('min_confidence', 0.75, 0.95, step=0.01),
            'atr_threshold_multiplier': trial.suggest_float('atr_threshold_multiplier', 1.0, 3.5, step=0.1),
            'tp_atr_multiplier': trial.suggest_float('tp_atr_multiplier', 2.5, 4.5, step=0.1),
            'sl_atr_multiplier': trial.suggest_float('sl_atr_multiplier', 1.2, 2.2, step=0.05)
        }
        
        # Update the temporary config with the trial parameters for the target regime
        temp_strategy_config[regime].update(regime_params)
        
        # --- 3. Run the backtest with the temporary configuration ---
        results = backtest_engine.run_backtest(
            symbol=asset,
            strategy_type=strategy_name,
            days=365, # Use a fixed, recent 365-day window for robust optimization
            data_df=None, # Pass None to use the live data fetching for the specified window
            temp_strategy_config=temp_strategy_config,
            temp_risk_config=None
        )

        # --- 4. Evaluate the results ---
        if results is None or results['total_trades'] < 10: # Ensure a minimum number of trades
            return -10.0

        sharpe_ratio = results.get('sharpe_ratio', 0)
        profit_factor = results.get('profit_factor', 0)

        if sharpe_ratio < 0:
            return -5.0 + sharpe_ratio

        return (sharpe_ratio * 0.7) + (profit_factor * 0.3)

    except Exception as e:
        print(f"An error occurred during trial: {e}")
        import traceback
        traceback.print_exc()
        return -100.0

def run_optimization(asset, strategy, regime, trials):
    """Main function to set up and run the Optuna study."""
    print("="*80)
    print("🚀 STARTING OPTIMIZATION 🚀")
    print(f"Asset: {asset} | Strategy: {strategy} | Regime: {regime} | Trials: {trials}")
    print("="*80)

    try:
        # CORRECTED: Look up timeframe in the new nested structure
        timeframe = STRATEGIES[asset][strategy]['timeframe']
        regime_df = load_regime_data(asset, timeframe, regime)
    except KeyError:
        print(f"❌ ERROR: Strategy '{strategy}' not found for asset '{asset}' in config.py.")
        return
    except ValueError as e:
        print(f"❌ ERROR: {e}")
        return

    print(f"\n-- Pre-initializing AI Analyst and Backtest Engine for {asset} ({strategy}) --")
    analyst = AIAnalyst(symbol=asset, strategy_type=strategy)
    backtest_engine = BacktestEngine(analyst=analyst)
    print("-- Engines are ready. --")

    study_name = f"{asset}_{strategy}_{regime}"
    storage_name = f"sqlite:///{study_name}.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize"
    )
    
    # Pass the asset name and regime to the objective function
    objective_func = lambda trial: objective(trial, asset, strategy, regime, regime_df, backtest_engine)
    
    completed_trials = len(study.trials)
    if completed_trials >= trials:
        print(f"Study already has {completed_trials} trials. No new trials will be run.")
    else:
        remaining_trials = trials - completed_trials
        print(f"Resuming study. {completed_trials} trials already complete. Running {remaining_trials} more.")
        study.optimize(objective_func, n_trials=remaining_trials, show_progress_bar=True)

    print("\n" + "="*80)
    print("✨ OPTIMIZATION COMPLETE ✨")
    print(f"Best Score Achieved: {study.best_value:.4f}")
    print("Best Parameters Found:")
    best_params = study.best_params
    for key, value in best_params.items():
        print(f"  - {key}: {value}")
    
    if os.path.exists(OPTIMIZED_RESULTS_FILE):
        with open(OPTIMIZED_RESULTS_FILE, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}

    if asset not in all_results:
        all_results[asset] = {}
    if strategy not in all_results[asset]:
        all_results[asset][strategy] = {}
    
    result_data = best_params
    result_data['score'] = study.best_value
    all_results[asset][strategy][regime] = result_data

    with open(OPTIMIZED_RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
        
    print(f"\n✅ Results saved to {OPTIMIZED_RESULTS_FILE}")
    print(f"💡 Study progress saved to database: {study_name}.db")
    print("="*80)

if __name__ == "__main__":
    # --- CORRECTED: Create a unique list of all possible strategy names for argparse ---
    all_strategies = set()
    for asset_strategies in STRATEGIES.values():
        for strategy_name in asset_strategies.keys():
            all_strategies.add(strategy_name)
    strategy_choices = sorted(list(all_strategies))
    # --------------------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Automated Strategy Optimization using Optuna.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--asset', type=str, required=True, choices=ASSETS, help="e.g., 'BTC-USD'")
    parser.add_argument('--strategy', type=str, required=True, choices=strategy_choices, help="e.g., 'main', 'scalp'")
    parser.add_argument('--regime', type=str, required=True, choices=['Trending', 'Ranging', 'Chaotic'], help="The market regime to optimize for.")
    parser.add_argument('--trials', type=int, default=100, help="The TOTAL number of trials the study should have.")
    args = parser.parse_args()

    run_optimization(args.asset, args.strategy, args.regime, args.trials)
