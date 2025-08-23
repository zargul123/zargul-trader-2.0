import os
import sys
import json
import argparse
import optuna
import pandas as pd
import numpy as np
from copy import deepcopy

# --- Setup Project Environment ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

from scripts.core.backtest_engine import BacktestEngine
from scripts.core.data_engine import DataMaster
from scripts.core.regime_filter import MarketRegimeFilter
from scripts.config import ASSETS, STRATEGIES

# --- Constants ---
OPTIMIZED_RESULTS_FILE = 'optimized_strategies.json'

def load_regime_data(asset, timeframe, regime_type):
    """
    Loads a large dataset and filters it for a specific market regime.
    """
    print(f"\n-- Loading and filtering data for {asset} in {regime_type} regime --")
    data_master = DataMaster()
    # Load a substantial amount of data to ensure enough points for the regime
    df = data_master.get_historical_data(asset, timeframe, "3 years")
    if df is None or df.empty:
        raise ValueError(f"Could not load historical data for {asset}.")

    regime_filter = MarketRegimeFilter()
    df['regime'] = regime_filter.add_regime_column(df)
    
    filtered_df = df[df['regime'] == regime_type].copy()
    
    if len(filtered_df) < 200: # Ensure there's enough data to run a meaningful backtest
        print(f"⚠️ Warning: Only {len(filtered_df)} data points found for the {regime_type} regime.")
        if len(filtered_df) < 50:
             raise ValueError("Insufficient data for backtesting after regime filtering.")

    print(f"-- Found {len(filtered_df)} candles for the '{regime_type}' regime --")
    return filtered_df

def objective(trial, asset, strategy_name, regime_df):
    """
    The core Optuna objective function.
    """
    try:
        # --- 1. Suggest Parameters ---
        # Create a deepcopy to avoid modifying the global config
        strategy_config = deepcopy(STRATEGIES[strategy_name])

        # Define the search space for each parameter
        strategy_config['min_confidence'] = trial.suggest_float('min_confidence', 0.60, 0.95, step=0.01)
        strategy_config['sequence_length'] = trial.suggest_int('sequence_length', 20, 150, step=5)
        strategy_config['atr_threshold_multiplier'] = trial.suggest_float('atr_threshold_multiplier', 0.5, 3.0, step=0.1)
        strategy_config['risk_reward_ratio'] = trial.suggest_float('risk_reward_ratio', 1.0, 5.0, step=0.25)

        # --- 2. Run the Backtest ---
        # The backtest engine will use the temporary config for this run
        backtest_engine = BacktestEngine(regime_df, strategy_config)
        results = backtest_engine.run_backtest()

        # --- 3. Return the Objective Metric ---
        if results is None or results['total_trades'] == 0:
            # Penalize parameter sets that result in no trades or errors
            return -10.0 

        # We want to maximize Sharpe Ratio, but also ensure profitability
        sharpe_ratio = results.get('sharpe_ratio', 0)
        profit_factor = results.get('profit_factor', 0)

        # If Sharpe is negative (losing strategy), penalize it heavily
        if sharpe_ratio < 0:
            return sharpe_ratio * 2 # Make it more negative
        
        # Favor strategies that are both profitable and have good risk-adjusted returns
        # This simple combination helps guide Optuna to more robust solutions
        return sharpe_ratio * (1 + (profit_factor / 10))

    except Exception as e:
        print(f"An error occurred during trial: {e}")
        # Return a large negative number to penalize failing trials
        return -100.0

def run_optimization(asset, strategy, regime, trials):
    """
    Main function to set up and run the Optuna study.
    """
    print("="*80)
    print("🚀 STARTING OPTIMIZATION 🚀")
    print(f"Asset: {asset} | Strategy: {strategy} | Regime: {regime} | Trials: {trials}")
    print("="*80)

    # --- 1. Load and Prepare Data ---
    timeframe = STRATEGIES[strategy]['timeframe']
    try:
        regime_df = load_regime_data(asset, timeframe, regime)
    except ValueError as e:
        print(f"❌ ERROR: {e}")
        return

    # --- 2. Run Optuna Study ---
    study = optuna.create_study(direction="maximize")
    objective_func = lambda trial: objective(trial, asset, strategy, regime_df)
    
    study.optimize(objective_func, n_trials=trials, show_progress_bar=True)

    # --- 3. Process and Save Results ---
    print("\n" + "="*80)
    print("✨ OPTIMIZATION COMPLETE ✨")
    print(f"Best Sharpe Ratio Achieved: {study.best_value:.4f}")
    print("Best Parameters Found:")
    best_params = study.best_params
    for key, value in best_params.items():
        print(f"  - {key}: {value}")
    
    # Load existing results or create new dictionary
    if os.path.exists(OPTIMIZED_RESULTS_FILE):
        with open(OPTIMIZED_RESULTS_FILE, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}

    # Update dictionary with new results
    # This nested structure ensures we don't overwrite other results
    if asset not in all_results:
        all_results[asset] = {}
    if strategy not in all_results[asset]:
        all_results[asset][strategy] = {}
    
    # Store the best params and the final Sharpe ratio
    result_data = best_params
    result_data['sharpe_ratio'] = study.best_value
    all_results[asset][strategy][regime] = result_data

    # Save updated results to file
    with open(OPTIMIZED_RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
        
    print(f"\n✅ Results saved to {OPTIMIZED_RESULTS_FILE}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automated Strategy Optimization using Optuna.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--asset',
        type=str,
        required=True,
        help="The crypto asset to optimize (e.g., 'BTC-USD')."
    )
    parser.add_argument(
        '--strategy',
        type=str,
        required=True,
        choices=STRATEGIES.keys(),
        help="The strategy to optimize (e.g., 'main', 'swing')."
    )
    parser.add_argument(
        '--regime',
        type=str,
        required=True,
        choices=['Trending', 'Ranging'],
        help="The market regime to optimize for."
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=100,
        help="The number of optimization attempts Optuna should run."
    )
    args = parser.parse_args()

    run_optimization(args.asset, args.strategy, args.regime, args.trials)
