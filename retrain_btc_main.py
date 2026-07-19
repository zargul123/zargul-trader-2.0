import sys
import os
import glob
import argparse

from dotenv import load_dotenv
load_dotenv()

# Add project root to path to allow script imports
sys.path.insert(0, os.getcwd())

from scripts.core.analysis_engine import AIAnalyst

# On Windows, multiprocessing (used by pandas_ta indicator calculation)
# re-imports this script in child processes. Without this guard, every
# child re-runs the whole retraining flow: deleting model files again and
# hammering the TwelveData API until it returns 429 rate-limit errors.
# Argument parsing also lives inside the guard so spawned workers never
# touch sys.argv during import.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Focused retraining of a single model.")
    parser.add_argument('--symbol', type=str, default="BTC-USD", help="Asset to retrain (default BTC-USD)")
    parser.add_argument('--strategy', type=str, default="main", help="Strategy to retrain (default main)")
    args = parser.parse_args()
    SYMBOL_TO_TRAIN = args.symbol
    STRATEGY_TO_TRAIN = args.strategy

    print(f'--- FOCUSED RETRAINING SCRIPT ---')
    print(f'Targeting: {SYMBOL_TO_TRAIN} ({STRATEGY_TO_TRAIN})')

    # 1. Delete the specific old model files to force retraining
    model_path = f'trained_models/{SYMBOL_TO_TRAIN}_{STRATEGY_TO_TRAIN}_model.h5'
    scaler_path = f'trained_models/{SYMBOL_TO_TRAIN}_{STRATEGY_TO_TRAIN}_scaler.joblib'
    calibrator_path = f'trained_models/{SYMBOL_TO_TRAIN}_{STRATEGY_TO_TRAIN}_calibrator.joblib'

    for path in [model_path, scaler_path, calibrator_path]:
        if os.path.exists(path):
            print(f'   - Deleting old file: {path}')
            os.remove(path)

    # 2. Initialize the AIAnalyst
    # It will find the specific model is missing and train ONLY that one.
    print('\nInitializing AI Analyst to trigger retraining...')
    analyst = AIAnalyst()

    print(f'\n✅ Focused retraining complete for {SYMBOL_TO_TRAIN} ({STRATEGY_TO_TRAIN}).')
