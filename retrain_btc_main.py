import sys
import os
import glob

# Add project root to path to allow script imports
sys.path.insert(0, os.getcwd())

from scripts.core.analysis_engine import AIAnalyst

# --- Configuration ---
SYMBOL_TO_TRAIN = "BTC-USD"
STRATEGY_TO_TRAIN = "main"
# -------------------

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
