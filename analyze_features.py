import os
import sys
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# --- Setup Project Environment ---
# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
warnings.filterwarnings("ignore")

from scripts.core.analysis_engine import AIAnalyst
from scripts.core.data_engine import DataMaster
from scripts.config import ASSETS, STRATEGIES, TECHNICAL_INDICATORS

def analyze_feature_importance():
    """
    Trains the AI models and uses SHAP to analyze and visualize feature importance.
    This is a diagnostic script and does not affect live trading operations.
    """
    print("="*80)
    print("🔬 INITIATING FEATURE IMPORTANCE ANALYSIS 🔬")
    print("="*80)
    print("This script will retrain the AI models on the latest data and configuration.")
    print("It will then analyze the trained models to determine which features are most predictive.")
    print("-" * 80)

    # --- 1. Initialize and Train Models ---
    # We instantiate AIAnalyst with train_all=True to force retraining with our new
    # 3-year dataset and elite features.
    print("\n[PHASE 1/3] Forcing model retraining with new data and features...")
    try:
        # Set train_all=True to ensure we're analyzing the new models
        ai_analyst = AIAnalyst(train_all=True)
        print("✅ Model training complete.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR during model training: {e}")
        print("Aborting analysis.")
        return

    # --- 2. Prepare Data for SHAP Analysis ---
    print("\n[PHASE 2/3] Preparing data for SHAP analysis...")
    data_master = DataMaster()
    background_data = {}
    test_data = {}
    features_dict = {}

    for symbol in ASSETS:
        print(f"  - Fetching and preparing data for {symbol}...")
        # Use the 'main' strategy config for data preparation
        strategy_name = 'main'
        timeframe = STRATEGIES[strategy_name]['timeframe']
        sequence_length = STRATEGIES[strategy_name]['sequence_length']
        
        # Fetch the full dataset
        df = data_master.get_training_data(symbol, timeframe)
        if df is None or df.empty:
            print(f"  - ⚠️ Could not get data for {symbol}. Skipping.")
            continue

        # Prepare data exactly as the AI would for training
        features = ['open', 'high', 'low', 'close', 'volume'] + [indi for indi in TECHNICAL_INDICATORS if indi in df.columns]
        features_dict[symbol] = features
        df_features = df[features].astype('float32')
        
        # Use the already-fitted scaler from the AIAnalyst
        scaler = ai_analyst.scalers[symbol][strategy_name]
        scaled_data = scaler.transform(df_features.values)

        X, _ = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i - sequence_length:i])
        X = np.array(X)

        # We'll use a subset of the data for SHAP to keep it fast
        # A random sample of 100 data points is a good background set
        background_data[symbol] = shap.sample(X, 100)
        # And another 50 points to explain predictions for
        test_data[symbol] = shap.sample(X, 50)
        
    print("✅ Data preparation complete.")

    # --- 3. Run SHAP Analysis and Generate Plots ---
    print("\n[PHASE 3/3] Running SHAP analysis and generating plots...")
    os.makedirs('feature_analysis', exist_ok=True) # Directory to save plots

    for symbol in ASSETS:
        if symbol not in background_data:
            continue
            
        print(f"\n  --- Analyzing Model for: {symbol} ---")
        
        strategy_name = 'main'
        if symbol in ai_analyst.models and strategy_name in ai_analyst.models[symbol]:
            print(f"    - Strategy: {strategy_name}")
            model = ai_analyst.models[symbol][strategy_name]
            
            # Get parameters needed for reshaping inside the wrapper
            sequence_length = STRATEGIES[strategy_name]['sequence_length']
            features = features_dict[symbol]
            
            def model_predict(x):
                """Wrapper function that reshapes input and returns predictions"""
                x_reshaped = x.reshape((x.shape[0], sequence_length, len(features)))
                preds = model.predict(x_reshaped, verbose=0)
                if isinstance(preds, (tuple, list)):
                    return preds[0]
                return preds

            background_flat = background_data[symbol].reshape(background_data[symbol].shape[0], -1)
            
            print("    - Creating KernelExplainer...")
            explainer = shap.KernelExplainer(model_predict, background_flat)

            test_flat = test_data[symbol].reshape(test_data[symbol].shape[0], -1)

            print(f"    - Calculating SHAP values for {len(test_flat)} samples...")
            shap_values = explainer.shap_values(test_flat)

            shap_values_3d = shap_values.reshape((test_data[symbol].shape[0], sequence_length, len(features)))
            shap_values_avg = np.abs(shap_values_3d).mean(axis=1)

            if len(features) == shap_values_avg.shape[1]:
                shap_df = pd.DataFrame(shap_values_avg, columns=features)

                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_df.values, feature_names=features, plot_type="bar", show=False)
                plt.title(f'SHAP Feature Importance for {symbol} ({strategy_name.upper()})')
                plt.xlabel("Average SHAP Value (Impact on model output)")
                plt.tight_layout()
                
                plot_path = f"feature_analysis/{symbol}_{strategy_name}_feature_importance.png"
                plt.savefig(plot_path)
                plt.close()
                
                print(f"    ✅ Saved feature importance plot to: {plot_path}")
            else:
                print(f"    ❌ Error: Mismatch between feature names ({len(features)}) and SHAP values ({shap_values_avg.shape[1]}).")

    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print("Review the generated .png files in the 'feature_analysis' directory.")

if __name__ == "__main__":
    analyze_feature_importance()