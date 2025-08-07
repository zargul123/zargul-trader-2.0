
import os
import sys
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import psutil
from tqdm import tqdm

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
    print("\n[PHASE 1/3] Forcing model retraining with new data and features...")
    try:
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
        strategy_name = 'main'
        timeframe = STRATEGIES[strategy_name]['timeframe']
        sequence_length = STRATEGIES[strategy_name]['sequence_length']
        
        df = data_master.get_training_data(symbol, timeframe)
        if df is None or df.empty:
            print(f"  - ⚠️ Could not get data for {symbol}. Skipping.")
            continue

        features = ['open', 'high', 'low', 'close', 'volume'] + [indi for indi in TECHNICAL_INDICATORS if indi in df.columns]
        features_dict[symbol] = features
        df_features = df[features].astype('float32')
        
        scaler = ai_analyst.scalers[symbol][strategy_name]
        scaled_data = scaler.transform(df_features.values)

        X, _ = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i - sequence_length:i])
        X = np.array(X)

        # Use a smaller subset of data to avoid memory issues
        background_data[symbol] = shap.sample(X, 20)
        test_data[symbol] = shap.sample(X, 10)
        
    print("✅ Data preparation complete.")

    # --- 3. Run SHAP Analysis and Generate Plots ---
    print("\n[PHASE 3/3] Running SHAP analysis and generating plots...")
    os.makedirs('feature_analysis', exist_ok=True)

    for symbol in ASSETS:
        if symbol not in background_data:
            continue
            
        print(f"\n  --- Analyzing Model for: {symbol} ---")
        
        strategy_name = 'main'
        if symbol in ai_analyst.models and strategy_name in ai_analyst.models[symbol]:
            print(f"    - Strategy: {strategy_name}")
            model = ai_analyst.models[symbol][strategy_name]
            
            sequence_length = STRATEGIES[strategy_name]['sequence_length']
            features = features_dict[symbol]
            
            # 1. Reduce computational load
            background_samples = 10
            test_samples = 5
            
            if len(background_data[symbol]) < background_samples:
                background_samples = len(background_data[symbol])
            if len(test_data[symbol]) < test_samples:
                test_samples = len(test_data[symbol])

            background_flat = background_data[symbol][:background_samples].reshape(background_samples, -1)
            test_flat = test_data[symbol][:test_samples].reshape(test_samples, -1)

            # 2. Define efficient prediction wrapper
            def model_predict(x):
                x_reshaped = x.reshape((x.shape[0], sequence_length, len(features)))
                batch_size = 5
                predictions = []
                for i in range(0, x_reshaped.shape[0], batch_size):
                    batch = x_reshaped[i:i+batch_size]
                    preds = model.predict(batch, verbose=0)
                    if isinstance(preds, (tuple, list)):
                        predictions.append(preds[0])
                    else:
                        predictions.append(preds)
                return np.vstack(predictions)

            # 3. Create KernelExplainer with fewer samples
            print("    - Creating optimized KernelExplainer...")
            explainer = shap.KernelExplainer(model_predict, background_flat, link="identity")

            # 4. Calculate SHAP values with progress tracking
            print(f"    - Memory usage before SHAP: {psutil.virtual_memory().percent}%")
            print(f"    - Calculating SHAP values for {test_samples} samples (this may take a few minutes)...")
            
            shap_values_list = []
            for i in tqdm(range(test_samples), desc=f"Explaining {symbol}"):
                sv = explainer.shap_values(test_flat[i:i+1], nsamples=50)
                shap_values_list.append(sv)
            shap_values = np.vstack(shap_values_list)

            # 5. Process results
            shap_values_3d = shap_values.reshape((test_samples, sequence_length, len(features)))
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
