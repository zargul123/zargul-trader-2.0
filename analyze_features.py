
import os
import sys
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import psutil
import gc
import time
from tqdm import tqdm
from sklearn.decomposition import PCA

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
    
    # --- 1. Initialize and Train Models ---
    print(f"\n[PHASE 1/3] Forcing model retraining for all assets...")
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

        background_data[symbol] = shap.sample(X, 20)
        test_data[symbol] = shap.sample(X, 10)
        
    print("✅ Data preparation complete.")

    # --- 3. Run SHAP Analysis and Generate Plots ---
    print("\n[PHASE 3/3] Running SHAP analysis and generating plots...")
    os.makedirs('feature_analysis', exist_ok=True)

    for symbol in ASSETS:
        if symbol not in background_data:
            continue
        
        try:
            print(f"\n  --- Analyzing Model for: {symbol} ---")
            
            strategy_name = 'main'
            if not (symbol in ai_analyst.models and strategy_name in ai_analyst.models[symbol]):
                print(f"    - No model found for {symbol}. Skipping.")
                continue

            model = ai_analyst.models[symbol][strategy_name]
            sequence_length = STRATEGIES[strategy_name]['sequence_length']
            features = features_dict[symbol]
            
            test_samples = min(5, len(test_data[symbol]))
            background_samples = min(10, len(background_data[symbol]))

            background_flat = background_data[symbol][:background_samples].reshape(background_samples, -1)
            test_flat = test_data[symbol][:test_samples].reshape(test_samples, -1)

            # --- PCA FALLBACK IMPLEMENTATION ---
            print("    - Using PCA to compress features for SHAP analysis.")
            
            # Dynamically set n_components to be less than the number of samples
            n_components = min(20, background_flat.shape[0] - 1)
            if n_components < 1:
                print("    - Not enough background samples to perform PCA. Skipping SHAP analysis.")
                continue

            print(f"    - Compressing features to {n_components} components.")
            pca = PCA(n_components=n_components)
            background_pca = pca.fit_transform(background_flat)
            test_pca = pca.transform(test_flat)

            def pca_predict(x_pca):
                x_flat = pca.inverse_transform(x_pca)
                x_reshaped = x_flat.reshape(-1, sequence_length, len(features))
                preds = model.predict(x_reshaped, verbose=0)
                return preds[0] if isinstance(preds, (tuple, list)) else preds

            print("    - Creating PermutationExplainer on PCA-compressed data...")
            explainer = shap.PermutationExplainer(pca_predict, background_pca, max_evals=2*n_components+1)

            print(f"    - Calculating SHAP values for {test_samples} samples...")
            
            shap_values_list = []
            for i in tqdm(range(test_samples), desc=f"Explaining {symbol}"):
                explanation = explainer(test_pca[i:i+1])
                shap_values_list.append(explanation.values)
                gc.collect()

            if not shap_values_list:
                print("    - No SHAP values were calculated. Skipping plot.")
                continue

            shap_values = np.vstack(shap_values_list)
            
            pc_feature_names = [f'PC_{i+1}' for i in range(shap_values.shape[1])]
            shap_df = pd.DataFrame(shap_values, columns=pc_feature_names)

            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_df.values, feature_names=pc_feature_names, plot_type="bar", show=False)
            plt.title(f'SHAP PCA Feature Importance for {symbol} ({strategy_name.upper()})')
            plt.xlabel("Average SHAP Value (Impact on model output)")
            plt.tight_layout()
            
            plot_path = f"feature_analysis/{symbol}_{strategy_name}_pca_feature_importance.png"
            plt.savefig(plot_path)
            plt.close()
            
            print(f"    ✅ Saved PCA feature importance plot to: {plot_path}")
        
        except Exception as e:
            print(f"❌ An unexpected error occurred while analyzing {symbol}: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            del model, explainer, shap_values
            gc.collect()
            print(f"    - Cleanup complete for {symbol}.")

    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print("Review the generated .png files in the 'feature_analysis' directory.")

if __name__ == "__main__":
    analyze_feature_importance()
