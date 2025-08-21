
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

def analyze_feature_importance(strategy_to_run, train_models=True):
    """
    Trains the AI models and uses SHAP to analyze and visualize feature importance.
    This is a diagnostic script and does not affect live trading operations.
    """
    print("="*80)
    print("🔬 INITIATING FEATURE IMPORTANCE ANALYSIS 🔬")
    print("="*80)
    
    # --- 1. Initialize and Train Models ---
    print(f"\n[PHASE 1/3] Model Training Phase")
    try:
        if train_models:
            print("  - Training all models as requested...")
            ai_analyst = AIAnalyst(train_all=True)
            print("  - ✅ Model training complete.")
        else:
            print("  - Skipping training and loading existing models...")
            ai_analyst = AIAnalyst(train_all=False)
            print("  - ✅ Models loaded successfully.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR during model training: {e}")
        print("Aborting analysis.")
        return

    # Determine which strategies to run
    if strategy_to_run == 'all':
        strategies_to_analyze = ['main', 'scalp', 'swing']
    else:
        strategies_to_analyze = [strategy_to_run]

    print(f"\nTargeting strategies: {', '.join(s.upper() for s in strategies_to_analyze)}")

    for strategy_name in strategies_to_analyze:
        print("\n" + "="*80)
        print(f"🔬 ANALYZING STRATEGY: {strategy_name.upper()} 🔬")
        print("="*80)

        # --- 2. Prepare Data for SHAP Analysis ---
        print(f"\n[PHASE 2/3] Preparing data for SHAP analysis ({strategy_name.upper()})...")
        data_master = DataMaster()
        background_data = {}
        test_data = {}
        features_dict = {}

        for symbol in ASSETS:
            print(f"  - Fetching and preparing data for {symbol}...")
            timeframe = STRATEGIES[strategy_name]['timeframe']
            sequence_length = STRATEGIES[strategy_name]['sequence_length']
            
            df = data_master.get_training_data(symbol, timeframe)
            if df is None or df.empty:
                print(f"  - ⚠️ Could not get data for {symbol}. Skipping.")
                continue

            df_features = ai_analyst._align_df_features(df)
            features = df_features.columns.tolist()
            features_dict[symbol] = features

            # Ensure the model and scaler exist before proceeding
            if symbol not in ai_analyst.scalers or strategy_name not in ai_analyst.scalers[symbol]:
                print(f"  - ⚠️ Scaler for {symbol} ({strategy_name}) not found. Skipping.")
                continue
            
            scaler = ai_analyst.scalers[symbol][strategy_name]
            scaled_data = scaler.transform(df_features.values)

            X, _ = [], []
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i - sequence_length:i])
            X = np.array(X)

            if X.shape[0] < 30: # Ensure there's enough data for sampling
                print(f"  - ⚠️ Insufficient data samples ({X.shape[0]}) for {symbol} ({strategy_name}). Skipping.")
                continue

            background_data[symbol] = shap.sample(X, 20)
            test_data[symbol] = shap.sample(X, 10)
            
        print(f"✅ Data preparation complete for {strategy_name.upper()}.")

        # --- 3. Run SHAP Analysis and Generate Plots ---
        print(f"\n[PHASE 3/3] Running SHAP analysis and generating plots ({strategy_name.upper()})...")
        os.makedirs('feature_analysis', exist_ok=True)

        for symbol in ASSETS:
            if symbol not in background_data:
                continue
            
            try:
                print(f"\n  --- Analyzing Model for: {symbol} ({strategy_name.upper()}) ---")
                
                if not (symbol in ai_analyst.models and strategy_name in ai_analyst.models[symbol]):
                    print(f"    - No model found for {symbol} ({strategy_name.upper()}). Skipping.")
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
                
                n_components = min(20, background_flat.shape[0] - 1)
                if n_components < 1:
                    print("    - Not enough background samples to perform PCA. Skipping SHAP analysis.")
                    continue

                print(f"    - Compressing features to {n_components} components.")
                pca = PCA(n_components=n_components)
                background_pca = pca.fit_transform(background_flat)
                test_pca = pca.transform(test_flat)

                def shap_predictor(x_pca):
                    x_flat = pca.inverse_transform(x_pca)
                    x_reshaped = x_flat.reshape(x_flat.shape[0], sequence_length, len(features))
                    preds = model.predict(x_reshaped, verbose=0)
                    if isinstance(preds, (tuple, list)):
                        preds = preds[0]
                    return preds[:, 1]

                print("    - Creating PermutationExplainer on PCA-compressed data...")
                explainer = shap.PermutationExplainer(shap_predictor, background_pca, max_evals=2*n_components+1)

                print(f"    - Calculating SHAP values for {test_samples} samples...")
                
                shap_values_list = []
                for i in tqdm(range(test_samples), desc=f"Explaining {symbol} ({strategy_name})"):
                    explanation = explainer(test_pca[i:i+1])
                    shap_values_list.append(explanation.values)
                    gc.collect()

                if not shap_values_list:
                    print("    - No SHAP values were calculated. Skipping plot.")
                    continue

                shap_values = np.vstack(shap_values_list)

                # --- NEW: Map PCA Importance back to Original Features ---
                print("    - Mapping PCA SHAP values back to original features...")
                
                abs_shap_values = np.abs(shap_values)
                pc_importance = np.mean(abs_shap_values, axis=0)
                pca_loadings = pca.components_
                feature_importance_timestep = np.dot(pc_importance, np.abs(pca_loadings))
                
                n_features = len(features_dict[symbol])
                aggregated_feature_importance = feature_importance_timestep.reshape(sequence_length, n_features).sum(axis=0)

                feature_names = features_dict[symbol]
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': aggregated_feature_importance
                }).sort_values(by='importance', ascending=False)

                print(f"    - Top 5 most important features for {symbol} ({strategy_name.upper()}):")
                print(importance_df.head(5).to_string(index=False))
                
                # --- PLOTTING THE NEW, INTERPRETABLE RESULTS ---
                plt.figure(figsize=(12, 10))
                plt.barh(importance_df['feature'], importance_df['importance'], color='skyblue')
                plt.xlabel("Mean Absolute SHAP Value (Calculated from PCA)")
                plt.ylabel("Feature")
                plt.title(f'Feature Importance for {symbol} ({strategy_name.upper()})')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                
                plot_path = f"feature_analysis/{symbol}_{strategy_name}_feature_importance.png"
                plt.savefig(plot_path)
                plt.close()
                
                print(f"    ✅ Saved INTERPRETABLE feature importance plot to: {plot_path}")
            
            except Exception as e:
                print(f"❌ An unexpected error occurred while analyzing {symbol} ({strategy_name.upper()}): {e}")
                import traceback
                traceback.print_exc()
            
            finally:
                if 'model' in locals(): del model
                if 'explainer' in locals(): del explainer
                if 'shap_values' in locals(): del shap_values
                gc.collect()
                print(f"    - Cleanup complete for {symbol} ({strategy_name.upper()}).")

    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print("Review the generated .png files in the 'feature_analysis' directory.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run SHAP feature importance analysis on trained models.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--strategy',
        type=str,
        required=True,
        choices=['main', 'scalp', 'swing', 'all'],
        help="The trading strategy to analyze (e.g., 'main', 'scalp', 'swing') or 'all' for all strategies."
    )
    parser.add_argument(
        '--no-train',
        action='store_true',
        help="Skip the model training phase and use existing models.\n             This is useful for quickly re-running analysis after a code change."
    )
    args = parser.parse_args()

    # If --no-train is specified, train_models will be False.
    analyze_feature_importance(strategy_to_run=args.strategy, train_models=not args.no_train)
