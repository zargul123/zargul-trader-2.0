import sys
import os
import numpy as np
import pandas as pd
import random

# Add project root to path to allow script imports
sys.path.insert(0, os.getcwd())

from scripts.core.analysis_engine import AIAnalyst
from scripts.core.data_engine import DataMaster
from scripts.config import STRATEGIES

def run_inference_check():
    """
    Performs a forward pass on a few random data points to check the model's
    raw internal predictions.
    """
    print("="*80)
    print("🔬 RUNNING INFERENCE CHECK 🔬")
    print("="*80)

    try:
        # --- Configuration ---
        symbol = "BTC-USD"
        strategy = "main"
        num_checks = 5
        # -------------------

        # 1. Initialize components in a focused way
        print(f"   - Loading AI Analyst for {symbol} ({strategy})...")
        analyst = AIAnalyst(symbol=symbol, strategy_type=strategy)
        data = DataMaster()

        # 2. Load a recent chunk of data
        print("   - Loading last 100 bars of historical data...")
        timeframe = STRATEGIES[symbol][strategy]['timeframe']
        df = data.get_data(symbol, timeframe, limit=100)
        if df is None or df.empty or len(df) < 50:
            print("   - ❌ ERROR: Could not load sufficient data. Aborting.")
            return
        
        print(f"   - ✅ Data loaded successfully.")

        # 3. Run the inference check on random data points
        sequence_length = STRATEGIES[symbol][strategy]['sequence_length']
        
        # Pick 5 random starting points for our sequences
        random_indices = random.sample(range(sequence_length, len(df)), num_checks)

        print("\n" + "="*80)
        print("📊 INFERENCE CHECK REPORT 📊")
        print("="*80)
        print(f"{ 'Timestamp':<22} | {'Raw Prediction (B,S,H)':<25} | {'Predicted Class':<18} | {'Confidence':<12}")
        print("-" * 80)

        for i in random_indices:
            window_data = df.iloc[i - sequence_length : i]
            timestamp = window_data.index[-1]

            # --- Perform the core prediction ---
            # This block is a simplified version of the logic in AIAnalyst.predict()
            model = analyst.models[symbol][strategy]
            scaler = analyst.scalers[symbol][strategy]
            calibrator = analyst.calibrators[symbol][strategy]

            df_aligned = analyst._align_df_features(window_data)
            scaled_data = scaler.transform(df_aligned.values)
            input_data = scaled_data.reshape(1, sequence_length, df_aligned.shape[1])
            
            raw_prediction = model.predict(input_data, verbose=0)[0]
            calibrated_probs = calibrator.predict_proba([raw_prediction])[0]
            predicted_class_index = np.argmax(calibrated_probs)
            confidence = calibrated_probs[predicted_class_index]
            # --- End of prediction logic ---

            # Format the raw prediction for readability
            raw_pred_str = f"[{raw_prediction[0]:.2f}, {raw_prediction[1]:.2f}, {raw_prediction[2]:.2f}]"
            
            # Map index to a human-readable class
            class_map = {0: "BUY", 1: "SELL", 2: "HOLD"}
            predicted_class_str = class_map[predicted_class_index]

            print(f"{str(timestamp):<22} | {raw_pred_str:<25} | {predicted_class_str:<18} | {confidence:.2%}")

        print("="*80)

    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_inference_check()
