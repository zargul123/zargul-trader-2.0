import sys
import os
import numpy as np

# Add project root to path to allow script imports
sys.path.insert(0, os.getcwd())

from scripts.core.analysis_engine import AIAnalyst
from scripts.core.data_engine import DataMaster

def run_label_diagnostics():
    """
    Loads the training data and runs only the labeling function to analyze
    its output distribution.
    """
    print("="*80)
    print("🔬 RUNNING LABEL DIAGNOSTICS 🔬")
    print("="*80)

    try:
        # 1. Initialize the necessary components
        # We only need the analyst for its helper functions, no need to load models
        analyst = AIAnalyst()
        data = DataMaster()

        # 2. Load the full training dataset
        print("   - Loading 3 years of historical data for BTC-USD (1h)...")
        df = data.get_training_data('BTC-USD', '1h', days=1095)
        if df is None or df.empty:
            print("   - ❌ ERROR: Could not load data. Aborting.")
            return
        
        print(f"   - ✅ Data loaded successfully with {len(df)} candles.")

        # 3. Align features (necessary before labeling)
        df_features = analyst._align_df_features(df)

        # 4. Generate the labels using the function we want to test
        print("   - Generating labels using the current logic in AIAnalyst...")
        labels_one_hot = analyst._create_forward_looking_labels(df_features)
        print("   - ✅ Label generation complete.")

        # 5. Analyze and report the distribution
        buy_count = np.sum(labels_one_hot[:, 0])
        sell_count = np.sum(labels_one_hot[:, 1])
        hold_count = np.sum(labels_one_hot[:, 2])
        total_count = len(labels_one_hot)

        if total_count == 0:
            print("   - ❌ ERROR: No labels were generated.")
            return

        buy_pct = (buy_count / total_count) * 100
        sell_pct = (sell_count / total_count) * 100
        hold_pct = (hold_count / total_count) * 100

        print("\n" + "="*80)
        print("📊 LABEL DISTRIBUTION REPORT 📊")
        print("="*80)
        print(f"   - Total Labels: {total_count}")
        print(f"   - Buy Signals:  {int(buy_count)} ({buy_pct:.2f}%)")
        print(f"   - Sell Signals: {int(sell_count)} ({sell_pct:.2f}%)")
        print(f"   - Hold Signals: {int(hold_count)} ({hold_pct:.2f}%)")
        print("="*80)

    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_label_diagnostics()
