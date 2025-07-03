
import os
import sys
import pandas as pd
from scripts.core.data_engine import DataMaster
from scripts.core.analysis_engine import AIAnalyst

def verify_environment():
    print("\n🔍 Environment Check:")
    print(f"OMP_NUM_THREADS: {os.getenv('OMP_NUM_THREADS')}")
    print(f"TF_CPP_MIN_LOG_LEVEL: {os.getenv('TF_CPP_MIN_LOG_LEVEL')}")

def verify_data_loading():
    print("\n📊 Data Loading Test:")
    try:
        dm = DataMaster()
        df = dm.get_data("BTC-USD")
        print("\nFirst 5 rows of BTC-USD data:")
        print(df.head())
    except Exception as e:
        print(f"❌ Data loading failed: {str(e)}")

def main():
    print("\n🔬 Running Verification Tests")
    print("=" * 50)
    verify_environment()
    verify_data_loading()

if __name__ == "__main__":
    main()
