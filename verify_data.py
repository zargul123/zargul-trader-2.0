import joblib
import pandas as pd
import os

# Set pandas display options for better readability
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 15)

CACHE_FILE = os.path.join('cache', 'BTC-USD_1h_Trending_data.pkl')

def verify():
    print("="*80)
    print(f"🔬 Verifying data snapshot from: {CACHE_FILE}")
    print("="*80)

    if not os.path.exists(CACHE_FILE):
        print(f"❌ ERROR: Cache file not found at '{CACHE_FILE}'.")
        print("Please run the optimizer script first to generate the cache.")
        return

    try:
        # Load the cached DataFrame
        df = joblib.load(CACHE_FILE)
        print(f"✅ Successfully loaded cached data. Total candles: {len(df)}")

        print("\n--- First 5 Rows (Oldest Data) ---")
        print(df.head(5)[['open', 'high', 'low', 'close', 'volume', 'regime']])

        print("\n--- Last 5 Rows (Most Recent Data) ---")
        print(df.tail(5)[['open', 'high', 'low', 'close', 'volume', 'regime']])
        print("\n" + "="*80)
        print("✅ Verification complete.")

    except Exception as e:
        print(f"❌ An error occurred while reading the cache file: {e}")

if __name__ == "__main__":
    verify()
