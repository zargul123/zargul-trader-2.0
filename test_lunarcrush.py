import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.core.social_analyzer import SocialAnalyzer, get_ttl_hash
from scripts.config import LUNARCRUSH_CONFIG

def test_api_connection():
    """
    A simple test to verify the LunarCrush API connection and key.
    """
    print("="*50)
    print("🔬 Testing LunarCrush API Connection...")
    print("="*50)

    try:
        # 1. Initialize the analyzer
        analyzer = SocialAnalyzer(
            api_key=LUNARCRUSH_CONFIG['api_key'],
            base_url=LUNARCRUSH_CONFIG['api_url']
        )
        print("✅ SocialAnalyzer initialized successfully.")
        print(f"   - API Key Loaded: {'Yes' if LUNARCRUSH_CONFIG['api_key'] else 'No'}")

        # 2. Fetch metrics for a test asset
        test_asset = "BTC-USD"
        print(f"\nFetching social metrics for {test_asset}...")
        metrics = analyzer.get_social_metrics(test_asset, ttl_hash=get_ttl_hash())

        # 3. Check and print the results
        if not metrics:
            print("\n❌ TEST FAILED: The API did not return any data.")
            print("   - Please double-check that your API key in Replit Secrets is correct.")
            print("   - The KEY should be 'LUNARCRUSH_API_KEY'.")
            return

        print("\n✅ TEST SUCCESSFUL! Received data from LunarCrush.")
        print("-------------------------------------------------")
        for key, value in metrics.items():
            if key in LUNARCRUSH_CONFIG['metrics_to_use']:
                print(f"   - {key}: {value}")
        print("-------------------------------------------------")
        print("\nYour API key is working correctly.")

    except Exception as e:
        print(f"\n❌ TEST FAILED with an error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_api_connection()
