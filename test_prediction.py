
#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.core.analysis_engine import AIAnalyst
from scripts.core.data_engine import DataMaster

def test_prediction():
    print("🧪 Testing Single Prediction")
    print("=" * 40)
    
    try:
        # Initialize components
        print("🔧 Initializing AI Analyst...")
        analyst = AIAnalyst()
        
        print("📊 Fetching BTC-USD data...")
        data_master = DataMaster()
        df = data_master.get_data("BTC-USD", "1h")
        
        if df.empty:
            print("⚠️ No data found, using synthetic data")
            df = data_master._generate_synthetic_data("BTC-USD")
        
        print(f"✅ Got {len(df)} data points")
        print(f"📈 Latest close price: ${df['close'].iloc[-1]:.2f}")
        
        # Run prediction
        print("\n🤖 Running prediction...")
        pred = analyst.predict("BTC-USD", df)
        
        # Display results
        print("\n🎯 PREDICTION RESULTS:")
        print("=" * 30)
        print(f"Final Direction: {pred['direction']} | Change: {pred['pct_change']:.2f}%")
        print(f"Confidence: {pred['confidence']:.2%}")
        print(f"Current Price: ${pred['current_price']:.2f}")
        print(f"Predicted Price: ${pred['price']:.2f}")
        print(f"Strategy Type: {pred['type']}")
        
        return pred
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_prediction()
