#!/usr/bin/env python3
"""
Simple script to auto-train BTC-USD main model if missing
"""
import os
import sys

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.core.analysis_engine import AIAnalyst

def train_btc_main_if_missing():
    """
    Checks if BTC-USD main model exists, trains it if missing
    """
    model_files = [
        'trained_models/BTC-USD_main_model.h5',
        'trained_models/BTC-USD_main_scaler.joblib', 
        'trained_models/BTC-USD_main_calibrator.joblib'
    ]
    
    missing_files = [f for f in model_files if not os.path.exists(f)]
    
    if missing_files:
        print("🔍 BTC-USD main model files missing:")
        for f in missing_files:
            print(f"   - {f}")
        
        print("\n🤖 Auto-training BTC-USD main model...")
        try:
            # This will automatically train if files are missing
            analyst = AIAnalyst(symbol='BTC-USD', strategy_type='main')
            print("✅ BTC-USD main model training completed!")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    else:
        print("✅ BTC-USD main model already exists - no training needed")
    
    return True

if __name__ == "__main__":
    print("="*60)
    print("BTC-USD MAIN MODEL AUTO-TRAINER")
    print("="*60)
    train_btc_main_if_missing()
    print("="*60)