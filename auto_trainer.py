#!/usr/bin/env python3
import time
import sys
import os
from datetime import datetime, timedelta

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.main import ZargulTrader
from scripts.config import AUTO_TRAIN_SCHEDULE, STRATEGIES, ASSETS
from scripts.core.analysis_engine import AIAnalyst

def should_retrain():
    """
    Checks if retraining is needed based on a schedule or low AI confidence.
    """
    # 1. Check if scheduled retraining is enabled and the time is right
    if AUTO_TRAIN_SCHEDULE['enabled']:
        now = datetime.now()
        is_day = now.strftime("%A").lower() == AUTO_TRAIN_SCHEDULE['day_of_week']
        is_time = now.strftime("%H:%M") == AUTO_TRAIN_SCHEDULE['time_utc']
        if is_day and is_time:
            print(f"⏰ It's {now.strftime('%A %H:%M')}, scheduled retraining is due.")
            return True

    # 2. Check if any model's confidence is critically low
    # Note: This is a simplified check. A real-world scenario might be more complex.
    print("Checking model confidence levels...")
    analyst = AIAnalyst() # This will load existing models, not retrain them
    
    # We need some recent data to make a test prediction
    from scripts.core.data_engine import DataMaster
    data_master = DataMaster()
    
    min_confidence_threshold = STRATEGIES['main']['min_confidence']

    for symbol in ASSETS:
        if symbol in analyst.models:
            df = data_master.get_data(symbol, STRATEGIES['main']['timeframe'])
            if not df.empty:
                test_pred = analyst.predict(symbol, df)
                if test_pred and test_pred.get('confidence', 1.0) < min_confidence_threshold:
                    print(f"⚠️ EMERGENCY RETRAIN: Confidence for {symbol} ({test_pred['confidence']:.2f}) is below threshold ({min_confidence_threshold:.2f}).")
                    return True
    
    return False

def main():
    """
    The main loop for the auto-trainer script.
    """
    print("🤖 Auto-Trainer Initialized. Monitoring for retraining conditions...")
    while True:
        if should_retrain():
            print("\n" + "="*50)
            print("🚀 Retraining conditions met. Initiating full model training.")
            print("="*50 + "\n")
            try:
                # Initialize the trader with train_all=True and run_once=True
                # This will force retraining and then exit, handing control back to the auto-trainer.
                trader = ZargulTrader(train_all=True, run_once=True)
                trader.run()
                print("\n" + "="*50)
                print("✅ Retraining cycle completed successfully!")
                print("="*50 + "\n")
            except Exception as e:
                print(f"❌ An error occurred during the retraining cycle: {e}")
        
        # Wait for an hour before checking again.
        print(f"Next check at {datetime.now() + timedelta(hours=1):%Y-%m-%d %H:%M}")
        time.sleep(3600)

if __name__ == "__main__":
    main()
