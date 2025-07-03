#!/usr/bin/env python3
import time
from datetime import datetime
from scripts.main import ZargulTrader
from scripts.config import RETRAIN_DAY, RETRAIN_TIME, MIN_CONFIDENCE
from scripts.core.analysis_engine import AIAnalyst

def should_retrain():
    """Check if it's retrain day or confidence is low"""
    now = datetime.now()

    # Weekly schedule check
    if now.strftime("%A").lower() == RETRAIN_DAY and now.strftime("%H:%M") == RETRAIN_TIME:
        print(f"⏰ Scheduled retraining at {RETRAIN_TIME} {RETRAIN_DAY.capitalize()}")
        return True

    # Emergency check (confidence too low)
    analyst = AIAnalyst()
    for symbol in analyst.models:
        last_pred = analyst.predict(symbol)
        if last_pred and last_pred.get('confidence', 0) < MIN_CONFIDENCE:
            print(f"⚠️ Low confidence ({last_pred['confidence']*100:.0f}%) - retraining {symbol}")
            return True

    return False

def main():
    print("🤖 Auto-Trainer Active - Monitoring System")
    while True:
        if should_retrain():
            print("🚀 Starting model retraining...")
            try:
                ZargulTrader(train_all=True).run()
                print("✅ Retraining completed successfully!")
            except Exception as e:
                print(f"❌ Retraining failed: {str(e)}")

        # Check every hour
        time.sleep(3600)

if __name__ == "__main__":
    main()