#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import os
import sys
import time
import argparse
from datetime import datetime
from dotenv import load_dotenv
import warnings

# --- Path Setup ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Safety Imports ---
from scripts.core.safety import armor_get
def safe_get(d, key, default=None): 
    return armor_get(d, key, default)

# --- Suppress Warnings ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore")

# --- Core Imports ---
from scripts.core.data_engine import DataMaster
from scripts.core.analysis_engine import AIAnalyst
from scripts.config import ASSETS, LONG_THRESHOLD, SHORT_THRESHOLD, MIN_CONFIDENCE
from scripts.core.data_monitor import DataHealthMonitor
from scripts.core.guru_wisdom import GuruDetector
from scripts.core.risk_engine import RiskManager

class ZargulTrader:
    def __init__(self, train_all=False):
        try:
            print("\n🤖 Initializing Trading System...")
            self.data = DataMaster()
            self.ai = AIAnalyst(train_all)
            self.data_monitor = DataHealthMonitor()
            self.update_interval = 300  # 5 minutes
            self.win_streak = 0
            print("✅ System ready for trading")
        except Exception as e:
            print(f"\n💥 Initialization Error: {str(e)}")
            raise

    def analyze_asset(self, asset):
        if not isinstance(asset, str):
            print(f"⚡ Invalid asset type: {type(asset)}")
            return

        try:
            # --- Data Fetching ---
            df = self.data.get_data(asset, "1h")
            self.data_monitor.log_result(str(self.data.last_used_source), bool(not df.empty))

            if df.empty or len(df) < 60:
                print(f"⚠️ {asset}: Using synthetic data")
                df = self.data._generate_synthetic_data(asset)

            # --- Predictions ---
            main_pred = self.ai.predict(asset, df)
            swing_pred = self.ai.predict_swing(asset)
            scalp_pred = self.ai.predict_scalp(asset)

            # --- Sentiment Boost ---
            if isinstance(main_pred, dict) and armor_get(main_pred, 'confidence'):
                demo_news = ["Bitcoin ETF approved!", "Whale buying Ethereum"]
                try:
                    guru = GuruDetector()
                    wisdom = guru.find_patterns(df, demo_news) if df.shape[0] > 100 else []
                    if wisdom:
                        main_pred['confidence'] = min(0.99, armor_get(main_pred, 'confidence') * 1.25)
                        print(f"\n💎 GURU BOOST! {wisdom[0]['source']} strategy")
                except Exception as e:
                    print(f"🔇 Sentiment skip: {str(e)}")

            # --- Display Predictions ---
            print(f"\n📊 {asset} Main Strategy:")
            self._print_prediction(main_pred)

            print(f"\n📈 {asset} Swing Trade:")
            self._print_prediction(swing_pred)
            self._evaluate_trade(swing_pred)

            print(f"\n⚡ {asset} Scalp Trade:")
            self._print_prediction(scalp_pred)
            self._evaluate_trade(scalp_pred)

        except Exception as e:
            print(f"\n❌ {asset} error: {str(e)}")

    def _evaluate_trade(self, prediction):
        """Final professional trade evaluator"""
        if not prediction or not isinstance(prediction, dict):
            print("⚠️ Invalid prediction")
            return
        
        # Extract values safely
        pct = prediction.get('pct_change', 0)
        conf = prediction.get('confidence', 0)  # Already in percentage
        direction = prediction.get('direction', '').lower()
        
        # Check trade conditions
        if conf >= MIN_CONFIDENCE * 100:  # Compare against threshold
            if (direction == 'long' and pct >= LONG_THRESHOLD) or \
               (direction == 'short' and pct <= -SHORT_THRESHOLD):
                print(f"└ ✅ TRADE SIGNAL! {direction.upper()} {pct:.2f}% (Confidence: {conf:.0f}%)")
                self.log_trade(prediction)
            else:
                print(f"└ ⚠️ Needs bigger move (Current: {pct:.2f}%, Required: {LONG_THRESHOLD:.2f}%)")
        else:
            print(f"└ ⚠️ Needs more confidence (Current: {conf:.0f}%, Required: {MIN_CONFIDENCE*100:.0f}%)")

    def _print_prediction(self, prediction):
        """Clean formatted output"""
        if not prediction:
            print("│ No trade signal")
            return

        current_price = armor_get(prediction, 'current_price', armor_get(prediction, 'price', 'N/A'))
        print(f"│ Current: ${float(current_price):.2f}")
        print(f"│ Direction: {prediction['direction'].upper()}")
        print(f"│ Confidence: {float(prediction['confidence']) * 100:.0f}%")  # Force percentage

        if 'pct_change' in prediction:
            change_sign = '+' if prediction['direction'] == 'long' else ''
            print(f"│ Predicted Move: {change_sign}{prediction['pct_change']:.2f}%")

        if 'hold_time' in prediction:
            print(f"│ Hold Time: ~{prediction['hold_time']/3600:.1f} hours")

    def log_trade(self, prediction, success=True):
        if not prediction or not isinstance(prediction, dict):
            print("⚠️ Invalid trade log")
            return

        # Win/loss tracking
        self.win_streak = min(5, self.win_streak + 1) if success else max(-5, self.win_streak - 1)
        self.ai.MIN_CONFIDENCE = 0.7 + (0.01 * self.win_streak)

        try:
            with open("trading_journal.csv", "a") as f:
                if os.stat("trading_journal.csv").st_size == 0:
                    f.write("timestamp,asset,direction,confidence,price_change,strategy_type\n")
                f.write(
                    f"{datetime.now()},"
                    f"{armor_get(prediction, 'asset', 'unknown')},"
                    f"{armor_get(prediction, 'direction', 'none')},"
                    f"{float(armor_get(prediction, 'confidence', 0)) * 100:.2f},"
                    f"{armor_get(prediction, 'pct_change', 0):.2f},"
                    f"{armor_get(prediction, 'type', 'main')}\n"
                )
            print(f"📝 Journal updated: {prediction.get('asset', '?')} {prediction.get('direction', '?')}")
        except Exception as e:
            print(f"\n❌ Failed to log trade: {str(e)}")

    def run(self):
        print("\n🚀 Zargul Trader 2.0 - Operational")
        print("==================================================")
        while True:
            for asset in ASSETS:
                self.analyze_asset(asset)
            print(f"\n⏳ Next update in {self.update_interval//60} minutes...")
            time.sleep(self.update_interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-all', action='store_true', help='Retrain all models')
    args = parser.parse_args()
    try:
        ZargulTrader(args.train_all).run()
    except Exception as e:
        print(f"\n💥 UNCAUGHT ERROR: {str(e)}")
        sys.exit(1)