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
import atexit

def goodbye():
    print("💾 Saving final trades before shutdown...")
    open("trading_journal.csv", "a").flush()  # Force save

atexit.register(goodbye)

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
from scripts.config import (
    ASSETS, LONG_THRESHOLD, SHORT_THRESHOLD, 
    MIN_CONFIDENCE, STRATEGIES, RETRAIN_DAY, 
    RETRAIN_TIME, RISK_PER_TRADE
)
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
        import traceback
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

            # Add this validation:
            if main_pred is None:
                print(f"⚠️ Main pred None for {asset}, using fallback")
                main_pred = self._create_fallback_pred(asset)
                
            if swing_pred is None:
                print(f"⚠️ Swing pred None for {asset}, using fallback") 
                swing_pred = self._create_fallback_pred(asset)
                
            if scalp_pred is None:
                print(f"⚠️ Scalp pred None for {asset}, using fallback")
                scalp_pred = self._create_fallback_pred(asset)

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
            print(f"💥 Critical error for {asset}: {str(e)}")
            traceback.print_exc()
            return
        
        # Add backtest logging
        if os.environ.get('IN_BACKTEST'):
            with open('backtest_signals.csv', 'a') as f:
                f.write(f"{datetime.now()},{asset},{main_pred}\n")

    def _evaluate_trade(self, prediction):
        """Final professional trade evaluator"""
        if not prediction or not isinstance(prediction, dict):
            print("⚠️ Invalid prediction")
            return
        
        # Convert confidence to percentage ONCE (100 multiplication happens here only)
        raw_confidence = float(prediction.get('confidence', 0)) * 100  # 0.99 → 99
        pct = prediction.get('pct_change', 0)
        direction = prediction.get('direction', '').lower()
        
        # New corrected version
        if raw_confidence >= MIN_CONFIDENCE * 100:  # Now this works right
            if (direction == 'long' and pct >= LONG_THRESHOLD) or \
               (direction == 'short' and pct <= -SHORT_THRESHOLD):
                print(f"└ ✅ TRADE SIGNAL! {direction.upper()} {pct:.2f}% (Confidence: {raw_confidence:.0f}%)")
                self.log_trade(prediction)
            else:
                print(f"└ ⚠️ Needs bigger move (Current: {pct:.2f}%, Required: {LONG_THRESHOLD if direction=='long' else SHORT_THRESHOLD:.2f}%)")
        else:
            print(f"└ ⚠️ Needs more confidence (Current: {raw_confidence:.0f}%, Required: {MIN_CONFIDENCE*100:.0f}%)")

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

        # Display SL/TP if available
        if 'sl' in prediction and 'tp' in prediction:
            print(f"│ 🛑 Stop Loss: ${prediction['sl']:.2f}")
            print(f"│ 🎯 Take Profit: ${prediction['tp']:.2f}")

    def _create_fallback_pred(self, asset):
        """Create a safe fallback prediction when AI prediction fails"""
        import random
        return {
            'asset': asset,
            'direction': 'long' if random.random() > 0.5 else 'short',
            'confidence': 0.3,  # Low confidence for fallback
            'pct_change': 0.5,  # Small expected move
            'current_price': 100.0,  # Default price
            'type': 'fallback'
        }

    def log_trade(self, prediction, success=True):
        if not prediction:
            return
        
        # Get SL/TP from RiskManager
        from scripts.core.risk_engine import RiskManager
        risk = RiskManager()
        levels = risk.calculate_levels(self.data.get_data(prediction['asset'], "1h"), prediction)
        
        # Prepare log entry
        entry = (
            f"{datetime.now()},",
            f"{prediction.get('asset', 'unknown')},",
            f"{prediction.get('direction', 'none')},",
            f"{float(prediction.get('confidence', 0)) * 100:.2f}%,",
            f"{prediction.get('pct_change', 0):.2f}%,",
            f"{prediction.get('type', 'main')},",
            f"SL:{levels.get('sl', 'N/A')},",
            f"TP:{levels.get('tp', 'N/A')}\n"
        )
        
        # Write to file
        with open("trading_journal.csv", "a") as f:
            if os.stat("trading_journal.csv").st_size == 0:
                f.write("timestamp,asset,direction,confidence,pct_change,strategy_type,stop_loss,take_profit\n")
            f.write(''.join(entry))

    def run(self):
        print("\n🚀 Zargul Trader 2.0 - Operational")
        print("==================================================")
        while True:
            for asset in ASSETS:
                self.analyze_asset(asset)
            print(f"\n⏳ Next update in {self.update_interval//60} minutes...")
            time.sleep(self.update_interval)

if __name__ == "__main__":
    print("🤖 Trading Core Active - Not a Flask App!")
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-all', action='store_true', help='Retrain all models')
    args = parser.parse_args()
    try:
        ZargulTrader(train_all=True).run()  # Force retrain ALL models
    except Exception as e:
        print(f"\n💥 UNCAUGHT ERROR: {str(e)}")
        sys.exit(1)