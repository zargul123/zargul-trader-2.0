import os
import sys
import time
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Suppress all warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.core.data_engine import DataMaster
from scripts.core.analysis_engine import AIAnalyst
from scripts.config import ASSETS, LONG_THRESHOLD, SHORT_THRESHOLD, MIN_CONFIDENCE
from scripts.core.data_monitor import DataHealthMonitor
import random
from scripts.core.guru_wisdom import GuruDetector  # Your existing wisdom detector

class ZargulTrader:
    def __init__(self, train_all=False):
        try:
            print("\n🤖 Initializing Trading System...")
            self.data = DataMaster()
            self.data_monitor = DataHealthMonitor()
            self.ai = AIAnalyst(train_all)
            self.data_monitor = DataHealthMonitor()  # Initialize here
            self.update_interval = 300  # 5 minutes
            self.win_streak = 0  # Track last 5 trades
            print("✅ System ready for trading")
        except Exception as e:
            print(f"\n💥 Initialization Error: {str(e)}")
            raise

    def analyze_asset(self, asset):
        try:
            # Get market data
            df = self.data.get_data(asset, "1h")
            self.data_monitor.log_result(self.data.last_used_source, not df.empty)

            if df.empty or len(df) < 60:
                print(f"⚠️ {asset}: Using synthetic data")
                df = self.data._generate_synthetic_data(asset)

            # Get all prediction types
            main_pred = self.ai.predict(asset, df)
            swing_pred = self.ai.predict_swing(asset)
            scalp_pred = self.ai.predict_scalp(asset)

            # Display main analysis
            print(f"\n📊 {asset} Main Strategy:")
            self._print_prediction(main_pred)

            # Display swing analysis if available
            if swing_pred:
                print(f"\n📈 {asset} Swing Trade:")
                self._print_prediction(swing_pred)
                self._evaluate_trade(swing_pred)

            # Display scalp analysis if available
            if scalp_pred:
                print(f"\n⚡ {asset} Scalp Trade:")
                self._print_prediction(scalp_pred)
                self._evaluate_trade(scalp_pred)

        except Exception as e:
            print(f"\n❌ {asset} error: {str(e)}")

    def _print_prediction(self, prediction):
        """Helper to format prediction output"""
        if not prediction:
            print("│ No trade signal")
            return

        current_price = prediction['current_price'] if isinstance(prediction, dict) and 'current_price' in prediction else prediction['price'] if isinstance(prediction, dict) and 'price' in prediction else 'N/A'
        print(f"│ Current: ${current_price:.2f}")
        print(f"│ Prediction: {prediction['direction'].upper()} {abs(prediction['pct_change']):.2f}%")
        print(f"│ Confidence: {prediction['confidence']:.2%} ★")

        trade_type = prediction['type'] if isinstance(prediction, dict) and 'type' in prediction else None
        if trade_type:
            print(f"│ Type: {trade_type.upper()}")
            hold_time = prediction['hold_time'] if isinstance(prediction, dict) and 'hold_time' in prediction else None
            if hold_time:
                hrs = hold_time / 3600
                print(f"│ Hold Time: ~{hrs:.1f} hours")

    def _evaluate_trade(self, prediction):
        """Evaluate if trade meets thresholds"""
        pct_change = prediction['pct_change']
        confidence = prediction['confidence'] if isinstance(prediction, dict) and 'confidence' in prediction else 0  # Safe!
        direction = prediction['direction'] if isinstance(prediction, dict) and 'direction' in prediction else None  # Safe!

        if confidence >= MIN_CONFIDENCE:
            if (direction == 'long' and pct_change >= LONG_THRESHOLD) or \
               (direction == 'short' and pct_change <= -SHORT_THRESHOLD):
                print(f"└ 🚨 TRADE SIGNAL!")
                self.log_trade(prediction)
            else:
                print(f"└ ⚠️ Needs bigger price move")
        else:
            print(f"└ ⚠️ Needs more confidence")

    def log_trade(self, prediction, success=True):
        if success:
            self.win_streak = min(5, self.win_streak + 1)
        else:
            self.win_streak = max(-5, self.win_streak - 1)

        # Slightly adjust confidence needs
        self.ai.MIN_CONFIDENCE = 0.7 + (0.01 * self.win_streak)

        try:
            # Prepare journal entry
            entry = (
                f"{datetime.now()},"
                f"{prediction['asset']},"
                f"{prediction['direction']},"
                f"{prediction['confidence']:.2f},"
                f"{prediction['pct_change']:.2f},"
                f"{prediction.get('type', 'main')}\n"
            )

            # Write to journal
            with open("trading_journal.csv", "a") as f:
                if os.stat("trading_journal.csv").st_size == 0:
                    f.write("timestamp,asset,direction,confidence,price_change,strategy_type\n")
                f.write(entry)

            print(f"📝 Journal updated: {prediction['asset']} {prediction['direction']}")

        except Exception as e:
            print(f"\n❌ Failed to log trade: {str(e)}")

    def run(self):
        print("\n🚀 Zargul Trader 2.0 - Operational")
        print("==================================================")

        while True:
            for asset in ASSETS:
                # Get market data
                df = self.data.get_data(asset, "1h")
                self.data_monitor.log_result(self.data.last_used_source, not df.empty)

                if df.empty or len(df) < 60:
                    print(f"⚠️ {asset}: Using synthetic data")
                    df = self.data._generate_synthetic_data(asset)

                # Get prediction
                prediction = self.ai.predict(asset, df)
                if not prediction:
                    print(f"⚠️ {asset}: No prediction returned")
                    continue

                # ===== START COPY HERE =====
                if not prediction or not isinstance(prediction, dict) or 'confidence' not in prediction:
                    print("🔇 Invalid prediction - skipping boost")
                else:
                    # Safe demo news - won't crash
                    demo_news = [
                        "Bitcoin ETF approved!",
                        "Whale buying Ethereum"
                    ]

                    try:
                        guru = GuruDetector()
                        wisdom = guru.find_patterns(df, demo_news) if df.shape[0] > 100 else []

                        if wisdom:
                            new_confidence = float(prediction['confidence']) * 1.25
                            prediction['confidence'] = min(0.99, new_confidence)
                            print(f"\n💥 GURU BOOST! Confidence → {prediction['confidence']:.0%} ★")
                    except Exception as e:
                        print(f"🔇 Sentiment skip: {str(e)}")
                # ===== END COPY =====

                # Calculate risk levels
                from scripts.core.risk_engine import RiskManager
                risk = RiskManager()
                stop_loss = risk.calculate_levels(df, prediction)

                print(f"│ 🛑 STOP LOSS: ${stop_loss['sl']:.2f}")
                print(f"│ 🎯 TAKE PROFIT: ${stop_loss['tp']:.2f}")

                # Calculate percentage change
                # The rest of the original analyze_asset function is intentionally left out
                # because this edited snippet is not complete and only provides the code
                # that needs to be inserted in the middle of the analyze_asset function.
                # The original analyze_asset function will handle the remaining logic.

            print(f"\n⏳ Next update in {self.update_interval//60} minutes...")
            time.sleep(self.update_interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-all', action='store_true', help='Retrain all models')
    args = parser.parse_args()

    try:
        trader = ZargulTrader(args.train_all)
        trader.run()
    except Exception as e:
        print(f"\n💥 Critical Failure: {str(e)}")
        sys.exit(1)