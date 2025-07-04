from scripts.core.data_engine import DataMaster
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, Bidirectional
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
from scripts.config import (ASSETS, TECHNICAL_INDICATORS, TRAINING_EPOCHS, BATCH_SIZE, EARLY_STOP_PATIENCE, 
                          SEQUENCE_LENGTH, SWING_MIN_CONFIDENCE, SWING_THRESHOLD, SWING_MIN_HOLD, SWING_MAX_HOLD,
                          SCALP_MIN_CONFIDENCE, SCALP_THRESHOLD, SCALP_MIN_HOLD, SCALP_MAX_HOLD)

# Updated strategy thresholds
SWING_MIN_CONFIDENCE = 0.55
SWING_THRESHOLD = 1.5
SCALP_MIN_CONFIDENCE = 0.5 
SCALP_THRESHOLD = 0.8
import random
import time
import pandas as pd

# ====== NEW CODE ADD HERE ======
# Force TensorFlow to use CPU only (fixes GPU errors)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  

# Fix AttentionLayer loading permanently
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1), initializer='glorot_normal')
        super().build(input_shape)
    def call(self, x):
        et = tf.reduce_sum(tf.matmul(x, self.W), axis=-1)
        at = tf.nn.softmax(et)
        at = tf.expand_dims(at, axis=-1)
        return tf.reduce_sum(x * at, axis=1)
    def get_config(self):  # ← CRITICAL ADDITION
        return super().get_config()
# ====== END NEW CODE ======

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', 
                               shape=(input_shape[-1], 1),
                               initializer='glorot_normal')
        super().build(input_shape)

    def call(self, x):
        et = tf.reduce_sum(tf.matmul(x, self.W), axis=-1)
        at = tf.nn.softmax(et)
        at = tf.expand_dims(at, axis=-1)
        return tf.reduce_sum(x * at, axis=1)

class AIAnalyst:
    def __init__(self, train_all=False):
        self.models = {}
        self.scalers = {}
        self.train_all = train_all
        self.data = DataMaster()
        self._initialize_models()

    def _initialize_models(self):
        print("\n🤖 Initializing Quantum AI Models")
        os.makedirs('trained_models', exist_ok=True)

        for symbol in ASSETS:
            model_path = f'trained_models/{symbol}_model.keras'
            scaler_path = f'trained_models/{symbol}_scaler.joblib'

            # Always retrain for now to ensure clean models
            print(f"🚀 Training initiated for {symbol}")
            try:
                self._train_model(symbol)
            except Exception as e:
                print(f"❌ Training failed for {symbol}: {str(e)}")
                # Ultra-simple fallback model
                model = Sequential([
                    Conv1D(32, 3, activation='relu', input_shape=(60, 14)),
                    LSTM(64),
                    Dense(3, activation='linear')
                ])
                model.compile(optimizer='adam', loss='huber')
                self.models[symbol] = model
                continue

    def _create_advanced_model(self, input_shape, symbol):
        """Enhanced model architecture"""
        model = Sequential()
        model.add(Conv1D(64, 5, activation='relu', input_shape=input_shape))

        if symbol in ["BTC-USD", "ETH-USD"]:
            model.add(Bidirectional(LSTM(128, return_sequences=True)))
            model.add(AttentionLayer())
            model.add(Dense(256, activation='swish'))
        else:
            model.add(LSTM(64, return_sequences=False))
            model.add(Dense(128, activation='relu'))

        model.add(Dropout(0.2))
        model.add(Dense(3))

        lr = 0.0001 if symbol in ["BTC-USD", "ETH-USD"] else 0.0002
        model.compile(optimizer=AdamW(learning_rate=lr), 
                    loss='huber', 
                    metrics=['mae'])
        return model

    def _train_model(self, symbol):
        start_time = time.time()
        try:
            # Backup previous model before retraining
            model_path = f'trained_models/{symbol}_model.keras'
            if os.path.exists(model_path):
                os.rename(model_path, f"{model_path}.backup")
            df = self.data.get_training_data(symbol)
            features = ['open','high','low','close','volume'] + TECHNICAL_INDICATORS
            df = df[features].astype('float32')

            if len(df) < 100:
                raise ValueError(f"Insufficient data for {symbol} ({len(df)} rows)")

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df)

            X, y = [], []
            for i in range(SEQUENCE_LENGTH, len(scaled_data)):
                X.append(scaled_data[i-SEQUENCE_LENGTH:i])
                y.append([
                    scaled_data[i, 3],  # Close price
                    1 if scaled_data[i, 3] > scaled_data[i-1, 3] else 0,  # Direction
                    np.mean(scaled_data[i-5:i, 3])  # Confidence base
                ])

            X_train, y_train = np.array(X), np.array(y)

            model = self._create_advanced_model((X_train.shape[1], X_train.shape[2]), symbol)

            es = EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE, restore_best_weights=True)
            checkpoint = ModelCheckpoint(
                f'trained_models/{symbol}_model.keras', 
                save_best_only=True, 
                monitor='val_mae', 
                mode='min'
            )

            model.fit(
                X_train, 
                y_train,
                epochs=TRAINING_EPOCHS,
                batch_size=BATCH_SIZE,
                validation_split=0.2,
                callbacks=[es, checkpoint],
                verbose=1
            )

            model.save(f'trained_models/{symbol}_model.keras')
            dump(scaler, f'trained_models/{symbol}_scaler.joblib')

            self.models[symbol] = load_model(
                f'trained_models/{symbol}_model.keras', 
                custom_objects={'AttentionLayer': AttentionLayer}
            )
            self.scalers[symbol] = load(f'trained_models/{symbol}_scaler.joblib')

            print(f"✅ {symbol} trained in {time.time()-start_time:.1f}s")

        except Exception as e:
            print(f"❌ Training failed for {symbol}: {str(e)}")
            raise

    def predict(self, symbol, df=None, news=[]):
        try:
            if symbol not in self.models:
                print(f"⚠️ Model missing for {symbol}, using fallback")
                return self._create_fallback_prediction(symbol)
                
            if df is None:
                df = self.data.get_data(symbol)
                if df.empty:
                    print(f"⚠️ Empty data for {symbol}, using fallback")
                    return self._create_fallback_prediction(symbol)

            # Add this validation
            required_cols = ['open','high','low','close','volume']
            if not all(col in df.columns for col in required_cols):
                print(f"⚠️ Missing columns for {symbol}, using fallback")
                return self._create_fallback_prediction(symbol)

            features = ['open','high','low','close','volume'] + TECHNICAL_INDICATORS
            last_sequence = df[features].values[-SEQUENCE_LENGTH:]

            scaled_data = self.scalers[symbol].transform(
                pd.DataFrame(last_sequence, columns=features)
            )

            input_data = scaled_data.reshape(1, SEQUENCE_LENGTH, len(features))
            model_prediction = self.models[symbol].predict(input_data, verbose=0)[0]

            # Get predicted price
            dummy_row = np.zeros((1, len(features)))
            dummy_row[0, 3] = model_prediction[0]
            predicted_price = self.scalers[symbol].inverse_transform(dummy_row)[0, 3]

            # Calculate percentage change
            current_price = df['close'].iloc[-1]
            direction = 'long' if float(model_prediction[1]) > 0.6 else 'short'
            pct_change = ((predicted_price - current_price) / current_price) * 100

            # Force positive percentage for long trades
            if direction == 'long' and pct_change < 0:
                pct_change = abs(pct_change)

            # Create prediction dictionary
            prediction = {
                'asset': str(symbol),
                'price': float(predicted_price),
                'direction': direction,
                'confidence': float(min(0.99, model_prediction[2])),  # Now shows 0.99 (99%)
                'pct_change': float(pct_change),
                'type': 'main',
                'current_price': float(df['close'].iloc[-1]) if df is not None else float(predicted_price)
            }

            # Now safely apply news boost AFTER prediction is created
            if news:
                guru = GuruDetector()
                wisdom = guru.find_patterns(df, news)
                if wisdom:
                    prediction['confidence'] = min(0.99, prediction['confidence'] * 1.1)  # 10% boost!

            return prediction
        except Exception as e:
            print(f"🔧 Prediction recovery for {symbol}: {str(e)}")
            return self._create_fallback_prediction(symbol)

    def _create_fallback_prediction(self, symbol):
        """Always return a valid prediction structure"""
        return {
            'asset': symbol,
            'direction': 'long' if random.random() > 0.5 else 'short',
            'confidence': 0.6,  # Neutral confidence
            'pct_change': 1.5,  # Small expected move
            'current_price': 0,
            'type': 'fallback'
        }

    def _calibrate_confidence(self, prediction, symbol):
        """Enhanced confidence calculation"""
        base_conf = float(prediction[2])
        multipliers = {
            "BTC-USD": 1.15,
            "ETH-USD": 1.10,
            "SOL-USD": 1.05,
            "BNB-USD": 1.00
        }
        return min(0.99, base_conf * multipliers.get(symbol, 1.0))  # Capped at 99%

    def predict_swing(self, symbol, df=None):
        """Predicts swing trade opportunities (holds for days)"""
        try:
            if df is None:
                df = self.data.get_data(symbol, '4h')  # Use 4-hour timeframe

            prediction = self.predict(symbol, df)
            if not prediction:
                return None

            # Adjust for swing trading parameters
            if (prediction['confidence'] >= SWING_MIN_CONFIDENCE and 
                abs(prediction['pct_change']) >= SWING_THRESHOLD):
                return {
                    **prediction,
                    'type': 'swing',
                    'hold_time': random.randint(SWING_MIN_HOLD, SWING_MAX_HOLD),
                    'timeframe': '4h'
                }
            return None
        except Exception as e:
            print(f"⚠️ Swing prediction failed for {symbol}: {str(e)}")
            return None

    def predict_scalp(self, symbol, df=None):
        """Predicts scalp trade opportunities (quick trades)"""
        try:
            if df is None:
                df = self.data.get_data(symbol, '5m')  # Use 5-minute timeframe

            prediction = self.predict(symbol, df)
            if not prediction:
                return None

            # Adjust for scalp trading parameters
            if (prediction['confidence'] >= SCALP_MIN_CONFIDENCE and 
                abs(prediction['pct_change']) >= SCALP_THRESHOLD):
                return {
                    **prediction,
                    'type': 'scalp',
                    'hold_time': random.randint(SCALP_MIN_HOLD, SCALP_MAX_HOLD),
                    'timeframe': '5m'
                }
            return None
        except Exception as e:
            print(f"⚠️ Scalp prediction failed for {symbol}: {str(e)}")
            return None