import os
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, Bidirectional
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load

from scripts.core.data_engine import DataMaster
from scripts.config import ASSETS, TECHNICAL_INDICATORS, TRAINING_CONFIG, STRATEGIES

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
    def get_config(self):
        return super().get_config()

class AIAnalyst:
    def __init__(self, train_all=False):
        self.models = {}
        self.scalers = {}
        self.train_all = train_all
        self.data = DataMaster()
        self._initialize_models()

    def _initialize_models(self):
        print("\n🤖 Initializing AI Analyst...")
        os.makedirs('trained_models', exist_ok=True)
        for symbol in ASSETS:
            model_path = f'trained_models/{symbol}_model.keras'
            scaler_path = f'trained_models/{symbol}_scaler.joblib'
            if self.train_all or not os.path.exists(model_path) or not os.path.exists(scaler_path):
                print(f"🔧 No pre-trained model found for {symbol} or retraining was requested. Starting training...")
                try:
                    self._train_model(symbol)
                except Exception as e:
                    print(f"❌ Training failed for {symbol}: {e}")
            else:
                print(f"🧠 Loading pre-trained model for {symbol}...")
                try:
                    self.models[symbol] = load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
                    self.scalers[symbol] = load(scaler_path)
                    print(f"   - ✅ Model for {symbol} loaded successfully.")
                except Exception as e:
                    print(f"❌ Failed to load model for {symbol}: {e}")

    def _create_advanced_model(self, input_shape, symbol):
        model = Sequential([
            Conv1D(64, 5, activation='relu', input_shape=input_shape),
            Bidirectional(LSTM(128, return_sequences=True)) if symbol in ["BTC-USD", "ETH-USD"] else LSTM(64, return_sequences=False),
            AttentionLayer() if symbol in ["BTC-USD", "ETH-USD"] else Dropout(0.2),
            Dense(256, activation='swish') if symbol in ["BTC-USD", "ETH-USD"] else Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(3)
        ])
        lr = 0.0001 if symbol in ["BTC-USD", "ETH-USD"] else 0.0002
        model.compile(optimizer=AdamW(learning_rate=lr), loss='huber', metrics=['mae'])
        return model

    def _train_model(self, symbol):
        start_time = time.time()
        try:
            df = self.data.get_training_data(symbol)
            if df is None or df.empty:
                raise ValueError(f"Cannot train {symbol}, no data available.")

            # **ROBUSTNESS FIX**: Define feature order and ensure it exists.
            features = ['open', 'high', 'low', 'close', 'volume'] + [indi for indi in TECHNICAL_INDICATORS if indi in df.columns]
            df_features = df[features].astype('float32')

            # Fix: Use values to avoid feature name issues with MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df_features.values)

            sequence_length = STRATEGIES['main']['sequence_length']
            X, y = [], []
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i - sequence_length:i])
                y.append([scaled_data[i, 3], 1 if scaled_data[i, 3] > scaled_data[i-1, 3] else 0, np.mean(scaled_data[i-5:i, 3])])
            X_train, y_train = np.array(X), np.array(y)

            model = self._create_advanced_model((X_train.shape[1], X_train.shape[2]), symbol)
            es = EarlyStopping(monitor='val_loss', patience=TRAINING_CONFIG['early_stop_patience'], restore_best_weights=True)
            checkpoint = ModelCheckpoint(f'trained_models/{symbol}_model.keras', save_best_only=True, monitor='val_mae', mode='min')

            model.fit(X_train, y_train, epochs=TRAINING_CONFIG['epochs'], batch_size=TRAINING_CONFIG['batch_size'], validation_split=0.2, callbacks=[es, checkpoint], verbose=1)

            model.save(f'trained_models/{symbol}_model.keras')
            dump(scaler, f'trained_models/{symbol}_scaler.joblib')
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            print(f"✅ Model for {symbol} trained and saved in {time.time() - start_time:.1f}s.")
        except Exception as e:
            print(f"❌ Training process for {symbol} failed: {e}")
            raise

    def predict(self, symbol, df):
        if symbol not in self.models or symbol not in self.scalers:
            return None
        try:
            # **ROBUSTNESS FIX**: Use the same feature order as training.
            features = ['open', 'high', 'low', 'close', 'volume'] + [indi for indi in TECHNICAL_INDICATORS if indi in df.columns]
            sequence_length = STRATEGIES['main']['sequence_length']

            last_sequence_df = df[features].tail(sequence_length).astype('float32')
            if len(last_sequence_df) < sequence_length:
                return None # Not enough data to form a sequence

            # Use numpy arrays consistently
            last_sequence_values = last_sequence_df.values
            scaled_data = self.scalers[symbol].transform(last_sequence_values)
            input_data = scaled_data.reshape(1, sequence_length, len(features))
            raw_prediction = self.models[symbol].predict(input_data, verbose=0)[0]

            # Create dummy row for inverse transform
            dummy_row = np.zeros((1, len(features)))
            dummy_row[0, 3] = raw_prediction[0]
            predicted_price = self.scalers[symbol].inverse_transform(dummy_row)[0, 3]
            current_price = float(df['close'].iloc[-1])
            pct_change = ((predicted_price - current_price) / current_price) * 100
            direction = 'long' if predicted_price > current_price else 'short'

            # Fix: Handle potential NaN values in volatility calculation
            price_changes = df['close'].pct_change().dropna()
            if len(price_changes) > 0:
                price_volatility = price_changes.std() * 100
                # Fix: Handle NaN volatility
                if pd.isna(price_volatility) or price_volatility == 0:
                    price_volatility = 1.0
            else:
                price_volatility = 1.0

            confidence = min(0.95, max(0.30, 0.5 + (abs(pct_change) / (price_volatility * 2))))
            if abs(pct_change) < 0.05:
                direction = 'hold'
                confidence *= 0.5

            return {
                'asset': symbol, 
                'timestamp': pd.to_datetime('now', utc=True), 
                'price': float(predicted_price), 
                'direction': direction, 
                'confidence': float(confidence), 
                'pct_change': float(pct_change), 
                'current_price': current_price
            }
        except Exception as e:
            print(f"💥 Prediction failed for {symbol}: {e}")
            return None

    def predict_swing(self, symbol, df):
        prediction = self.predict(symbol, df)
        if prediction:
            prediction['type'] = 'swing'
        return prediction

    def predict_scalp(self, symbol, df):
        prediction = self.predict(symbol, df)
        if prediction:
            prediction['type'] = 'scalp'
        return prediction