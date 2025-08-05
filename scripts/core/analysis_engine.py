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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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
        self.models = {symbol: {} for symbol in ASSETS}
        self.scalers = {symbol: {} for symbol in ASSETS}
        self.calibrators = {symbol: {} for symbol in ASSETS} # For Platt Scaling
        self.prediction_functions = {symbol: {} for symbol in ASSETS}
        self.train_all = train_all
        self.data = DataMaster()
        self._initialize_models()

    def _initialize_models(self):
        print("\n🤖 Initializing AI Analyst...")
        os.makedirs('trained_models', exist_ok=True)
        for symbol in ASSETS:
            strategies_for_symbol = ['main', 'scalp']
            if symbol == 'BTC-USD':
                strategies_for_symbol.append('btc-swing')
            else:
                strategies_for_symbol.append('swing')
            
            for strategy_name in strategies_for_symbol:
                model_path = f'trained_models/{symbol}_{strategy_name}_model.h5'
                scaler_path = f'trained_models/{symbol}_{strategy_name}_scaler.joblib'
                calibrator_path = f'trained_models/{symbol}_{strategy_name}_calibrator.joblib'
                
                # Train if any of the three files are missing or if train_all is True
                if self.train_all or not all(os.path.exists(p) for p in [model_path, scaler_path, calibrator_path]):
                    print(f"🔧 No pre-trained model/scaler/calibrator for {symbol} ({strategy_name}) or retraining requested.")
                    try:
                        self._train_model(symbol, strategy_name)
                    except Exception as e:
                        print(f"❌ Training failed for {symbol} ({strategy_name}): {e}")
                else:
                    print(f"🧠 Loading pre-trained model, scaler, and calibrator for {symbol} ({strategy_name})...")
                    try:
                        self.models[symbol][strategy_name] = load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
                        self.scalers[symbol][strategy_name] = load(scaler_path)
                        self.calibrators[symbol][strategy_name] = load(calibrator_path)
                        print(f"   - ✅ Components for {symbol} ({strategy_name}) loaded successfully.")
                    except Exception as e:
                        print(f"❌ Failed to load components for {symbol} ({strategy_name}): {e}")

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
        optimizer = AdamW(learning_rate=lr, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model

    def _train_model(self, symbol, strategy_name):
        start_time = time.time()
        print(f"   - Starting training for {symbol} ({strategy_name})...")
        try:
            df = self.data.get_training_data(symbol, STRATEGIES[strategy_name]['timeframe'])
            if df is None or df.empty:
                raise ValueError(f"Cannot train {symbol} ({strategy_name}), no data available.")

            features = ['open', 'high', 'low', 'close', 'volume'] + [indi for indi in TECHNICAL_INDICATORS if indi in df.columns]
            df_features = df[features].astype('float32')

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df_features.values)

            sequence_length = STRATEGIES[strategy_name]['sequence_length']
            X, y = [], []
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i - sequence_length:i])
                # Target y is [price, direction (1=up, 0=down), avg_price]
                y.append([scaled_data[i, 3], 1 if scaled_data[i, 3] > scaled_data[i-1, 3] else 0, np.mean(scaled_data[i-5:i, 3])])
            X, y = np.array(X), np.array(y)

            # Split data for main model training and for calibrator training
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            model = self._create_advanced_model((X_train.shape[1], X_train.shape[2]), symbol)
            es = EarlyStopping(monitor='val_loss', patience=TRAINING_CONFIG['early_stop_patience'], restore_best_weights=True)
            
            model_path = f'trained_models/{symbol}_{strategy_name}_model.h5' # Use .h5 format
            scaler_path = f'trained_models/{symbol}_{strategy_name}_scaler.joblib'
            calibrator_path = f'trained_models/{symbol}_{strategy_name}_calibrator.joblib'
            
            checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_mae', mode='min', save_format='h5') # Add save_format

            model.fit(X_train, y_train, epochs=TRAINING_CONFIG['epochs'], batch_size=TRAINING_CONFIG['batch_size'], validation_data=(X_val, y_val), callbacks=[es, checkpoint], verbose=1)

            # --- PLATT SCALING CALIBRATOR TRAINING ---
            print(f"   - Training confidence calibrator for {symbol} ({strategy_name})...")
            # Get the raw, uncalibrated outputs from the main model on the validation set
            val_predictions_raw = model.predict(X_val)
            
            # The input for the calibrator is the model's raw output for the direction prediction
            calibrator_X = val_predictions_raw[:, 1].reshape(-1, 1) 
            # The target is the actual direction from the validation set
            calibrator_y = y_val[:, 1]

            calibrator = LogisticRegression(solver='liblinear')
            calibrator.fit(calibrator_X, calibrator_y)
            print("   - ✅ Calibrator trained.")

            # --- SAVE ALL COMPONENTS ---
            # No need to save the model again as ModelCheckpoint already saved the best version
            dump(scaler, scaler_path)
            dump(calibrator, calibrator_path)
            
            self.models[symbol][strategy_name] = model
            self.scalers[symbol][strategy_name] = scaler
            self.calibrators[symbol][strategy_name] = calibrator
            
            print(f"✅ Model, scaler, and calibrator for {symbol} ({strategy_name}) trained and saved in {time.time() - start_time:.1f}s.")
        except Exception as e:
            print(f"❌ Training process for {symbol} ({strategy_name}) failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def predict(self, symbol, df, strategy_name='main'):
        if not all(k in self.models.get(symbol, {}) for k in [strategy_name]):
            return None
        try:
            model = self.models[symbol][strategy_name]
            scaler = self.scalers[symbol][strategy_name]
            calibrator = self.calibrators[symbol][strategy_name]
            
            features = ['open', 'high', 'low', 'close', 'volume'] + [indi for indi in TECHNICAL_INDICATORS if indi in df.columns]
            sequence_length = STRATEGIES[strategy_name]['sequence_length']

            last_sequence_df = df[features].tail(sequence_length).astype('float32')
            if len(last_sequence_df) < sequence_length: return None

            scaled_data = scaler.transform(last_sequence_df.values)
            input_data = scaled_data.reshape(1, sequence_length, len(features))
            
            raw_prediction = model.predict(input_data, verbose=0)[0]

            # --- GET CALIBRATED CONFIDENCE ---
            # Get the raw score for the direction prediction
            direction_score = raw_prediction[1].reshape(-1, 1)
            # Get probabilities [P(down), P(up)] from the calibrator
            calibrated_probs = calibrator.predict_proba(direction_score)[0]

            dummy_row = np.zeros((1, len(features)))
            dummy_row[0, 3] = raw_prediction[0]
            predicted_price = scaler.inverse_transform(dummy_row)[0, 3]
            current_price = df['close'].iloc[-1].item()
            pct_change = ((predicted_price - current_price) / current_price) * 100
            
            # Determine direction and get the corresponding calibrated confidence
            if predicted_price > current_price:
                direction = 'long'
                confidence = calibrated_probs[1] # Probability of 'up' (class 1)
            else:
                direction = 'short'
                confidence = calibrated_probs[0] # Probability of 'down' (class 0)

            if abs(pct_change) < 0.05:
                direction = 'hold'
                confidence *= 0.5 # Reduce confidence for tiny moves

            return {
                'asset': symbol, 
                'timestamp': pd.to_datetime('now', utc=True), 
                'price': float(predicted_price), 
                'direction': direction, 
                'confidence': float(confidence), 
                'pct_change': float(pct_change), 
                'current_price': current_price,
                'atr': df['atr'].iloc[-1].item() if 'atr' in df.columns and not pd.isna(df['atr'].iloc[-1]) else 0,
                'strategy': strategy_name
            }
        except Exception as e:
            print(f"💥 Prediction failed for {symbol} ({strategy_name}): {e}")
            return None

    def predict_swing(self, symbol, df):
        return self.predict(symbol, df, strategy_name='swing')

    def predict_scalp(self, symbol, df):
        return self.predict(symbol, df, strategy_name='scalp')
    
    def predict_btc_swing(self, symbol, df):
        return self.predict(symbol, df, strategy_name='btc-swing')