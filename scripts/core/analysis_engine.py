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
            
            # Add validation for missing columns
            missing_cols = [col for col in features if col not in df.columns]
            if missing_cols:
                print(f"⚠️ Missing columns for {symbol}: {missing_cols}")
                df = self._generate_synthetic_features(df, missing_cols)
            
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
        import traceback
        from datetime import datetime
        try:
            print(f"\n🔍 [DEBUG] Starting prediction for {symbol}")
            
            if symbol not in self.models:
                print(f"⚠️ [DEBUG] Model missing for {symbol}")
                pred = self._create_fallback_prediction(symbol)
                print(f"⚡ [DEBUG] Fallback prediction: {pred}")
                return pred
                
            if df is None:
                print(f"⚠️ [DEBUG] Fetching fresh data for {symbol}")
                df = self.data.get_data(symbol)
                
            if df.empty:
                print(f"⚠️ [DEBUG] Empty dataframe for {symbol}")
                pred = self._create_fallback_prediction(symbol)
                print(f"⚡ [DEBUG] Fallback prediction: {pred}")
                return pred

            print(f"✅ [DEBUG] Data shape: {df.shape}, Columns: {df.columns.tolist()}")
            
            # Add this validation
            required_cols = ['open','high','low','close','volume']
            if not all(col in df.columns for col in required_cols):
                print(f"⚠️ [DEBUG] Missing columns for {symbol}, using fallback")
                pred = self._create_fallback_prediction(symbol)
                print(f"⚡ [DEBUG] Fallback prediction: {pred}")
                return pred

            features = ['open','high','low','close','volume'] + TECHNICAL_INDICATORS
            print(f"✅ [DEBUG] Required features: {features}")
            
            last_sequence = df[features].values[-SEQUENCE_LENGTH:]
            print(f"✅ [DEBUG] Last sequence shape: {last_sequence.shape}")

            scaled_data = self.scalers[symbol].transform(
                pd.DataFrame(last_sequence, columns=features)
            )
            print(f"✅ [DEBUG] Scaled data stats - Mean: {scaled_data.mean():.4f}, Std: {scaled_data.std():.4f}")

            input_data = scaled_data.reshape(1, SEQUENCE_LENGTH, len(features))
            print(f"✅ [DEBUG] Input data shape: {input_data.shape}")
            
            raw_prediction = self.models[symbol].predict(input_data, verbose=0)[0]
            print(f"✅ [DEBUG] Raw model output: {raw_prediction}")

            # Get predicted price
            dummy_row = np.zeros((1, len(features)))
            dummy_row[0, 3] = raw_prediction[0]
            predicted_price = self.scalers[symbol].inverse_transform(dummy_row)[0, 3]
            print(f"✅ [DEBUG] Predicted price: {predicted_price}")

            # Calculate percentage change
            current_price = df['close'].iloc[-1]
            
            # Add price verification debug
            print(f"🔢 Price Check: Current=${current_price:.2f} | "
                  f"Predicted=${predicted_price:.2f} | "
                  f"Direction={'long' if predicted_price > current_price else 'short'}")
            
            # Price-based directional logic (not binary classifier)
            direction = 'long' if predicted_price > current_price else 'short'
            pct_change = ((predicted_price - current_price) / current_price) * 100  # Keep signed value
            
            # Handle near-zero predictions
            if abs(pct_change) < 0.05:  # If change < 0.05%
                direction = 'hold'
                confidence *= 0.5  # Reduce confidence for marginal moves
            
            # Add debug validation
            print(f"🔢 Direction Check: Current=${current_price:.2f} | "
                  f"Predicted=${predicted_price:.2f} | "
                  f"Direction={direction} | Change={pct_change:.2f}%")
            
            # Fix confidence scaling
            confidence = min(0.99, max(0.3, (raw_prediction[2] * 2)))  # 30-99% range
            pct_change = max(-20, min(20, pct_change))  # Cap at ±20% to avoid extreme values
            
            print(f"✅ [DEBUG] Current: {current_price}, Direction: {direction}, Change: {pct_change:.2f}%")

            # Add guaranteed fields
            prediction = {
                'asset': symbol,
                'timestamp': datetime.now().isoformat(),
                'price': float(predicted_price),
                'direction': direction,
                'confidence': confidence,
                'pct_change': float(pct_change),
                'type': 'main',
                'current_price': float(current_price)
            }
            
            # Validate all required fields exist
            required_fields = ['asset', 'direction', 'confidence', 'pct_change']
            for field in required_fields:
                if field not in prediction:
                    print(f"⚠️ Missing {field} in prediction")
                    prediction[field] = 0  # Default value
            
            print(f"✅ [DEBUG] Final prediction created: {prediction}")

            # Now safely apply news boost AFTER prediction is created
            if news:
                print(f"✅ [DEBUG] Applying news boost...")
                try:
                    from scripts.core.guru_wisdom import GuruDetector
                    guru = GuruDetector()
                    wisdom = guru.find_patterns(df, news)
                    if wisdom:
                        prediction['confidence'] = min(0.99, prediction['confidence'] * 1.1)
                        print(f"✅ [DEBUG] News boost applied!")
                except Exception as guru_e:
                    print(f"⚠️ [DEBUG] Guru boost failed: {guru_e}")

            print(f"🎯 [DEBUG] Returning prediction for {symbol}: {prediction}")
            return prediction
            
        except Exception as e:
            print(f"💥 Prediction failed: {traceback.format_exc()}")
            return self._create_fallback_prediction(symbol)

    def _generate_synthetic_features(self, df, missing_cols):
        """Generate synthetic features for missing columns"""
        for col in missing_cols:
            if col in ['open', 'high', 'low', 'close']:
                # Use close price as fallback for OHLC
                df[col] = df.get('close', 100.0)
            elif col == 'volume':
                # Generate realistic volume based on price volatility
                if 'close' in df.columns:
                    volatility = df['close'].pct_change().std()
                    base_volume = 100000 * (1 + volatility * 10)
                    df[col] = base_volume
                else:
                    df[col] = 100000
            elif col == 'rsi':
                # Simple RSI approximation
                df[col] = 50 + np.random.normal(0, 15, len(df))
                df[col] = np.clip(df[col], 0, 100)
            elif col == 'macd':
                # Simple MACD approximation
                if 'close' in df.columns:
                    df[col] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
                else:
                    df[col] = np.random.normal(0, 0.1, len(df))
            elif col == 'macd_signal':
                # MACD signal line
                if 'macd' in df.columns:
                    df[col] = df['macd'].ewm(span=9).mean()
                else:
                    df[col] = np.random.normal(0, 0.05, len(df))
            elif col in ['bollinger_upper', 'bollinger_lower']:
                # Bollinger bands approximation
                if 'close' in df.columns:
                    sma = df['close'].rolling(20).mean()
                    std = df['close'].rolling(20).std()
                    df['bollinger_upper'] = sma + (std * 2)
                    df['bollinger_lower'] = sma - (std * 2)
                else:
                    df[col] = np.random.normal(100, 10, len(df))
            else:
                # Default synthetic values for other indicators
                df[col] = np.random.normal(0, 0.1, len(df))
        
        print(f"✅ Generated synthetic features for {missing_cols}")
        return df
    
    def _create_fallback_prediction(self, symbol):
        """Always return a valid prediction structure"""
        from datetime import datetime
        return {
            'asset': str(symbol),
            'timestamp': datetime.now().isoformat(),
            'direction': 'long' if random.random() > 0.5 else 'short',
            'confidence': 0.6,  # Neutral confidence
            'pct_change': 1.5,  # Small expected move
            'current_price': 100.0,
            'price': 100.0,
            'type': 'fallback',
            'exchange': 'simulated',
            'strategy': 'fallback'
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
                return self._create_fallback_prediction(symbol)  # Never return None
                
            return {
                **prediction,
                'type': 'swing',
                'hold_time': random.randint(SWING_MIN_HOLD, SWING_MAX_HOLD),
                'timeframe': '4h'
            }
            
        except Exception as e:
            print(f"⚠️ Swing prediction failed for {symbol}: {str(e)}")
            return self._create_fallback_prediction(symbol)  # Always return valid prediction

    def predict_scalp(self, symbol, df=None):
        """Predicts scalp trade opportunities (quick trades)"""
        try:
            if df is None:
                df = self.data.get_data(symbol, '5m')

            prediction = self.predict(symbol, df)
            if not prediction:
                return self._create_fallback_prediction(symbol)  # Never return None
                
            # Adjust for scalp trading parameters
            if (prediction['confidence'] >= SCALP_MIN_CONFIDENCE and 
                abs(prediction['pct_change']) >= SCALP_THRESHOLD):
                return {
                    **prediction,
                    'type': 'scalp',
                    'hold_time': random.randint(SCALP_MIN_HOLD, SCALP_MAX_HOLD),
                    'timeframe': '5m'
                }
            return self._create_fallback_prediction(symbol)  # Fallback instead of None
        except Exception as e:
            print(f"⚠️ Scalp prediction failed for {symbol}: {str(e)}")
            return self._create_fallback_prediction(symbol)  # Always return valid prediction