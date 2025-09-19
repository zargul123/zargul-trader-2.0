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
from scripts.config import ASSETS, TECHNICAL_INDICATORS, TRAINING_CONFIG, STRATEGIES, MODEL_HYPERPARAMS

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
    def __init__(self, train_all=False, symbol=None, strategy_type=None):
        self.models = {s: {} for s in ASSETS}
        self.scalers = {s: {} for s in ASSETS}
        self.calibrators = {s: {} for s in ASSETS}
        self.prediction_functions = {s: {} for s in ASSETS}
        self.train_all = train_all
        self.data = DataMaster()
        
        # This is the core of the change. We decide WHICH models to load.
        if symbol and strategy_type:
            # If a specific symbol and strategy are provided, only load that one.
            # This is the new, efficient path for backtesting.
            self._initialize_models(specific_symbol=symbol, specific_strategy=strategy_type)
        else:
            # Otherwise, run the original logic to load all models.
            # This preserves the behavior for the main trading bot.
            self._initialize_models()

    def _align_df_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aligns the DataFrame columns to the canonical feature list from the config.
        This ensures data passed to the scaler/model always has the exact same shape
        by adding missing columns with 0 and ensuring a consistent order.
        """
        # The full, canonical list of features the model expects.
        base_features = ['open', 'high', 'low', 'close', 'volume']
        # TECHNICAL_INDICATORS from config now includes the 'lc_' metrics.
        all_features = base_features + TECHNICAL_INDICATORS
        
        # Use dict.fromkeys to get a unique list while preserving order
        canonical_feature_list = list(dict.fromkeys(all_features))
        
        # Reindex the DataFrame to match the canonical list.
        # This adds missing columns with NaN and removes unexpected ones.
        aligned_df = df.reindex(columns=canonical_feature_list)
        
        # Fill any NaN values that resulted from missing columns (e.g., API failure)
        aligned_df.fillna(0, inplace=True)
        
        return aligned_df.astype('float32')

    def _initialize_models(self, specific_symbol=None, specific_strategy=None):
        """
        Initializes models. If specific_symbol and specific_strategy are provided,
        it only loads/trains that specific model. Otherwise, it loads all models.
        """
        print("\n🤖 Initializing AI Analyst...")
        os.makedirs('trained_models', exist_ok=True)

        # If a specific model is requested, create a targeted list of one
        if specific_symbol and specific_strategy:
            symbols_to_load = [specific_symbol]
        else:
            symbols_to_load = ASSETS

        for symbol in symbols_to_load:
            # Determine which strategies to load for the current symbol
            if specific_symbol and specific_strategy:
                strategies_to_load = [specific_strategy]
            else:
                # Original logic to determine all strategies for a symbol
                strategies_to_load = ['main', 'scalp']
                if symbol == 'BTC-USD':
                    strategies_to_load.append('btc-swing')
                else:
                    strategies_to_load.append('swing')

            for strategy_name in strategies_to_load:
                model_path = f'trained_models/{symbol}_{strategy_name}_model.h5'
                scaler_path = f'trained_models/{symbol}_{strategy_name}_scaler.joblib'
                calibrator_path = f'trained_models/{symbol}_{strategy_name}_calibrator.joblib'

                if self.train_all or not all(os.path.exists(p) for p in [model_path, scaler_path, calibrator_path]):
                    print(f"🔧 No pre-trained model/scaler/calibrator for {symbol} ({strategy_name}) or retraining requested.")
                    try:
                        self._train_model(symbol, strategy_name)
                    except Exception as e:
                        print(f"❌ Training failed for {symbol} ({strategy_name}): {e}")
                else:
                    print(f"🧠 Loading pre-trained model, scaler, and calibrator for {symbol} ({strategy_name})...")
                    try:
                        self.models[symbol][strategy_name] = load_model(model_path)
                        self.scalers[symbol][strategy_name] = load(scaler_path)
                        self.calibrators[symbol][strategy_name] = load(calibrator_path)
                        print(f"   - ✅ Components for {symbol} ({strategy_name}) loaded successfully.")
                    except Exception as e:
                        print(f"❌ Failed to load components for {symbol} ({strategy_name}): {e}")

    def _create_advanced_model(self, symbol, strategy_name, input_shape, n_outputs):
        """
        Creates a Keras model with hyperparameters dynamically loaded from the config
        based on the asset symbol and strategy name.
        """
        # --- 1. Load Hyperparameters with a robust fallback system ---
        asset_params = MODEL_HYPERPARAMS.get(symbol, {})
        params = asset_params.get(strategy_name, asset_params.get('main', MODEL_HYPERPARAMS['default']))
        print(f"   - Building model for {symbol} ({strategy_name}) with {params['n_layers']} LSTM layer(s).")

        # --- 2. Build Model Dynamically ---
        inputs = tf.keras.layers.Input(shape=input_shape)
        x = inputs
        
        for i in range(params['n_layers']):
            units = params[f'units_layer_{i}']
            dropout = params[f'dropout_layer_{i}']
            return_sequences = (i < params['n_layers'] - 1) # Return sequences for all but the last LSTM layer
            
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(units, return_sequences=return_sequences)
            )(x)
            x = tf.keras.layers.Dropout(dropout)(x)

        # --- 3. Output Layers ---
        output1 = tf.keras.layers.Dense(2, name='price_targets')(x)
        output2 = tf.keras.layers.Dense(n_outputs, activation='softmax', name='trade_signal')(x)

        model = tf.keras.models.Model(inputs=inputs, outputs=[output1, output2])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
        
        model.compile(
            optimizer=optimizer,
            loss={
                'price_targets': 'mse',
                'trade_signal': 'categorical_crossentropy'
            },
            metrics={'trade_signal': 'accuracy'}
        )
        return model

    def _train_model(self, symbol, strategy_name):
        start_time = time.time()
        print(f"   - Starting training for {symbol} ({strategy_name})...")
        try:
            df = self.data.get_training_data(symbol, STRATEGIES[symbol][strategy_name]['timeframe'])
            if df is None or df.empty:
                raise ValueError(f"Cannot train {symbol} ({strategy_name}), no data available.")

            # Align DF to canonical feature list to ensure consistent shape
            df_features = self._align_df_features(df)

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df_features.values)

            sequence_length = STRATEGIES[symbol][strategy_name]['sequence_length']
            X, y_pt, y_sig = [], [], []

            # --- Create Labels for the new dual-output model ---
            # FIXED: Use RAW percentage returns instead of scaled differences
            raw_close = df_features.iloc[:, 3].values  # Raw close prices BEFORE scaling
            future_close = pd.Series(raw_close).shift(-5).fillna(method='ffill').values
            percent_return = (future_close - raw_close) / raw_close
            
            # Use percentage-based thresholds appropriate for timeframe
            timeframe = STRATEGIES[symbol][strategy_name]['timeframe']
            if timeframe == '1h':
                threshold = 0.008  # 0.8% for 1h
            elif timeframe == '4h':
                threshold = 0.015  # 1.5% for 4h  
            elif timeframe == '5m':
                threshold = 0.003  # 0.3% for 5m
            else:
                threshold = 0.005  # Default 0.5%

            # Output 1: Price Targets [take_profit, stop_loss] - use scaled for consistency
            price_targets = np.zeros((len(scaled_data), 2))
            future_price_scaled = pd.Series(scaled_data[:, 3]).shift(-5).fillna(method='ffill')
            price_targets[:, 0] = future_price_scaled
            price_targets[:, 1] = scaled_data[:, 3] * 0.98

            # Output 2: Trade Signal [Buy, Sell, Hold] - FIXED to use raw returns
            trade_signal = np.zeros((len(scaled_data), 3))
            trade_signal[percent_return > threshold, 0] = 1   # Buy
            trade_signal[percent_return < -threshold, 1] = 1  # Sell  
            trade_signal[np.abs(percent_return) <= threshold, 2] = 1  # Hold
            
            # Print class distribution for debugging
            buy_count = np.sum(trade_signal[:, 0])
            sell_count = np.sum(trade_signal[:, 1])
            hold_count = np.sum(trade_signal[:, 2])
            total = len(trade_signal)
            print(f"   - Class distribution: Buy={buy_count} ({buy_count/total*100:.1f}%), Sell={sell_count} ({sell_count/total*100:.1f}%), Hold={hold_count} ({hold_count/total*100:.1f}%)")

            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i - sequence_length:i])
                y_pt.append(price_targets[i])
                y_sig.append(trade_signal[i])
            
            X, y_pt, y_sig = np.array(X), np.array(y_pt), np.array(y_sig)

            # Split data for main model training and for calibrator training
            X_train, X_val, y_pt_train, y_pt_val, y_sig_train, y_sig_val = train_test_split(
                X, y_pt, y_sig, test_size=0.2, random_state=42
            )
            
            y_train = {'price_targets': y_pt_train, 'trade_signal': y_sig_train}
            y_val = {'price_targets': y_pt_val, 'trade_signal': y_sig_val}

            model = self._create_advanced_model(symbol, strategy_name, (X_train.shape[1], X_train.shape[2]), n_outputs=y_sig_train.shape[1])
            es = EarlyStopping(monitor='val_loss', patience=TRAINING_CONFIG['early_stop_patience'], restore_best_weights=True)
            
            # Use .h5 format to avoid Keras native format issues with ModelCheckpoint
            model_path = f'trained_models/{symbol}_{strategy_name}_model.h5'
            scaler_path = f'trained_models/{symbol}_{strategy_name}_scaler.joblib'
            calibrator_path = f'trained_models/{symbol}_{strategy_name}_calibrator.joblib'
            
            # Explicitly save in h5 format
            checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_trade_signal_accuracy', mode='max', save_format='h5')

            model.fit(X_train, y_train, epochs=TRAINING_CONFIG['epochs'], batch_size=TRAINING_CONFIG['batch_size'], validation_data=(X_val, y_val), callbacks=[es, checkpoint], verbose=1)

            # --- PLATT SCALING CALIBRATOR TRAINING ---
            print(f"   - Training confidence calibrator for {symbol} ({strategy_name})...")
            val_predictions_raw = model.predict(X_val)
            
            # The input for the calibrator is the softmax output of the trade_signal head
            calibrator_X = val_predictions_raw[1] 
            # The target is the actual class index (0, 1, or 2)
            calibrator_y = np.argmax(y_sig_val, axis=1)

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
            
            sequence_length = STRATEGIES[symbol][strategy_name]['sequence_length']

            # Align DF to canonical feature list to ensure consistent shape
            df_aligned = self._align_df_features(df)
            last_sequence_df = df_aligned.tail(sequence_length)

            if len(last_sequence_df) < sequence_length: return None

            scaled_data = scaler.transform(last_sequence_df.values)
            # Use the DataFrame's shape for robustness
            input_data = scaled_data.reshape(1, sequence_length, last_sequence_df.shape[1])
            
            # Returns a list of two arrays: [price_targets, trade_signal]
            raw_prediction = model.predict(input_data, verbose=0)
            price_targets_pred = raw_prediction[0][0]
            trade_signal_pred = raw_prediction[1][0]

            # --- GET CALIBRATED CONFIDENCE ---
            # Get probabilities [P(buy), P(sell), P(hold)] from the calibrator
            calibrated_probs = calibrator.predict_proba(trade_signal_pred.reshape(1, -1))[0]

            # The predicted price is the first element of the price_targets output
            dummy_row = np.zeros((1, last_sequence_df.shape[1]))
            dummy_row[0, 3] = price_targets_pred[0] # Predicted take_profit level
            predicted_price = scaler.inverse_transform(dummy_row)[0, 3]
            
            current_price = df['close'].iloc[-1].item()
            pct_change = ((predicted_price - current_price) / current_price) * 100
            
            # --- NEW, SIMPLIFIED LOGIC: Trust the pct_change signal ---
            # The price prediction head is more reliable than the classifier head.
            # We will derive the direction directly from its output.
            
            move_threshold = 0.1 # A tiny 0.1% move is enough to be considered a signal
            
            if pct_change > move_threshold:
                direction = 'long'
                confidence = calibrated_probs[0] # Confidence for a 'long' signal
            elif pct_change < -move_threshold:
                direction = 'short'
                confidence = calibrated_probs[1] # Confidence for a 'short' signal
            else:
                direction = 'hold'
                confidence = calibrated_probs[2] # Confidence for a 'hold' signal

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