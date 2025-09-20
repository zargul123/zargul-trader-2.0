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
        Creates a Keras model with hyperparameters dynamically loaded from the config.
        This version is a pure classifier, focused only on the trade signal.
        """
        asset_params = MODEL_HYPERPARAMS.get(symbol, {})
        params = asset_params.get(strategy_name, asset_params.get('main', MODEL_HYPERPARAMS['default']))
        print(f"   - Building model for {symbol} ({strategy_name}) with {params['n_layers']} LSTM layer(s).")

        model = Sequential()
        for i in range(params['n_layers']):
            units = params[f'units_layer_{i}']
            dropout = params[f'dropout_layer_{i}']
            return_sequences = (i < params['n_layers'] - 1)
            
            if i == 0:
                model.add(Bidirectional(LSTM(units, return_sequences=return_sequences), input_shape=input_shape))
            else:
                model.add(Bidirectional(LSTM(units, return_sequences=return_sequences)))
            model.add(Dropout(dropout))

        model.add(Dense(n_outputs, activation='softmax', name='trade_signal'))
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def _create_forward_looking_labels(self, df: pd.DataFrame):
        """
        Generates labels based on a simple, robust forward-looking percentage change.
        This teaches the AI to only signal on significant price moves.
        """
        print("   - Generating labels based on significant forward moves...")
        n_outputs = 3  # 0: Buy, 1: Sell, 2: Hold
        
        # --- Parameters for the labeling strategy ---
        lookahead_window = 5   # Look 5 bars into the future
        min_move = 0.004        # 0.4% move required for a signal

        # Calculate the future percentage change
        future_close = df['close'].shift(-lookahead_window)
        percent_change = (future_close - df['close']) / df['close']
        
        # Create labels based on the magnitude of the change
        labels = np.full(len(df), 2)  # Default to "Hold"
        labels[percent_change >= min_move] = 0  # Buy signal
        labels[percent_change <= -min_move] = 1 # Sell signal
        
        # Convert to one-hot encoding
        return tf.keras.utils.to_categorical(labels, num_classes=n_outputs)

    def _train_model(self, symbol, strategy_name):
        start_time = time.time()
        print(f"   - Starting training for {symbol} ({strategy_name})...")
        try:
            df = self.data.get_training_data(symbol, STRATEGIES[symbol][strategy_name]['timeframe'])
            if df is None or df.empty:
                raise ValueError(f"Cannot train {symbol} ({strategy_name}), no data available.")

            # Align DF to canonical feature list to ensure consistent shape
            df_features = self._align_df_features(df)

            # --- NEW: Generate the sophisticated forward-looking labels ---
            trade_signal = self._create_forward_looking_labels(df_features)
            
            # --- The rest of the process remains largely the same ---
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df_features.values)

            sequence_length = STRATEGIES[symbol][strategy_name]['sequence_length']
            X, y_sig = [], []
            
            # Print class distribution for debugging
            buy_count = np.sum(trade_signal[:, 0])
            sell_count = np.sum(trade_signal[:, 1])
            hold_count = np.sum(trade_signal[:, 2])
            total = len(trade_signal)
            print(f"   - Class distribution: Buy={buy_count} ({buy_count/total*100:.1f}%), Sell={sell_count} ({sell_count/total*100:.1f}%), Hold={hold_count} ({hold_count/total*100:.1f}%)")

            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i - sequence_length:i])
                y_sig.append(trade_signal[i])
            
            X, y_sig = np.array(X), np.array(y_sig)

            # Split data for main model training and for calibrator training
            X_train, X_val, y_sig_train, y_sig_val = train_test_split(
                X, y_sig, test_size=0.2, random_state=42, stratify=y_sig # Stratify to maintain class balance
            )
            
            # --- REMOVED: The y_pt (price_target) logic is no longer needed ---
            # --- SINGLE OUTPUT: The y_pt logic is removed ---
            y_train = y_sig_train
            y_val = y_sig_val

            # --- NEW: Calculate class weights to handle imbalance ---
            total_samples = len(y_sig_train)
            # Add a small epsilon to prevent division by zero if a class is missing in a batch
            epsilon = 1e-7
            class_weights = {
                0: total_samples / (3 * (np.sum(y_sig_train[:, 0]) + epsilon)), # Buy
                1: total_samples / (3 * (np.sum(y_sig_train[:, 1]) + epsilon)), # Sell
                2: total_samples / (3 * (np.sum(y_sig_train[:, 2]) + epsilon))  # Hold
            }
            print(f"   - Applying class weights: {{0: {class_weights[0]:.2f}, 1: {class_weights[1]:.2f}, 2: {class_weights[2]:.2f}}}")

            model = self._create_advanced_model(symbol, strategy_name, (X_train.shape[1], X_train.shape[2]), n_outputs=y_sig_train.shape[1])
            es = EarlyStopping(monitor='val_loss', patience=TRAINING_CONFIG['early_stop_patience'], restore_best_weights=True)
            
            model_path = f'trained_models/{symbol}_{strategy_name}_model.h5'
            scaler_path = f'trained_models/{symbol}_{strategy_name}_scaler.joblib'
            calibrator_path = f'trained_models/{symbol}_{strategy_name}_calibrator.joblib'
            
            checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', mode='max', save_format='h5')

            # --- UPDATED: Pass class_weight to the fit method for a single output ---
            model.fit(X_train, y_train, epochs=TRAINING_CONFIG['epochs'], batch_size=TRAINING_CONFIG['batch_size'], validation_data=(X_val, y_val), callbacks=[es, checkpoint], verbose=1, class_weight=class_weights)

            # --- PLATT SCALING CALIBRATOR TRAINING ---
            print(f"   - Training confidence calibrator for {symbol} ({strategy_name})...")
            # UPDATED: The model now returns a single array of predictions
            calibrator_X = model.predict(X_val)
            calibrator_y = np.argmax(y_val, axis=1)

            calibrator = LogisticRegression(solver='liblinear')
            calibrator.fit(calibrator_X, calibrator_y)
            print("   - ✅ Calibrator trained.")

            # --- SAVE ALL COMPONENTS ---
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

            df_aligned = self._align_df_features(df)
            last_sequence_df = df_aligned.tail(sequence_length)

            if len(last_sequence_df) < sequence_length: return None

            scaled_data = scaler.transform(last_sequence_df.values)
            input_data = scaled_data.reshape(1, sequence_length, last_sequence_df.shape[1])
            
            # --- REWORKED PREDICTION LOGIC ---
            # 1. Get raw softmax output from the model
            raw_prediction = model.predict(input_data, verbose=0)[0]
            
            # 2. Get calibrated confidence score
            # The calibrator expects a 2D array
            calibrated_probs = calibrator.predict_proba([raw_prediction])[0]
            predicted_class = np.argmax(calibrated_probs)
            confidence = calibrated_probs[predicted_class]

            # 3. Map class index to direction
            direction_map = {0: 'long', 1: 'short', 2: 'hold'}
            direction = direction_map[predicted_class]

            # 4. Synthesize pct_change to be consistent with our new labeling rule.
            # The risk engine will use this to confirm the trade meets its own thresholds.
            if direction == 'long':
                pct_change = 0.4 # Corresponds to the 0.4% min_move
            elif direction == 'short':
                pct_change = -0.4 # Corresponds to the 0.4% min_move
            else:
                pct_change = 0.0

            return {
                'asset': symbol, 
                'timestamp': pd.to_datetime('now', utc=True), 
                'price': df['close'].iloc[-1].item(), # Use current price as reference
                'direction': direction, 
                'confidence': float(confidence), 
                'pct_change': float(pct_change), 
                'current_price': df['close'].iloc[-1].item(),
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