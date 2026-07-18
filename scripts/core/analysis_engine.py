import os
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, Bidirectional
try:
    from tensorflow.keras.optimizers import AdamW
except ImportError:
    # TensorFlow 2.10 (DirectML build for AMD GPUs) ships AdamW under .experimental
    from tensorflow.keras.optimizers.experimental import AdamW
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

def _simulate_long_trade(future_highs, future_lows, take_profit, stop_loss):
    """Simulates a long trade, returning 1 for win, -1 for loss, 0 for no outcome."""
    for high, low in zip(future_highs, future_lows):
        if high >= take_profit:
            return 1  # Win
        if low <= stop_loss:
            return -1  # Loss
    return 0 # No outcome within the window

def _simulate_short_trade(future_highs, future_lows, take_profit, stop_loss):
    """Simulates a short trade, returning 1 for win, -1 for loss, 0 for no outcome."""
    for high, low in zip(future_highs, future_lows):
        if low <= take_profit:
            return 1  # Win
        if high >= stop_loss:
            return -1  # Loss
    return 0 # No outcome within the window

class AIAnalyst:
    def __init__(self, train_all=False, symbol=None, strategy_type=None):
        self.models = {s: {} for s in ASSETS}
        self.scalers = {s: {} for s in ASSETS}
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

                if self.train_all or not all(os.path.exists(p) for p in [model_path, scaler_path]):
                    print(f"🔧 No pre-trained model/scaler for {symbol} ({strategy_name}) or retraining requested.")
                    try:
                        self._train_model(symbol, strategy_name)
                    except Exception as e:
                        print(f"❌ Training failed for {symbol} ({strategy_name}): {e}")
                else:
                    print(f"🧠 Loading pre-trained model and scaler for {symbol} ({strategy_name})...")
                    try:
                        self.models[symbol][strategy_name] = load_model(model_path)
                        self.scalers[symbol][strategy_name] = load(scaler_path)
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
        print(f"   - Starting training for {symbol} ({strategy_name}) with UNIFIED ATR-based labeling...")
        try:
            df = self.data.get_training_data(symbol, STRATEGIES[symbol][strategy_name]['timeframe'])
            if df is None or df.empty or 'atr' not in df.columns:
                raise ValueError(f"Cannot train {symbol}, no data or ATR column available.")

            df_features = self._align_df_features(df)
            
            print("   - Generating unified, strategy-aligned labels...")
            
            strategy_rules = STRATEGIES[symbol][strategy_name].get('Trending', {})
            tp_atr_mult = strategy_rules.get('tp_atr_multiplier', 2.0)
            sl_atr_mult = strategy_rules.get('sl_atr_multiplier', 1.5)
            future_window = 20

            highs = df_features['high'].values
            lows = df_features['low'].values
            closes = df_features['close'].values
            atrs = df_features['atr'].values

            signal_labels = np.zeros(len(df_features))
            price_target_labels = np.zeros((len(df_features), 2))

            for i in range(len(df_features) - future_window):
                current_close = closes[i]
                current_atr = atrs[i]

                if current_atr == 0:
                    signal_labels[i] = 0 # Hold
                    price_target_labels[i, 0] = current_close
                    price_target_labels[i, 1] = current_close
                    continue

                long_target = current_close + (tp_atr_mult * current_atr)
                long_stop = current_close - (sl_atr_mult * current_atr)
                
                future_highs = highs[i+1 : i+1+future_window]
                future_lows = lows[i+1 : i+1+future_window]

                long_outcome = _simulate_long_trade(future_highs, future_lows, long_target, long_stop)
                
                if long_outcome == 1:
                    signal_labels[i] = 1  # Buy
                elif long_outcome == -1:
                    signal_labels[i] = -1 # Sell
                else:
                    signal_labels[i] = 0  # Hold
                
                price_target_labels[i, 0] = long_target
                price_target_labels[i, 1] = long_stop

            # Correctly one-hot encode the signal labels
            trade_signal = np.zeros((len(signal_labels), 3))
            trade_signal[np.where(signal_labels == 1), 0] = 1
            trade_signal[np.where(signal_labels == -1), 1] = 1
            trade_signal[np.where(signal_labels == 0), 2] = 1

            buy_count = np.sum(trade_signal[:, 0])
            sell_count = np.sum(trade_signal[:, 1])
            hold_count = np.sum(trade_signal[:, 2])
            total = len(trade_signal)
            print(f"   - Class distribution: Buy={buy_count} ({buy_count/total*100:.1f}%), Sell={sell_count} ({sell_count/total*100:.1f}%), Hold={hold_count} ({hold_count/total*100:.1f}%)")

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df_features.values)
            
            pt_scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_price_targets = pt_scaler.fit_transform(price_target_labels)

            sequence_length = STRATEGIES[symbol][strategy_name]['sequence_length']
            
            X, y_pt, y_sig = [], [], []
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i - sequence_length:i])
                y_sig.append(trade_signal[i])
                y_pt.append(scaled_price_targets[i])

            X, y_pt, y_sig = np.array(X), np.array(y_pt), np.array(y_sig)

            X_train, X_val, y_pt_train, y_pt_val, y_sig_train, y_sig_val = train_test_split(
                X, y_pt, y_sig, test_size=0.2, random_state=42, stratify=y_sig
            )
            
            y_train = {'price_targets': y_pt_train, 'trade_signal': y_sig_train}
            y_val = {'price_targets': y_pt_val, 'trade_signal': y_sig_val}

            model = self._create_advanced_model(symbol, strategy_name, (X_train.shape[1], X_train.shape[2]), n_outputs=y_sig_train.shape[1])
            es = EarlyStopping(monitor='val_loss', patience=TRAINING_CONFIG['early_stop_patience'], restore_best_weights=True)
            
            model_path = f'trained_models/{symbol}_{strategy_name}_model.h5'
            scaler_path = f'trained_models/{symbol}_{strategy_name}_scaler.joblib'
            
            checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_trade_signal_accuracy', mode='max', save_format='h5')

            model.fit(X_train, y_train, epochs=TRAINING_CONFIG['epochs'], batch_size=TRAINING_CONFIG['batch_size'], validation_data=(X_val, y_val), callbacks=[es, checkpoint], verbose=1)

            dump(scaler, scaler_path)
            self.models[symbol][strategy_name] = model
            self.scalers[symbol][strategy_name] = scaler
            
            print(f"✅ Model and scaler for {symbol} ({strategy_name}) trained and saved in {time.time() - start_time:.1f}s.")
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

            # --- USE RAW MODEL OUTPUT (CALIBRATOR PERMANENTLY DISABLED) ---
            raw_probs = trade_signal_pred

            # To inverse transform, we need to create a dummy array of the correct shape
            dummy_row = np.zeros((1, scaled_data.shape[1]))
            # Place the scaled TP and SL predictions into the 'high' and 'low' columns
            # as they are the most likely to contain the min/max range of the scaler.
            high_col_index = df_aligned.columns.get_loc('high')
            low_col_index = df_aligned.columns.get_loc('low')
            dummy_row[0, high_col_index] = price_targets_pred[0] # Predicted TP
            dummy_row[0, low_col_index] = price_targets_pred[1]  # Predicted SL
            
            # Inverse transform the entire row using the main data scaler
            unscaled_row = scaler.inverse_transform(dummy_row)
            
            # Extract the unscaled TP price
            predicted_price = unscaled_row[0, high_col_index]

            current_price = df['close'].iloc[-1].item()
            # --- UNIFIED LOGIC: Calculate pct_change based on the PREDICTED TAKE PROFIT ---
            pct_change = ((predicted_price - current_price) / current_price) * 100
            
            # Determine direction from the class with the highest probability
            signal_index = np.argmax(raw_probs)
            if signal_index == 0: # Buy
                direction = 'long'
                confidence = raw_probs[0]
            elif signal_index == 1: # Sell
                direction = 'short'
                confidence = raw_probs[1]
            else: # Hold
                direction = 'hold'
                confidence = raw_probs[2]

            # Override direction for tiny moves, but use the hold confidence
            if abs(pct_change) < 0.05:
                direction = 'hold'
                confidence = raw_probs[2]

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