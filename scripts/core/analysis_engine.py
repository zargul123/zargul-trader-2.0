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
        
        if symbol and strategy_type:
            self._initialize_models(specific_symbol=symbol, specific_strategy=strategy_type)
        else:
            self._initialize_models()

    def _align_df_features(self, df: pd.DataFrame) -> pd.DataFrame:
        base_features = ['open', 'high', 'low', 'close', 'volume']
        all_features = base_features + TECHNICAL_INDICATORS
        canonical_feature_list = list(dict.fromkeys(all_features))
        aligned_df = df.reindex(columns=canonical_feature_list)
        aligned_df.fillna(0, inplace=True)
        return aligned_df.astype('float32')

    def _initialize_models(self, specific_symbol=None, specific_strategy=None):
        print("\n🤖 Initializing AI Analyst...")
        os.makedirs('trained_models', exist_ok=True)

        symbols_to_load = [specific_symbol] if specific_symbol else ASSETS
        for symbol in symbols_to_load:
            strategies_to_load = [specific_strategy] if specific_strategy else list(STRATEGIES.get(symbol, {}).keys())
            
            for strategy_name in strategies_to_load:
                if not STRATEGIES.get(symbol, {}).get(strategy_name, {}).get('enabled', True):
                    continue

                model_path = f'trained_models/{symbol}_{strategy_name}_model.h5'
                scaler_path = f'trained_models/{symbol}_{strategy_name}_scaler.joblib'
                calibrator_path = f'trained_models/{symbol}_{strategy_name}_calibrator.joblib'

                if self.train_all or not all(os.path.exists(p) for p in [model_path, scaler_path, calibrator_path]):
                    print(f"🔧 No pre-trained model for {symbol} ({strategy_name}) or retraining requested.")
                    try:
                        self._train_model(symbol, strategy_name)
                    except Exception as e:
                        print(f"❌ Training failed for {symbol} ({strategy_name}): {e}")
                        # Exit gracefully if a model fails to train
                        self.models[symbol][strategy_name] = None
                        self.scalers[symbol][strategy_name] = None
                        self.calibrators[symbol][strategy_name] = None
                else:
                    print(f"🧠 Loading pre-trained model for {symbol} ({strategy_name})...")
                    try:
                        self.models[symbol][strategy_name] = load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
                        self.scalers[symbol][strategy_name] = load(scaler_path)
                        self.calibrators[symbol][strategy_name] = load(calibrator_path)
                        print(f"   - ✅ Components for {symbol} ({strategy_name}) loaded.")
                    except Exception as e:
                        print(f"❌ Failed to load components for {symbol} ({strategy_name}): {e}")

    def _create_advanced_model(self, symbol, strategy_name, input_shape, n_outputs):
        asset_params = MODEL_HYPERPARAMS.get(symbol, {})
        params = asset_params.get(strategy_name, asset_params.get('main', MODEL_HYPERPARAMS['default']))
        print(f"   - Building model for {symbol} ({strategy_name}) with {params['n_layers']} LSTM layer(s).")

        inputs = tf.keras.layers.Input(shape=input_shape)
        x = inputs
        
        for i in range(params['n_layers']):
            units = params[f'units_layer_{i}']
            dropout = params[f'dropout_layer_{i}']
            return_sequences = (i < params['n_layers'] - 1)
            
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(units, return_sequences=return_sequences)
            )(x)
            x = tf.keras.layers.Dropout(dropout)(x)

        output1 = tf.keras.layers.Dense(2, name='price_targets')(x)
        output2 = tf.keras.layers.Dense(n_outputs, activation='softmax', name='trade_signal')(x)

        model = tf.keras.models.Model(inputs=inputs, outputs=[output1, output2])
        optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
        
        model.compile(
            optimizer=optimizer,
            loss={'price_targets': 'mse', 'trade_signal': 'categorical_crossentropy'},
            metrics={'trade_signal': 'accuracy'}
        )
        return model

    def _calculate_historical_regimes(self, df: pd.DataFrame) -> pd.Series:
        price_returns = df['close'].pct_change().fillna(0)
        rolling_entropy = price_returns.rolling(window=50).apply(lambda x: -np.sum(pd.cut(x, bins=[-np.inf, -0.005, 0, 0.005, np.inf], labels=False, right=False) / len(x) * np.log2(pd.cut(x, bins=[-np.inf, -0.005, 0, 0.005, np.inf], labels=False, right=False) / len(x) + 1e-9)), raw=True)
        smoothed_entropy = rolling_entropy.ewm(alpha=0.1).mean()
        
        is_chaotic = smoothed_entropy > 1.5
        is_trending = df['adx'] > 25
        
        regimes = pd.Series("Ranging", index=df.index)
        regimes[is_trending] = "Trending"
        regimes[is_chaotic] = "Chaotic"
        return regimes

    def _create_forward_looking_labels(self, df: pd.DataFrame):
        print("   - Generating robust, dual-simulation labels...")
        n_outputs = 3
        
        df['regime'] = self._calculate_historical_regimes(df)
        
        atr = df['atr'].values
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        regime = df['regime'].values
        
        labels = np.full(len(df), 2)

        for i in range(len(df) - 40):
            current_regime = regime[i]
            
            if current_regime == "Trending":
                tp_mult, sl_mult, window = 1.5, 1.0, 20 # WIDER win-condition for richer labels
            elif current_regime == "Ranging":
                tp_mult, sl_mult, window = 1.0, 1.0, 40
            else:
                continue

            entry_price = close[i]
            
            tp_long = entry_price + (atr[i] * tp_mult)
            sl_long = entry_price - (atr[i] * sl_mult)
            long_win, long_lose = False, False
            
            tp_short = entry_price - (atr[i] * tp_mult)
            sl_short = entry_price + (atr[i] * sl_mult)
            short_win, short_lose = False, False

            for j in range(1, window + 1):
                future_high, future_low = high[i + j], low[i + j]
                
                if not long_win and not long_lose:
                    if future_high >= tp_long: long_win = True
                    if future_low <= sl_long: long_lose = True

                if not short_win and not short_lose:
                    if future_low <= tp_short: short_win = True
                    if future_high >= sl_short: short_lose = True
            
            if long_win and not short_win:
                labels[i] = 0
            elif short_win and not long_win:
                labels[i] = 1
        
        buy_count, sell_count, hold_count = np.sum(labels == 0), np.sum(labels == 1), np.sum(labels == 2)
        total_count = len(labels)
        print("\n" + "="*50)
        print("📊 POST-LABELING DISTRIBUTION REPORT 📊")
        print(f"   - Buy Signals:  {buy_count} ({(buy_count/total_count)*100:.2f}%)")
        print(f"   - Sell Signals: {sell_count} ({(sell_count/total_count)*100:.2f}%)")
        print(f"   - Hold Signals: {hold_count} ({(hold_count/total_count)*100:.2f}%)")
        print("="*50 + "\n")
        
        if buy_count < 10 or sell_count < 10:
            raise ValueError("Insufficient buy or sell signals generated. Halting training.")

        return tf.keras.utils.to_categorical(labels, num_classes=n_outputs)

    def _train_model(self, symbol, strategy_name):
        start_time = time.time()
        print(f"   - Starting training for {symbol} ({strategy_name})...")
        
        df = self.data.get_training_data(symbol, STRATEGIES[symbol][strategy_name]['timeframe'])
        if df is None or df.empty:
            raise ValueError("No data available for training.")

        df_features = self._align_df_features(df)
        y_sig_categorical = self._create_forward_looking_labels(df_features)
        
        if 'regime' in df_features.columns:
            df_features = df_features.drop(columns=['regime'])

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df_features.values)

        sequence_length = STRATEGIES[symbol][strategy_name]['sequence_length']
        X, y_sig = [], []

        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i - sequence_length:i])
            y_sig.append(y_sig_categorical[i])
        
        X, y_sig = np.array(X), np.array(y_sig)
        y_pt = np.zeros((X.shape[0], 2))

        X_train, X_val, y_pt_train, y_pt_val, y_sig_train, y_sig_val = train_test_split(
            X, y_pt, y_sig, test_size=0.2, random_state=42, stratify=y_sig
        )
        
        y_train = {'price_targets': y_pt_train, 'trade_signal': y_sig_train}
        y_val = {'price_targets': y_pt_val, 'trade_signal': y_sig_val}

        model = self._create_advanced_model(symbol, strategy_name, (X_train.shape[1], X_train.shape[2]), n_outputs=y_sig_train.shape[1])
        es = EarlyStopping(monitor='val_loss', patience=TRAINING_CONFIG['early_stop_patience'], restore_best_weights=True)
        
        model_path = f'trained_models/{symbol}_{strategy_name}_model.h5'
        scaler_path = f'trained_models/{symbol}_{strategy_name}_scaler.joblib'
        calibrator_path = f'trained_models/{symbol}_{strategy_name}_calibrator.joblib'
        
        checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_trade_signal_accuracy', mode='max', save_format='h5')

        model.fit(X_train, y_train, epochs=TRAINING_CONFIG['epochs'], batch_size=TRAINING_CONFIG['batch_size'], validation_data=(X_val, y_val), callbacks=[es, checkpoint], verbose=1)

        print(f"   - Training confidence calibrator for {symbol} ({strategy_name})...")
        val_predictions_raw = model.predict(X_val)
        
        calibrator_X = val_predictions_raw[1] 
        calibrator_y = np.argmax(y_sig_val, axis=1)

        calibrator = LogisticRegression(solver='liblinear')
        calibrator.fit(calibrator_X, calibrator_y)
        print("   - ✅ Calibrator trained.")

        dump(scaler, scaler_path)
        dump(calibrator, calibrator_path)
        
        self.models[symbol][strategy_name] = model
        self.scalers[symbol][strategy_name] = scaler
        self.calibrators[symbol][strategy_name] = calibrator
        
        print(f"✅ Model, scaler, and calibrator for {symbol} ({strategy_name}) trained and saved in {time.time() - start_time:.1f}s.")

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
            
            raw_prediction = model.predict(input_data, verbose=0)
            price_targets_pred = raw_prediction[0][0]
            trade_signal_pred = raw_prediction[1][0]

            calibrated_probs = calibrator.predict_proba(trade_signal_pred.reshape(1, -1))[0]

            dummy_row = np.zeros((1, last_sequence_df.shape[1]))
            dummy_row[0, 3] = price_targets_pred[0]
            predicted_price = scaler.inverse_transform(dummy_row)[0, 3]
            
            current_price = df['close'].iloc[-1].item()
            pct_change = ((predicted_price - current_price) / current_price) * 100
            
            signal_index = np.argmax(calibrated_probs)
            if signal_index == 0:
                direction = 'long'
                confidence = calibrated_probs[0]
            elif signal_index == 1:
                direction = 'short'
                confidence = calibrated_probs[1]
            else:
                direction = 'hold'
                confidence = calibrated_probs[2]

            if abs(pct_change) < 0.05:
                direction = 'hold'
                confidence = calibrated_probs[2]

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