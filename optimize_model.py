

import os
import sys
import numpy as np
import optuna
import warnings
import gc

# --- Setup Project Environment ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
warnings.filterwarnings("ignore")

# TensorFlow imports must be after the environment variable is set
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from scripts.core.data_engine import DataMaster
from scripts.config import ASSETS, STRATEGIES, TECHNICAL_INDICATORS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import argparse

# --- Global Configuration ---
N_TRIALS = 50  # Number of optimization trials to run
EPOCHS = 20    # Number of epochs to train each model

def create_model(trial, input_shape, n_outputs):
    """
    Creates a Keras model with hyperparameters suggested by Optuna.
    """
    # --- Define Search Space ---
    n_layers = trial.suggest_int('n_layers', 1, 3)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    
    inputs = Input(shape=input_shape)
    x = inputs
    
    for i in range(n_layers):
        units = trial.suggest_int(f'units_layer_{i}', 32, 256, log=True)
        dropout = trial.suggest_float(f'dropout_layer_{i}', 0.1, 0.5)
        return_sequences = (i < n_layers - 1)
        
        x = Bidirectional(LSTM(units, return_sequences=return_sequences))(x)
        x = Dropout(dropout)(x)

    # --- Output Layer ---
    # This matches the structure of your existing AIAnalyst model
    # Output 1: Take Profit and Stop Loss levels
    output1 = Dense(2, name='price_targets')(x) 
    # Output 2: Trade signal (Buy/Sell/Hold)
    output2 = Dense(n_outputs, activation='softmax', name='trade_signal')(x)

    model = Model(inputs=inputs, outputs=[output1, output2])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss={
            'price_targets': 'mse',
            'trade_signal': 'categorical_crossentropy'
        },
        metrics={'trade_signal': 'accuracy'}
    )
    
    return model

def objective(trial, X_train, y_train, X_val, y_val):
    """
    The objective function for Optuna to minimize.
    """
    tf.keras.backend.clear_session()
    gc.collect()

    input_shape = (X_train.shape[1], X_train.shape[2])
    # y_train is a list of two arrays, get n_outputs from the second one
    n_outputs = y_train[1].shape[1] 
    
    model = create_model(trial, input_shape, n_outputs);
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train,
        {'price_targets': y_train[0], 'trade_signal': y_train[1]},
        validation_data=(X_val, {'price_targets': y_val[0], 'trade_signal': y_val[1]}),
        epochs=EPOCHS,
        callbacks=[early_stopping],
        verbose=0
    )
    
    val_loss = np.min(history.history['val_loss'])
    
    # --- Log Interpretable Metric (as suggested by friend) ---
    # We are getting the price targets from the validation set (y_val[0])
    # and calculating the RMSE in percentage terms.
    rmse_pct = 100 * np.sqrt(val_loss) / np.mean(y_val[0])
    trial.set_user_attr("rmse_%", f"{rmse_pct:.2f}%")

    return val_loss

def run_optimization(asset_to_optimize, strategy_name):
    """
    Main function to run the hyperparameter optimization.
    """
    print("="*80)
    print("🤖 INITIATING AI MODEL HYPERPARAMETER OPTIMIZATION 🤖")
    print(f"Asset: {asset_to_optimize}, Strategy: {strategy_name}")
    print(f"Running {N_TRIALS} trials of {EPOCHS} epochs each.")
    print("="*80)

    # --- 1. Load and Prepare Data ---
    print("\n[PHASE 1/3] Loading and preparing data...")
    data_master = DataMaster()
    timeframe = STRATEGIES[strategy_name]['timeframe']
    sequence_length = STRATEGIES[strategy_name]['sequence_length']
    
    df = data_master.get_training_data(asset_to_optimize, timeframe, days=365*3) # 3 years of data
    if df is None or df.empty:
        print(f"❌ Could not get data for {asset_to_optimize}. Aborting.")
        return


    features = ['open', 'high', 'low', 'close', 'volume'] + [indi for indi in TECHNICAL_INDICATORS if indi in df.columns]
    df_features = df[features].astype('float32')
    
    # --- Create Labels ---
    # Simplified labeling for optimization purposes
    df['future_price'] = df['close'].shift(-5)
    df.dropna(inplace=True)
    
    # Price target labels
    price_targets = np.zeros((len(df), 2))
    price_targets[:, 0] = df['future_price'] # Simplified Take Profit
    price_targets[:, 1] = df['close'] * 0.98 # Simplified Stop Loss

    # Trade signal labels
    price_change = df['future_price'] - df['close']
    y_signal = np.zeros((len(df), 3)) # [Buy, Sell, Hold]
    y_signal[price_change > df['close'] * 0.01, 0] = 1  # Buy
    y_signal[price_change < -df['close'] * 0.01, 1] = 1 # Sell
    y_signal[np.abs(price_change) <= df['close'] * 0.01, 2] = 1 # Hold

    # --- Scale and Create Sequences ---
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_features.loc[df.index])

    X, y_pt, y_sig = [], [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length:i])
        y_pt.append(price_targets[i])
        y_sig.append(y_signal[i])
        
    X = np.array(X)
    y_pt = np.array(y_pt)
    y_sig = np.array(y_sig)

    # --- Split Data ---
    X_train, X_val, y_pt_train, y_pt_val, y_sig_train, y_sig_val = train_test_split(
        X, y_pt, y_sig, test_size=0.2, random_state=42
    )
    
    y_train = [y_pt_train, y_sig_train]
    y_val = [y_pt_val, y_sig_val]

    print("✅ Data preparation complete.")
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")

    # --- 2. Run Optuna Study ---
    print("\n[PHASE 2/3] Starting Optuna optimization study...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=N_TRIALS, show_progress_bar=True)
    print("✅ Optimization study complete.")

    # --- 3. Display Results ---
    print("\n[PHASE 3/3] Optimization Results:")
    print(f"  Number of finished trials: {len(study.trials)}")
    
    best_trial = study.best_trial
    print(f"  Best trial value (validation loss): {best_trial.value:.4f}")
    
    print("  Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
        
    print("\n" + "="*80)
    print("✅ OPTIMIZATION COMPLETE")
    print("="*80)
    print("You can now use these hyperparameters to update the model in 'scripts/core/analysis_engine.py'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run hyperparameter optimization for a specific asset and strategy.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--asset',
        type=str,
        required=True,
        help="The asset to optimize (e.g., 'BTC-USD')."
    )
    parser.add_argument(
        '--strategy',
        type=str,
        required=True,
        choices=STRATEGIES.keys(),
        help="The strategy to optimize (e.g., 'main', 'scalp')."
    )
    args = parser.parse_args()
    
    run_optimization(args.asset, args.strategy)
