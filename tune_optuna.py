"""
Optuna Hyperparameter Tuning for Milk Composition Predictor
============================================================
Runs 30 trials to find the best hyperparameters for the neural network.
Saves the best hyperparameters to best_params.json and best model to tuned_model.keras.

Run with:
    python tune_optuna.py
"""

import numpy as np
import pandas as pd
import optuna
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json

# ---------------------------------------------------------------
# 1. Load and preprocess the dataset (same as training notebook)
# ---------------------------------------------------------------
print("Loading dataset...")
df = pd.read_csv(r"D:\ML Zoom Camp\Zenodo+Tensorflow\data_table.csv")

# Build spectral column names
def block(prefix, n=256):
    return [f"Trans_{prefix}_{i}" for i in range(1, n + 1)]

dark   = df[block("Dark")].values.astype(np.float64)
sample = df[block("Sample")].values.astype(np.float64)
white  = df[block("White")].values.astype(np.float64)

# Absorbance
eps = 1e-6
I = (sample - dark) / (white - dark + eps)
I = np.clip(I, eps, 1.0)
X = -np.log10(I).astype(np.float32)

# SNV normalization
def snv(x):
    mean = x.mean(axis=1, keepdims=True)
    std  = x.std(axis=1, keepdims=True) + 1e-8
    return (x - mean) / std

X = snv(X)

# Targets
targets = ['Fat', 'Prot', 'Lact']
y = df[targets].values.astype(np.float32)

# Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# Scale targets
scaler = StandardScaler().fit(y_train)
y_train_s = scaler.transform(y_train).astype(np.float32)
y_val_s   = scaler.transform(y_val).astype(np.float32)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ---------------------------------------------------------------
# 2. Define the Optuna objective function
# ---------------------------------------------------------------
def objective(trial):
    # Ask Optuna to suggest hyperparameters
    lr          = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    layer1      = trial.suggest_categorical("layer1_size", [64, 128, 256])
    layer2      = trial.suggest_categorical("layer2_size", [32, 64, 128])
    dropout     = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size  = trial.suggest_categorical("batch_size", [16, 32, 64])

    # Build model with these hyperparameters
    model = keras.Sequential([
        layers.Input(shape=(256,)),
        layers.Dense(layer1, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(layer2, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(32, activation='relu'),
        layers.Dense(3)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=['mae']
    )

    # Train with EarlyStopping (short patience for speed)
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train_s,
        validation_data=(X_val, y_val_s),
        epochs=100,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )

    # Return best validation loss (Optuna minimizes this)
    best_val_loss = min(history.history['val_loss'])
    return best_val_loss


# ---------------------------------------------------------------
# 3. Run the study
# ---------------------------------------------------------------
print("\nStarting Optuna study — 30 trials...")
print("This will take 30-60 minutes. You can watch progress below.\n")

study = optuna.create_study(
    direction="minimize",
    study_name="milk_model_tuning",
    sampler=optuna.samplers.TPESampler(seed=42)
)

# Show progress as each trial completes
def print_callback(study, trial):
    print(f"Trial {trial.number + 1}/30 | val_loss={trial.value:.4f} | best={study.best_value:.4f}")

study.optimize(objective, n_trials=30, callbacks=[print_callback])

# ---------------------------------------------------------------
# 4. Save the best hyperparameters
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("BEST TRIAL")
print("=" * 60)
print(f"Best val_loss: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# Save to JSON
with open("best_params.json", "w") as f:
    json.dump(study.best_params, f, indent=2)

print("\nSaved best params to best_params.json")
print("Retrain with these params in the next step (train_tuned.py)")