"""
Retrain with Optuna's best hyperparameters
============================================
Loads best_params.json, retrains the model on the full training set,
evaluates on the test set, and saves the tuned model.

Run with:
    python train_tuned.py
"""

import numpy as np
import pandas as pd
import json
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. Load best hyperparameters from Optuna
# ---------------------------------------------------------------
with open("best_params.json", "r") as f:
    best = json.load(f)

print("=" * 60)
print("Retraining with Optuna's best hyperparameters:")
print("=" * 60)
for k, v in best.items():
    print(f"  {k}: {v}")
print("=" * 60)

# ---------------------------------------------------------------
# 2. Load and preprocess the dataset
# ---------------------------------------------------------------
print("\nLoading dataset...")
df = pd.read_csv(r"D:\ML Zoom Camp\Zenodo+Tensorflow\data_table.csv")

def block(prefix, n=256):
    return [f"Trans_{prefix}_{i}" for i in range(1, n + 1)]

dark   = df[block("Dark")].values.astype(np.float64)
sample = df[block("Sample")].values.astype(np.float64)
white  = df[block("White")].values.astype(np.float64)

eps = 1e-6
I = (sample - dark) / (white - dark + eps)
I = np.clip(I, eps, 1.0)
X = -np.log10(I).astype(np.float32)

def snv(x):
    mean = x.mean(axis=1, keepdims=True)
    std  = x.std(axis=1, keepdims=True) + 1e-8
    return (x - mean) / std

X = snv(X)

targets = ['Fat', 'Prot', 'Lact']
y = df[targets].values.astype(np.float32)

# Same split as before (random_state=42 ensures consistency)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

scaler = StandardScaler().fit(y_train)
y_train_s = scaler.transform(y_train).astype(np.float32)
y_val_s   = scaler.transform(y_val).astype(np.float32)
y_test_s  = scaler.transform(y_test).astype(np.float32)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ---------------------------------------------------------------
# 3. Build model with best hyperparameters
# ---------------------------------------------------------------
model = keras.Sequential([
    layers.Input(shape=(256,)),
    layers.Dense(best['layer1_size'], activation='relu'),
    layers.Dropout(best['dropout']),
    layers.Dense(best['layer2_size'], activation='relu'),
    layers.Dropout(best['dropout']),
    layers.Dense(32, activation='relu'),
    layers.Dense(3)
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=best['learning_rate']),
    loss='mse',
    metrics=['mae', keras.metrics.RootMeanSquaredError(name='rmse')]
)

model.summary()

# ---------------------------------------------------------------
# 4. Train with the full callback set
# ---------------------------------------------------------------
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=40,
        restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=15,
        min_lr=1e-7, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'tuned_model.keras', monitor='val_loss',
        save_best_only=True, verbose=1
    ),
]

print("\nTraining with tuned hyperparameters...")
history = model.fit(
    X_train, y_train_s,
    validation_data=(X_val, y_val_s),
    epochs=300,
    batch_size=best['batch_size'],
    callbacks=callbacks,
    verbose=1
)

# ---------------------------------------------------------------
# 5. Evaluate on test set
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST SET EVALUATION (TUNED MODEL)")
print("=" * 60)

# Load best saved model
best_model = keras.models.load_model('tuned_model.keras')

# Predict
y_pred_s = best_model.predict(X_test, verbose=0)
y_pred = scaler.inverse_transform(y_pred_s)

# Metrics per target
print(f"\n{'Target':<10} {'RMSE':>10} {'MAE':>10} {'R2':>10}")
print("-" * 45)
results = {}
for i, name in enumerate(targets):
    rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
    mae  = mean_absolute_error(y_test[:, i], y_pred[:, i])
    r2   = r2_score(y_test[:, i], y_pred[:, i])
    results[name] = {'rmse': rmse, 'mae': mae, 'r2': r2}
    print(f"{name:<10} {rmse:>10.4f} {mae:>10.4f} {r2:>10.4f}")

# ---------------------------------------------------------------
# 6. Save results
# ---------------------------------------------------------------
with open("tuning_results.json", "w") as f:
    json.dump({'best_params': best, 'test_results': results}, f, indent=2)

print("\nSaved results to tuning_results.json")
print("Saved tuned model to tuned_model.keras")

# ---------------------------------------------------------------
# 7. Plot training history
# ---------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history.history['loss'], label='Train')
ax[0].plot(history.history['val_loss'], label='Validation')
ax[0].set_title('MSE Loss (Tuned Model)')
ax[0].set_xlabel('Epoch'); ax[0].legend(); ax[0].grid(alpha=0.3)

ax[1].plot(history.history['rmse'], label='Train')
ax[1].plot(history.history['val_rmse'], label='Validation')
ax[1].set_title('RMSE (Tuned Model)')
ax[1].set_xlabel('Epoch'); ax[1].legend(); ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('tuned_training_history.png', dpi=150)
print("Saved plot: tuned_training_history.png")