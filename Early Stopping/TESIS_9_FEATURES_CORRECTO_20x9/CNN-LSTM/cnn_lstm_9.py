#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cnn_lstm_9_corregido.py

Modelo hibrido CNN-LSTM para 9 caracteristicas con entrada temporal real 20 x 9.

Correcciones incorporadas:
    - Training Accuracy y Validation Accuracy en un mismo grafico.
    - Training Loss y Validation Loss en un mismo grafico.
    - EarlyStopping usando val_loss con restauracion de los mejores pesos.
    - ModelCheckpoint para guardar el mejor modelo segun val_loss.
    - ReduceLROnPlateau para bajar el learning rate si la validacion se estanca.
    - Exportacion del historial de entrenamiento.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from utils_9_features_20x9 import (
    set_seed,
    load_20x9_data,
    split_data,
    scale_3d,
    evaluate_and_export,
    TIMESTEPS,
    N_FEATURES,
    N_CLASSES,
)

# ============================================================
# CONFIGURACION MANUAL
# ============================================================

EPOCHS = 120
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
OUT_PREFIX = "cnn_lstm_9"

EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 1e-4


set_seed()

X, Y, y_labels = load_20x9_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, Y, y_labels)
X_train, X_val, X_test, scaler = scale_3d(X_train, X_val, X_test)

model = Sequential([
    Conv1D(
        filters=150,
        kernel_size=3,
        activation="relu",
        padding="same",
        input_shape=(TIMESTEPS, N_FEATURES),
    ),
    Conv1D(
        filters=100,
        kernel_size=3,
        activation="relu",
        padding="same",
    ),
    BatchNormalization(),
    Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu",
        padding="same",
    ),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.20),

    LSTM(units=128, return_sequences=True),
    Dropout(0.20),

    LSTM(units=64, return_sequences=False),
    Dropout(0.20),

    Dense(64, activation="relu"),
    Dropout(0.20),

    Dense(64, activation="relu"),
    Dropout(0.20),

    Dense(N_CLASSES, activation="softmax"),
])

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# Early Stopping:
# Detiene el entrenamiento cuando la perdida de validacion deja de mejorar.
# restore_best_weights=True conserva los pesos de la mejor epoca, no los de la ultima.
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=EARLY_STOPPING_PATIENCE,
    min_delta=EARLY_STOPPING_MIN_DELTA,
    restore_best_weights=True,
    verbose=1,
)

checkpoint = ModelCheckpoint(
    f"mejor_modelo_{OUT_PREFIX}.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=6,
    min_lr=1e-5,
    verbose=1,
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping, checkpoint, reduce_lr],
    verbose=1,
)

model.save(f"modelo_{OUT_PREFIX}.keras")
joblib.dump(scaler, f"scaler_{OUT_PREFIX}.pkl")

y_prob = model.predict(X_test, verbose=0)

evaluate_and_export(
    "CNN-LSTM 9 caracteristicas",
    OUT_PREFIX,
    y_test,
    y_prob,
    history=history,
    extra_config={
        "arquitectura": (
            "Conv1D(150) + Conv1D(100) + BatchNormalization + Conv1D(64) "
            "+ BatchNormalization + MaxPooling1D + LSTM(128) + LSTM(64) "
            "+ Dense(64) + Dense(64) + softmax"
        ),
        "entrada_modelo": "20 pasos x 9 variables",
        "tipo_modelo": "Hibrido CNN-LSTM",
        "epochs_maximas": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "early_stopping": {
            "monitor": "val_loss",
            "patience": EARLY_STOPPING_PATIENCE,
            "min_delta": EARLY_STOPPING_MIN_DELTA,
            "restore_best_weights": True,
        },
        "reduce_lr_on_plateau": {
            "monitor": "val_loss",
            "factor": 0.5,
            "patience": 6,
            "min_lr": 1e-5,
        },
        "graficos_entrenamiento": [
            f"accuracy_{OUT_PREFIX}.png",
            f"loss_{OUT_PREFIX}.png",
            f"training_curves_{OUT_PREFIX}.png",
        ],
    },
)
