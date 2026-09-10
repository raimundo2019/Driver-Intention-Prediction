#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from utils_9_features_20x9 import (
    set_seed, load_20x9_data, split_data, scale_3d,
    evaluate_and_export, TIMESTEPS, N_FEATURES, N_CLASSES
)

set_seed()

X, Y, y_labels = load_20x9_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, Y, y_labels)
X_train, X_val, X_test, scaler = scale_3d(X_train, X_val, X_test)

model = Sequential([
    Conv1D(
        64,
        kernel_size=3,
        activation="relu",
        padding="same",
        input_shape=(TIMESTEPS, N_FEATURES),
    ),
    MaxPooling1D(pool_size=2),
    Dropout(0.25),

    Conv1D(32, kernel_size=3, activation="relu", padding="same"),
    GlobalAveragePooling1D(),

    Dense(32, activation="relu"),
    Dropout(0.20),
    Dense(N_CLASSES, activation="softmax"),
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Early Stopping:
# Detiene el entrenamiento cuando la perdida de validacion deja de mejorar.
# restore_best_weights=True conserva los pesos de la mejor epoca, no los de la ultima.
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=15,
    min_delta=1e-4,
    restore_best_weights=True,
    verbose=1,
)

checkpoint = ModelCheckpoint(
    "mejor_modelo_cnn_9.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=120,
    batch_size=32,
    callbacks=[early_stopping, checkpoint],
    verbose=1,
)

model.save("modelo_cnn_9.keras")
joblib.dump(scaler, "scaler_cnn_9.pkl")

y_prob = model.predict(X_test, verbose=0)

evaluate_and_export(
    "CNN 9 caracteristicas",
    "cnn_9",
    y_test,
    y_prob,
    history=history,
    extra_config={
        "arquitectura": "Conv1D + MaxPooling1D + Dropout + Conv1D + GlobalAveragePooling1D + Dense + Dropout + softmax",
        "early_stopping": {
            "monitor": "val_loss",
            "patience": 15,
            "min_delta": 1e-4,
            "restore_best_weights": True,
        },
        "graficos_entrenamiento": [
            "accuracy_cnn_9.png",
            "loss_cnn_9.png",
            "training_curves_cnn_9.png",
        ],
    },
)
