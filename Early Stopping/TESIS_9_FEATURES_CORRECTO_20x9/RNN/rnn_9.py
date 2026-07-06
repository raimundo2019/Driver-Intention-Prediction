#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rnn_9_corregido.py

Modelo RNN para 9 caracteristicas con entrada temporal real 20 x 9.

Correcciones incorporadas:
    - EarlyStopping sobre val_loss para cortar el entrenamiento cuando deja de mejorar.
    - ModelCheckpoint para guardar el mejor modelo segun val_loss.
    - ReduceLROnPlateau para reducir la tasa de aprendizaje si val_loss se estanca.
    - Grafico conjunto: Training Accuracy vs Validation Accuracy.
    - Grafico conjunto: Training Loss vs Validation Loss.
    - Conserva el grafico combinado generado por evaluate_and_export.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
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
    N_CLASSES
)

# ============================================================
# CONFIGURACION
# ============================================================
EPOCHS = 120
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
OUT_PREFIX = "rnn_9"


def guardar_curvas_entrenamiento(history, out_prefix):
    """Guarda accuracy y loss con entrenamiento y validacion en graficos separados."""
    history_dict = history.history
    epocas_reales = len(history_dict.get("loss", []))
    epocas = list(range(1, epocas_reales + 1))

    # Historial completo en CSV para trazabilidad.
    pd.DataFrame(history_dict).to_csv(f"history_{out_prefix}.csv", index=False)

    # Accuracy: entrenamiento y validacion juntos.
    plt.figure(figsize=(8, 5))
    plt.plot(epocas, history_dict.get("accuracy", []), label="Training Accuracy", linewidth=2)
    plt.plot(epocas, history_dict.get("val_accuracy", []), label="Validation Accuracy", linewidth=2)
    plt.title("Training Accuracy vs Validation Accuracy - RNN 9 caracteristicas")
    plt.xlabel("Epoca")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"accuracy_{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Loss: entrenamiento y validacion juntos.
    plt.figure(figsize=(8, 5))
    plt.plot(epocas, history_dict.get("loss", []), label="Training Loss", linewidth=2)
    plt.plot(epocas, history_dict.get("val_loss", []), label="Validation Loss", linewidth=2)
    plt.title("Training Loss vs Validation Loss - RNN 9 caracteristicas")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"loss_{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nCurvas de entrenamiento generadas:")
    print(f" - accuracy_{out_prefix}.png")
    print(f" - loss_{out_prefix}.png")
    print(f" - history_{out_prefix}.csv")


set_seed()

X, Y, y_labels = load_20x9_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, Y, y_labels)
X_train, X_val, X_test, scaler = scale_3d(X_train, X_val, X_test)

model = Sequential([
    SimpleRNN(
        64,
        input_shape=(TIMESTEPS, N_FEATURES),
        dropout=0.10,
        recurrent_dropout=0.05
    ),
    Dropout(0.25),
    Dense(32, activation="relu"),
    Dropout(0.20),
    Dense(N_CLASSES, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=15,
        min_delta=1e-4,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=7,
        min_lr=1e-5,
        verbose=1
    ),
    ModelCheckpoint(
        f"mejor_modelo_{OUT_PREFIX}.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

guardar_curvas_entrenamiento(history, OUT_PREFIX)

model.save(f"modelo_{OUT_PREFIX}.keras")

y_prob = model.predict(X_test, verbose=0)

evaluate_and_export(
    "RNN 9 caracteristicas",
    OUT_PREFIX,
    y_test,
    y_prob,
    history=history,
    extra_config={
        "arquitectura": "SimpleRNN(64) + Dense(32) + softmax",
        "entrada_modelo": "20 pasos x 9 variables",
        "tipo_modelo": "RNN",
        "epochs_configuradas": EPOCHS,
        "epochs_ejecutadas": len(history.history.get("loss", [])),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "early_stopping": True,
        "early_stopping_monitor": "val_loss",
        "early_stopping_patience": 15,
        "restore_best_weights": True
    }
)
