#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lstm_9_corregido.py

Modelo LSTM para 9 caracteristicas con entrada temporal real 20 x 9.

Correcciones realizadas:
    - Se agrega EarlyStopping para detener el entrenamiento cuando val_loss deja de mejorar.
    - Se conserva ReduceLROnPlateau para reducir la tasa de aprendizaje si el modelo se estanca.
    - Se guarda el mejor modelo segun val_loss mediante ModelCheckpoint.
    - Se generan graficos comparativos:
        1) Training Accuracy junto con Validation Accuracy.
        2) Training Loss junto con Validation Loss.
        3) Figura combinada con ambas curvas.
    - Se guarda el historial completo en CSV y JSON.
"""

import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
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
# CONFIGURACION MANUAL
# ============================================================

EPOCHS = 120
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
OUT_PREFIX = "lstm_9"

EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 7
MIN_LR = 1e-5


def guardar_curvas_entrenamiento(history, out_prefix):
    """
    Guarda tres archivos PNG:
        - accuracy_<out_prefix>.png
        - loss_<out_prefix>.png
        - training_curves_<out_prefix>.png
    Tambien guarda el historial en CSV y JSON.
    """
    history_dict = history.history
    epocas_reales = len(history_dict.get("loss", []))
    epocas = list(range(1, epocas_reales + 1))

    if epocas_reales <= 30:
        ticks_epocas = epocas
    else:
        paso = max(1, epocas_reales // 12)
        ticks_epocas = list(range(1, epocas_reales + 1, paso))
        if epocas_reales not in ticks_epocas:
            ticks_epocas.append(epocas_reales)

    pd.DataFrame(history_dict).to_csv(f"history_{out_prefix}.csv", index=False)

    history_json = {
        clave: [float(valor) for valor in valores]
        for clave, valores in history_dict.items()
    }
    with open(f"history_{out_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(history_json, f, indent=4)

    # Accuracy entrenamiento + validacion
    plt.figure(figsize=(8, 5))
    plt.plot(epocas, history_dict.get("accuracy", []), linewidth=2, label="Training Accuracy")
    plt.plot(epocas, history_dict.get("val_accuracy", []), linewidth=2, label="Validation Accuracy")
    plt.title("Training Accuracy vs Validation Accuracy - LSTM 9 caracteristicas")
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.xticks(ticks_epocas)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"accuracy_{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Loss entrenamiento + validacion
    plt.figure(figsize=(8, 5))
    plt.plot(epocas, history_dict.get("loss", []), linewidth=2, label="Training Loss")
    plt.plot(epocas, history_dict.get("val_loss", []), linewidth=2, label="Validation Loss")
    plt.title("Training Loss vs Validation Loss - LSTM 9 caracteristicas")
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.xticks(ticks_epocas)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"loss_{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figura combinada
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    ax[0].plot(epocas, history_dict.get("accuracy", []), linewidth=2, label="Training Accuracy")
    ax[0].plot(epocas, history_dict.get("val_accuracy", []), linewidth=2, label="Validation Accuracy")
    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("Epocas")
    ax[0].set_ylabel("Accuracy")
    ax[0].set_xticks(ticks_epocas)
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(epocas, history_dict.get("loss", []), linewidth=2, label="Training Loss")
    ax[1].plot(epocas, history_dict.get("val_loss", []), linewidth=2, label="Validation Loss")
    ax[1].set_title("Loss")
    ax[1].set_xlabel("Epocas")
    ax[1].set_ylabel("Loss")
    ax[1].set_xticks(ticks_epocas)
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    fig.suptitle(f"Curvas de entrenamiento - LSTM 9 caracteristicas ({epocas_reales}/{EPOCHS} epocas)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"training_curves_{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nCurvas de entrenamiento generadas:")
    print(f" - accuracy_{out_prefix}.png")
    print(f" - loss_{out_prefix}.png")
    print(f" - training_curves_{out_prefix}.png")
    print(f" - history_{out_prefix}.csv")
    print(f" - history_{out_prefix}.json")


set_seed()

X, Y, y_labels = load_20x9_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, Y, y_labels)
X_train, X_val, X_test, scaler = scale_3d(X_train, X_val, X_test)

model = Sequential([
    LSTM(
        250,
        return_sequences=True,
        input_shape=(TIMESTEPS, N_FEATURES)
    ),
    BatchNormalization(),
    Dropout(0.25),

    LSTM(
        200,
        return_sequences=True
    ),
    BatchNormalization(),
    Dropout(0.25),

    LSTM(
        200,
        return_sequences=False
    ),
    BatchNormalization(),
    Dropout(0.25),

    Dense(300, activation="sigmoid"),
    Dropout(0.20),

    Dense(250, activation="sigmoid"),
    Dropout(0.20), 

    Dense(150, activation="sigmoid"),
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
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=REDUCE_LR_PATIENCE,
        min_lr=MIN_LR,
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
    "LSTM 9 caracteristicas",
    OUT_PREFIX,
    y_test,
    y_prob,
    history=history,
    extra_config={
        "arquitectura": "LSTM(250) + LSTM(200) + Dense(150) + Dense(100) + softmax",
        "entrada_modelo": "20 pasos x 9 variables",
        "tipo_modelo": "LSTM",
        "epochs_configuradas": EPOCHS,
        "epochs_ejecutadas": len(history.history.get("loss", [])),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "early_stopping": True,
        "early_stopping_monitor": "val_loss",
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "restore_best_weights": True,
        "reduce_lr_on_plateau": True,
        "reduce_lr_patience": REDUCE_LR_PATIENCE,
        "min_lr": MIN_LR
    }
)
