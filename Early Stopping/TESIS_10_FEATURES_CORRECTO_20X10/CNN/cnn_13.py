#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cnn_13_earlystopping_train_val.py

CNN 1D para 13 caracteristicas.

Correcciones incorporadas:
1) Uso explicito de conjunto de validacion.
2) Uso de EarlyStopping sobre val_loss para evitar sobreajuste.
3) Restauracion automatica de los mejores pesos.
4) Grafico Training Accuracy + Validation Accuracy.
5) Grafico Training Loss + Validation Loss.
6) EPOCHS, BATCH_SIZE, PATIENCE y LEARNING_RATE configurables manualmente.
7) Guardado del historial en JSON convirtiendo float32 a float.
"""

import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from utils_13_features_20x13 import (
    set_seed,
    load_20x13_data,
    split_data,
    scale_3d,
    evaluate_and_export,
    TIMESTEPS,
    N_FEATURES,
    N_CLASSES
)


# ============================================================
# CONFIGURACION MANUAL
# Cambia estos valores vos mismo.
# ============================================================

EPOCHS = 120
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
PATIENCE = 15
MIN_DELTA = 1e-4

OUT_PREFIX = "cnn_13"


def guardar_grafico_entrenamiento(history, out_prefix, epochs_configuradas, patience):
    """
    Guarda una imagen con dos graficas:
        - Training Accuracy + Validation Accuracy
        - Training Loss + Validation Loss
    """
    history_dict = history.history
    epocas_reales = len(history_dict.get("loss", []))
    epocas = list(range(1, epocas_reales + 1))

    if epocas_reales <= 30:
        ticks = epocas
    else:
        paso = max(1, epocas_reales // 12)
        ticks = list(range(1, epocas_reales + 1, paso))
        if epocas_reales not in ticks:
            ticks.append(epocas_reales)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(
        epocas,
        history_dict.get("accuracy", []),
        label="Training accuracy",
        linewidth=2
    )
    axes[0].plot(
        epocas,
        history_dict.get("val_accuracy", []),
        label="Validation accuracy",
        linewidth=2
    )
    axes[0].set_title("Accuracy durante el entrenamiento")
    axes[0].set_xlabel("Epocas")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks(ticks)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        epocas,
        history_dict.get("loss", []),
        label="Training loss",
        linewidth=2
    )
    axes[1].plot(
        epocas,
        history_dict.get("val_loss", []),
        label="Validation loss",
        linewidth=2
    )
    axes[1].set_title("Loss durante el entrenamiento")
    axes[1].set_xlabel("Epocas")
    axes[1].set_ylabel("Loss")
    axes[1].set_xticks(ticks)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(
        f"CNN 13 caracteristicas - Early Stopping "
        f"({epocas_reales}/{epochs_configuradas} epocas, patience={patience})",
        fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    png_path = f"{out_prefix}_train_val_accuracy_loss_earlystopping_{epocas_reales}_epocas.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    history_json = {
        clave: [float(valor) for valor in valores]
        for clave, valores in history_dict.items()
    }

    json_path = f"{out_prefix}_historial_earlystopping_{epocas_reales}_epocas.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history_json, f, indent=4)

    print(f"Imagen de entrenamiento guardada en: {png_path}")
    print(f"Historial de entrenamiento guardado en: {json_path}")

    return png_path, json_path


set_seed()

X, Y, y_labels = load_20x13_data()

X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, Y, y_labels)

X_train, X_val, X_test, scaler = scale_3d(X_train, X_val, X_test)

model = Sequential([
    Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu",
        padding="same",
        input_shape=(TIMESTEPS, N_FEATURES)
    ),
    MaxPooling1D(pool_size=2),
    Dropout(0.25),

    Conv1D(
        filters=32,
        kernel_size=3,
        activation="relu",
        padding="same"
    ),
    GlobalAveragePooling1D(),

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
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=max(3, PATIENCE // 2),
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

guardar_grafico_entrenamiento(history, OUT_PREFIX, EPOCHS, PATIENCE)

model.save(f"modelo_{OUT_PREFIX}.keras")

y_prob = model.predict(X_test, verbose=0)

evaluate_and_export(
    "CNN 13 caracteristicas con Early Stopping",
    OUT_PREFIX,
    y_test,
    y_prob,
    history=history,
    extra_config={
        "arquitectura": "Conv1D(64) + MaxPooling1D + Dropout + Conv1D(32) + GlobalAveragePooling1D + Dense(32) + softmax",
        "entrada_modelo": "20 pasos x 13 variables",
        "archivo_entrada": "dato_expandido_con_entropia_ruido.csv",
        "archivo_salida": "salida_expandido.csv",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs_configuradas": EPOCHS,
        "epochs_ejecutadas": len(history.history.get("loss", [])),
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "early_stopping": True,
        "restore_best_weights": True
    }
)
