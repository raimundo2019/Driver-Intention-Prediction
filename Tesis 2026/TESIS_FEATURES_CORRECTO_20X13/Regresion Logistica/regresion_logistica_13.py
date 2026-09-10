#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regresion_logistica_13_earlystopping_train_val.py

Regresion Logistica para 13 caracteristicas con entrada temporal 20 x 13 aplanada a 260 variables.

Correcciones incorporadas:
1) Uso explicito de conjunto de validacion.
2) Entrenamiento por epocas usando warm_start=True.
3) Criterio tipo Early Stopping sobre val_loss para evitar sobreajuste.
4) Seleccion del mejor modelo segun val_loss.
5) Grafico Training Accuracy + Validation Accuracy.
6) Grafico Training Loss + Validation Loss.
7) EPOCHS, PATIENCE y LEARNING_RATE configurables manualmente.
"""

import sys
import json
import copy
import warnings
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from utils_13_features_20x13 import (
    set_seed,
    load_20x13_data,
    split_data,
    scale_3d,
    flatten_3d,
    evaluate_and_export
)


# ============================================================
# CONFIGURACION MANUAL
# Cambia estos valores vos mismo.
# ============================================================

EPOCHS = 120
PATIENCE = 15
MIN_DELTA = 1e-4
LEARNING_RATE = "interno_saga"
RANDOM_STATE = 50

OUT_PREFIX = "regresion_logistica_13"


def guardar_grafico_entrenamiento(history, out_prefix, epochs_configuradas, patience):
    """
    Guarda una imagen con dos graficas:
        - Training Accuracy + Validation Accuracy
        - Training Loss + Validation Loss
    """
    epocas_reales = len(history["loss"])
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
        history["accuracy"],
        label="Training accuracy",
        linewidth=2
    )
    axes[0].plot(
        epocas,
        history["val_accuracy"],
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
        history["loss"],
        label="Training loss",
        linewidth=2
    )
    axes[1].plot(
        epocas,
        history["val_loss"],
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
        f"Regresion Logistica 13 caracteristicas - Early Stopping "
        f"({epocas_reales}/{epochs_configuradas} epocas, patience={patience})",
        fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    png_path = f"{out_prefix}_train_val_accuracy_loss_earlystopping_{epocas_reales}_epocas.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    json_path = f"{out_prefix}_historial_earlystopping_{epocas_reales}_epocas.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    print(f"Imagen de entrenamiento guardada en: {png_path}")
    print(f"Historial de entrenamiento guardado en: {json_path}")

    return png_path, json_path


set_seed()

warnings.filterwarnings("ignore", category=ConvergenceWarning)

X, Y, y_labels = load_20x13_data()

X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, Y, y_labels)

X_train, X_val, X_test, scaler = scale_3d(X_train, X_val, X_test)

X_train_f = flatten_3d(X_train)
X_val_f = flatten_3d(X_val)
X_test_f = flatten_3d(X_test)

y_train_lbl = y_train.argmax(axis=1)
y_val_lbl = y_val.argmax(axis=1)

classes = np.arange(y_train.shape[1])

# ============================================================
# MODELO
# ============================================================
# scikit-learn no tiene "epocas" como Keras.
# Para obtener curvas por epoca se usa:
#   - solver="saga"
#   - max_iter=1
#   - warm_start=True
# Cada llamada a fit realiza una iteracion adicional aproximada.

model = LogisticRegression(
    max_iter=1,
    solver="saga",
    multi_class="multinomial",
    warm_start=True,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

history = {
    "epoch": [],
    "accuracy": [],
    "val_accuracy": [],
    "loss": [],
    "val_loss": []
}

best_model = None
best_val_loss = float("inf")
best_epoch = None
wait = 0

for epoch in range(1, EPOCHS + 1):
    model.fit(X_train_f, y_train_lbl)

    y_train_prob = model.predict_proba(X_train_f)
    y_val_prob = model.predict_proba(X_val_f)

    train_pred = np.argmax(y_train_prob, axis=1)
    val_pred = np.argmax(y_val_prob, axis=1)

    train_acc = accuracy_score(y_train_lbl, train_pred)
    val_acc = accuracy_score(y_val_lbl, val_pred)

    train_loss = log_loss(y_train_lbl, y_train_prob, labels=classes)
    val_loss = log_loss(y_val_lbl, y_val_prob, labels=classes)

    history["epoch"].append(epoch)
    history["accuracy"].append(float(train_acc))
    history["val_accuracy"].append(float(val_acc))
    history["loss"].append(float(train_loss))
    history["val_loss"].append(float(val_loss))

    print(
        f"Epoca {epoch:03d}/{EPOCHS} | "
        f"accuracy={train_acc:.4f} | "
        f"val_accuracy={val_acc:.4f} | "
        f"loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f}"
    )

    if val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = val_loss
        best_epoch = epoch
        best_model = copy.deepcopy(model)
        wait = 0
        print(f"  Nueva mejor val_loss: {best_val_loss:.4f} en epoca {best_epoch}")
    else:
        wait += 1
        print(f"  Sin mejora en val_loss. Espera: {wait}/{PATIENCE}")

    if wait >= PATIENCE:
        print(
            f"\nEarly Stopping activado en epoca {epoch}. "
            f"Mejor epoca: {best_epoch}, mejor val_loss: {best_val_loss:.4f}"
        )
        break


guardar_grafico_entrenamiento(history, OUT_PREFIX, EPOCHS, PATIENCE)

# Se usa el mejor modelo segun val_loss.
model = best_model

y_prob = model.predict_proba(X_test_f)

joblib.dump(model, f"modelo_{OUT_PREFIX}_earlystopping.pkl")
joblib.dump(scaler, f"scaler_{OUT_PREFIX}_earlystopping.pkl")
joblib.dump(history, f"historial_{OUT_PREFIX}_earlystopping.pkl")

evaluate_and_export(
    "Regresion Logistica 13 caracteristicas con Early Stopping",
    f"{OUT_PREFIX}_earlystopping",
    y_test,
    y_prob,
    extra_config={
        "entrada_modelo": "20x13 aplanado a 260 variables",
        "archivo_entrada": "dato_expandido_con_entropia_ruido.csv",
        "archivo_salida": "salida_expandido.csv",
        "epochs_configuradas": EPOCHS,
        "epochs_ejecutadas": len(history["loss"]),
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "solver": "saga",
        "warm_start": True,
        "max_iter_por_epoca": 1,
        "early_stopping": True
    }
)

print("\nArchivos principales generados:")
print(f" - {OUT_PREFIX}_train_val_accuracy_loss_earlystopping_{len(history['loss'])}_epocas.png")
print(f" - modelo_{OUT_PREFIX}_earlystopping.pkl")
print(f" - metricas_resumen_{OUT_PREFIX}_earlystopping.csv")
print(f" - confusion_{OUT_PREFIX}_earlystopping.png")
print(f" - roc_{OUT_PREFIX}_earlystopping.png")
