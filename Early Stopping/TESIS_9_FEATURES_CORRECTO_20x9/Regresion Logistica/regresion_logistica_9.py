#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regresion_logistica_9_corregido.py

Regresion Logistica para 9 caracteristicas con entrada temporal 20 x 9 aplanada.

Correcciones incorporadas:
    - Se usa el conjunto de validacion para seleccionar el mejor valor de C.
    - Se generan curvas juntas:
        * Training Accuracy vs Validation Accuracy
        * Training Loss vs Validation Loss
    - Se implementa una seleccion tipo Early Stopping sobre la busqueda de C.

Nota importante:
    La regresion logistica no entrena por epocas como una red neuronal.
    Por eso, el eje horizontal de las curvas no representa epocas, sino valores
    sucesivos de C, que controla la regularizacion del modelo.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from utils_9_features_20x9 import (
    set_seed,
    load_20x9_data,
    split_data,
    scale_3d,
    flatten_3d,
    evaluate_and_export,
    CLASS_NAMES
)


OUT_PREFIX = "regresion_logistica_9"
MAX_ITER = 5000
SOLVER = "lbfgs"
PATIENCE = 6
TOL_IMPROVEMENT = 1e-5

# C es el inverso de la regularizacion.
# C pequeno -> mas regularizacion.
# C grande  -> menos regularizacion.
C_VALUES = np.logspace(-4, 4, 25)


def guardar_curvas_regresion_logistica(history_df, out_prefix):
    """Guarda curvas de accuracy y loss para entrenamiento y validacion."""

    x = np.arange(1, len(history_df) + 1)
    x_labels = [f"{c:.1e}" for c in history_df["C"]]

    # Grafico 1: accuracy entrenamiento y validacion juntas
    plt.figure(figsize=(8, 5))
    plt.plot(x, history_df["train_accuracy"], marker="o", label="Training Accuracy")
    plt.plot(x, history_df["val_accuracy"], marker="o", label="Validation Accuracy")
    plt.xlabel("Iteracion de busqueda de C")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy vs Validation Accuracy - Regresion Logistica 9")
    plt.xticks(x, x_labels, rotation=45, ha="right")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"accuracy_{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Grafico 2: loss entrenamiento y validacion juntas
    plt.figure(figsize=(8, 5))
    plt.plot(x, history_df["train_loss"], marker="o", label="Training Loss")
    plt.plot(x, history_df["val_loss"], marker="o", label="Validation Loss")
    plt.xlabel("Iteracion de busqueda de C")
    plt.ylabel("Log Loss")
    plt.title("Training Loss vs Validation Loss - Regresion Logistica 9")
    plt.xticks(x, x_labels, rotation=45, ha="right")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"loss_{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Grafico combinado para informe
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    ax[0].plot(x, history_df["train_accuracy"], marker="o", label="Training Accuracy")
    ax[0].plot(x, history_df["val_accuracy"], marker="o", label="Validation Accuracy")
    ax[0].set_xlabel("Iteracion de busqueda de C")
    ax[0].set_ylabel("Accuracy")
    ax[0].set_title("Exactitud")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(x_labels, rotation=45, ha="right")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(x, history_df["train_loss"], marker="o", label="Training Loss")
    ax[1].plot(x, history_df["val_loss"], marker="o", label="Validation Loss")
    ax[1].set_xlabel("Iteracion de busqueda de C")
    ax[1].set_ylabel("Log Loss")
    ax[1].set_title("Perdida")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(x_labels, rotation=45, ha="right")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    fig.suptitle("Curvas de entrenamiento y validacion - Regresion Logistica 9", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"training_curves_{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


set_seed()

X, Y, y_labels = load_20x9_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, Y, y_labels)
X_train, X_val, X_test, scaler = scale_3d(X_train, X_val, X_test)

X_train_f = flatten_3d(X_train)
X_val_f = flatten_3d(X_val)
X_test_f = flatten_3d(X_test)

y_train_lbl = y_train.argmax(axis=1)
y_val_lbl = y_val.argmax(axis=1)

best_model = None
best_c = None
best_val_loss = np.inf
best_iteration = 0
wait = 0
history_rows = []

print("\nBusqueda de C con control tipo Early Stopping")
print("C pequeno implica mayor regularizacion. C grande implica menor regularizacion.\n")

for iteration, c_value in enumerate(C_VALUES, start=1):
    model = LogisticRegression(
        C=float(c_value),
        max_iter=MAX_ITER,
        solver=SOLVER,
        multi_class="auto"
    )

    model.fit(X_train_f, y_train_lbl)

    train_prob = model.predict_proba(X_train_f)
    val_prob = model.predict_proba(X_val_f)

    train_pred = np.argmax(train_prob, axis=1)
    val_pred = np.argmax(val_prob, axis=1)

    train_acc = accuracy_score(y_train_lbl, train_pred)
    val_acc = accuracy_score(y_val_lbl, val_pred)
    train_loss = log_loss(y_train_lbl, train_prob, labels=[0, 1, 2])
    val_loss = log_loss(y_val_lbl, val_prob, labels=[0, 1, 2])

    improved = val_loss < (best_val_loss - TOL_IMPROVEMENT)

    history_rows.append({
        "iteration": iteration,
        "C": float(c_value),
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "improved": improved
    })

    print(
        f"Iteracion {iteration:02d} | C={c_value:.6g} | "
        f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | "
        f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
    )

    if improved:
        best_val_loss = val_loss
        best_c = float(c_value)
        best_model = model
        best_iteration = iteration
        wait = 0
        joblib.dump(best_model, f"mejor_modelo_{OUT_PREFIX}.pkl")
    else:
        wait += 1

    if wait >= PATIENCE:
        print(f"\nEarly Stopping activado en la iteracion {iteration}.")
        print(f"Mejor iteracion: {best_iteration} | Mejor C: {best_c} | Mejor val_loss: {best_val_loss:.6f}")
        break

history_df = pd.DataFrame(history_rows)
history_df.to_csv(f"history_{OUT_PREFIX}.csv", index=False)
guardar_curvas_regresion_logistica(history_df, OUT_PREFIX)

# Se usa el mejor modelo seleccionado por perdida de validacion.
model = best_model

y_prob = model.predict_proba(X_test_f)

joblib.dump(model, f"modelo_{OUT_PREFIX}.pkl")
joblib.dump(scaler, f"scaler_{OUT_PREFIX}.pkl")

evaluate_and_export(
    "Regresion Logistica 9 caracteristicas",
    OUT_PREFIX,
    y_test,
    y_prob,
    extra_config={
        "entrada_modelo": "20x9 aplanado a 180 variables",
        "modelo": "Regresion Logistica",
        "regularizacion": "L2 por defecto en LogisticRegression de scikit-learn",
        "C_seleccionado": best_c,
        "mejor_iteracion": best_iteration,
        "early_stopping": True,
        "criterio_early_stopping": "detener busqueda de C si val_loss no mejora durante PATIENCE iteraciones",
        "patience": PATIENCE,
        "nota": "En regresion logistica no existen epocas como en redes neuronales; las curvas se construyen variando C."
    }
)

print("\nArchivos adicionales generados:")
print(f" - accuracy_{OUT_PREFIX}.png")
print(f" - loss_{OUT_PREFIX}.png")
print(f" - training_curves_{OUT_PREFIX}.png")
print(f" - history_{OUT_PREFIX}.csv")
print(f" - mejor_modelo_{OUT_PREFIX}.pkl")
print(f" - modelo_{OUT_PREFIX}.pkl")
print(f" - scaler_{OUT_PREFIX}.pkl")
