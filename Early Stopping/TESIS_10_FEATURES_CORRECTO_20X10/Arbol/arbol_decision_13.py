#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arbol_decision_13_earlystopping.py

Arbol de Decision para 13 caracteristicas.

Objetivo:
    - Generar imagen con curvas de entrenamiento:
        1) accuracy y val_accuracy juntas
        2) loss y val_loss juntas
    - Usar criterio tipo Early Stopping para evitar sobreajuste.
    - Permitir configurar manualmente EPOCHS y PATIENCE.

IMPORTANTE:
    Un arbol de decision de scikit-learn NO entrena por epocas reales como una red neuronal.
    Para visualizar una evolucion tipo epocas, se entrenan varios arboles aumentando
    max_depth desde 1 hasta EPOCHS.

    Cada profundidad se toma como una "epoca/etapa".
    Early Stopping se aplica sobre val_loss.
"""

import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, log_loss
from sklearn.tree import DecisionTreeClassifier, plot_tree

from utils_13_features_20x13 import (
    set_seed,
    load_20x13_data,
    split_data,
    scale_3d,
    flatten_3d,
    evaluate_and_export,
    CLASS_NAMES,
    TIMESTEPS,
    N_FEATURES,
    N_CLASSES
)


# ============================================================
# CONFIGURACION MANUAL
# Cambia estos valores vos mismo.
# ============================================================

EPOCHS = 80
PATIENCE = 10
MIN_DELTA = 1e-4
RANDOM_STATE = 50
OUT_PREFIX = "arbol_decision_13"


def align_proba(y_prob, model_classes, n_classes):
    aligned = pd.DataFrame(0.0, index=range(len(y_prob)), columns=range(n_classes))
    for col_idx, class_idx in enumerate(model_classes):
        aligned[int(class_idx)] = y_prob[:, col_idx]
    return aligned.to_numpy()


def guardar_curvas_entrenamiento(history, out_prefix, epochs_configuradas, patience):
    epochs_real = len(history["loss"])
    epochs_axis = list(range(1, epochs_real + 1))

    if epochs_real <= 30:
        ticks = epochs_axis
    else:
        step = max(1, epochs_real // 12)
        ticks = list(range(1, epochs_real + 1, step))
        if epochs_real not in ticks:
            ticks.append(epochs_real)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    ax[0].plot(epochs_axis, history["accuracy"], label="Train accuracy", linewidth=2)
    ax[0].plot(epochs_axis, history["val_accuracy"], label="Validation accuracy", linewidth=2)
    ax[0].set_title("Accuracy durante el entrenamiento")
    ax[0].set_xlabel("Epocas / profundidad maxima")
    ax[0].set_ylabel("Accuracy")
    ax[0].set_xticks(ticks)
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(epochs_axis, history["loss"], label="Train loss", linewidth=2)
    ax[1].plot(epochs_axis, history["val_loss"], label="Validation loss", linewidth=2)
    ax[1].set_title("Loss durante el entrenamiento")
    ax[1].set_xlabel("Epocas / profundidad maxima")
    ax[1].set_ylabel("Loss")
    ax[1].set_xticks(ticks)
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"Arbol de Decision 13 caracteristicas - Early Stopping "
        f"({epochs_real}/{epochs_configuradas} etapas, patience={patience})",
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    png_name = f"{out_prefix}_train_val_accuracy_loss_earlystopping_{epochs_real}_etapas.png"
    plt.savefig(png_name, dpi=300, bbox_inches="tight")
    plt.close()

    json_name = f"{out_prefix}_historial_earlystopping_{epochs_real}_etapas.json"
    with open(json_name, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    print(f"Imagen de entrenamiento guardada en: {png_name}")
    print(f"Historial guardado en: {json_name}")


set_seed()

X, Y, y_labels = load_20x13_data()

X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, Y, y_labels)

X_train, X_val, X_test, scaler = scale_3d(X_train, X_val, X_test)

X_train_f = flatten_3d(X_train)
X_val_f = flatten_3d(X_val)
X_test_f = flatten_3d(X_test)

y_train_lbl = y_train.argmax(axis=1)
y_val_lbl = y_val.argmax(axis=1)
y_test_lbl = y_test.argmax(axis=1)

history = {
    "epoch": [],
    "max_depth": [],
    "accuracy": [],
    "val_accuracy": [],
    "loss": [],
    "val_loss": []
}

best_val_loss = float("inf")
best_depth = None
best_model = None
wait = 0

for depth in range(1, EPOCHS + 1):
    model_stage = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        max_depth=depth
    )

    model_stage.fit(X_train_f, y_train_lbl)

    y_train_pred = model_stage.predict(X_train_f)
    y_val_pred = model_stage.predict(X_val_f)

    y_train_prob = align_proba(
        model_stage.predict_proba(X_train_f),
        model_stage.classes_,
        N_CLASSES
    )

    y_val_prob = align_proba(
        model_stage.predict_proba(X_val_f),
        model_stage.classes_,
        N_CLASSES
    )

    train_acc = accuracy_score(y_train_lbl, y_train_pred)
    val_acc = accuracy_score(y_val_lbl, y_val_pred)

    train_loss = log_loss(y_train_lbl, y_train_prob, labels=list(range(N_CLASSES)))
    val_loss = log_loss(y_val_lbl, y_val_prob, labels=list(range(N_CLASSES)))

    history["epoch"].append(depth)
    history["max_depth"].append(depth)
    history["accuracy"].append(float(train_acc))
    history["val_accuracy"].append(float(val_acc))
    history["loss"].append(float(train_loss))
    history["val_loss"].append(float(val_loss))

    print(
        f"Etapa {depth:03d}/{EPOCHS} | "
        f"max_depth={depth:03d} | "
        f"accuracy={train_acc:.4f} | "
        f"val_accuracy={val_acc:.4f} | "
        f"loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f}"
    )

    if val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = val_loss
        best_depth = depth
        best_model = model_stage
        wait = 0
        print(f"  Nueva mejor val_loss: {best_val_loss:.4f} en profundidad {best_depth}")
    else:
        wait += 1
        print(f"  Sin mejora en val_loss. Espera: {wait}/{PATIENCE}")

    if wait >= PATIENCE:
        print(
            f"\nEarly Stopping activado en etapa {depth}. "
            f"Mejor profundidad: {best_depth}, mejor val_loss: {best_val_loss:.4f}"
        )
        break


guardar_curvas_entrenamiento(history, OUT_PREFIX, EPOCHS, PATIENCE)

if best_model is None:
    raise RuntimeError(
        "No se encontro ningun mejor modelo. Revisar EPOCHS, PATIENCE o los datos de validacion."
    )

model = best_model

y_prob = align_proba(
    model.predict_proba(X_test_f),
    model.classes_,
    N_CLASSES
)

joblib.dump(model, f"modelo_{OUT_PREFIX}_earlystopping.pkl")
joblib.dump(scaler, f"scaler_{OUT_PREFIX}_earlystopping.pkl")

evaluate_and_export(
    "Arbol de Decision 13 caracteristicas con Early Stopping",
    f"{OUT_PREFIX}_earlystopping",
    y_test,
    y_prob,
    extra_config={
        "entrada_modelo": "20x13 aplanado a 260 variables",
        "archivo_entrada": "dato_expandido_con_entropia_ruido.csv",
        "archivo_salida": "salida_expandido.csv",
        "epochs_etapas_configuradas": EPOCHS,
        "epochs_etapas_ejecutadas": len(history["loss"]),
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "best_depth": best_depth,
        "best_val_loss": best_val_loss,
        "criterio_grafico": "max_depth creciente de 1 a EPOCHS; un arbol no entrena por epocas reales",
        "early_stopping": True
    }
)

feature_names = [
    f"x{j+1}_t{i+1:02d}"
    for i in range(TIMESTEPS)
    for j in range(N_FEATURES)
]

plt.figure(figsize=(30, 15))
plot_tree(
    model,
    feature_names=feature_names,
    class_names=CLASS_NAMES,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=7
)
plt.title(f"Arbol de decision - 13 caracteristicas - best max_depth={best_depth}")
plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}_earlystopping_mejor_arbol.png", dpi=300, bbox_inches="tight")
plt.close()

pd.DataFrame({
    "variable": feature_names,
    "importancia": model.feature_importances_
}).sort_values("importancia", ascending=False).to_csv(
    f"importancia_variables_{OUT_PREFIX}_earlystopping.csv",
    index=False
)

print("\nArchivos principales generados:")
print(f" - {OUT_PREFIX}_train_val_accuracy_loss_earlystopping_{len(history['loss'])}_etapas.png")
print(f" - modelo_{OUT_PREFIX}_earlystopping.pkl")
print(f" - metricas_resumen_{OUT_PREFIX}_earlystopping.csv")
print(f" - confusion_{OUT_PREFIX}_earlystopping.png")
print(f" - roc_{OUT_PREFIX}_earlystopping.png")
