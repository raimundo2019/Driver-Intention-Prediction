#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils_9_features_20x9.py

Funciones comunes para modelos con 9 caracteristicas y ventanas de 20 pasos.
Genera metricas, matriz de confusion, curvas ROC y curvas de entrenamiento
separadas en:
    - accuracy_<modelo>.png
    - loss_<modelo>.png
    - training_curves_<modelo>.png
"""

import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    auc,
)

SEED = 50
TIMESTEPS = 20
N_FEATURES = 9
N_CLASSES = 3
CLASS_NAMES = ["GR", "SD", "GI"]

TEST_SIZE = 0.20
VAL_SIZE = 0.20

X_PATH = "dato_expandido.csv"
Y_PATH = "salida_expandido.csv"


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    try:
        import tensorflow as tf
        tf.random.set_seed(SEED)
    except Exception:
        pass


def load_20x9_data():
    X_raw = pd.read_csv(X_PATH, header=None).values.astype(float)
    Y = pd.read_csv(Y_PATH, header=None).values.astype(float)

    if X_raw.shape[1] != N_FEATURES:
        raise ValueError(f"dato_expandido.csv debe tener 9 columnas. Tiene {X_raw.shape[1]}.")

    if Y.shape[1] != N_CLASSES:
        raise ValueError(f"salida_expandido.csv debe tener 3 columnas. Tiene {Y.shape[1]}.")

    n_samples = Y.shape[0]
    expected_rows = n_samples * TIMESTEPS

    if X_raw.shape[0] != expected_rows:
        raise ValueError(
            f"Cantidad de filas incompatible. dato_expandido.csv tiene {X_raw.shape[0]} filas, "
            f"pero se esperaban {expected_rows} = {n_samples} muestras x {TIMESTEPS} pasos."
        )

    X = X_raw.reshape(n_samples, TIMESTEPS, N_FEATURES)
    y_labels = np.argmax(Y, axis=1)

    print("X reconstruido:", X.shape)
    print("Y:", Y.shape)
    print("Distribucion de clases:", np.unique(y_labels, return_counts=True))

    return X, Y, y_labels


def split_data(X, Y, y_labels):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        Y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y_labels,
    )

    y_trainval_lbl = np.argmax(y_trainval, axis=1)

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=y_trainval_lbl,
    )

    print("Train:", X_train.shape, y_train.shape)
    print("Val:  ", X_val.shape, y_val.shape)
    print("Test: ", X_test.shape, y_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_3d(X_train, X_val, X_test):
    scaler = StandardScaler()

    train_shape = X_train.shape
    val_shape = X_val.shape
    test_shape = X_test.shape

    X_train_s = scaler.fit_transform(X_train.reshape(-1, N_FEATURES)).reshape(train_shape)
    X_val_s = scaler.transform(X_val.reshape(-1, N_FEATURES)).reshape(val_shape)
    X_test_s = scaler.transform(X_test.reshape(-1, N_FEATURES)).reshape(test_shape)

    return X_train_s, X_val_s, X_test_s, scaler


def flatten_3d(X):
    return X.reshape(X.shape[0], TIMESTEPS * N_FEATURES)


def plot_training_history(history, out_prefix):
    """Guarda accuracy y loss en graficos separados y tambien juntos."""
    hist = history.history
    epochs = range(1, len(hist.get("loss", [])) + 1)

    pd.DataFrame(hist).to_csv(f"history_{out_prefix}.csv", index=False)

    # Accuracy: entrenamiento y validacion en un mismo grafico.
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, hist.get("accuracy", []), marker="o", label="Training Accuracy")
    plt.plot(epochs, hist.get("val_accuracy", []), marker="o", label="Validation Accuracy")
    plt.title(f"Training Accuracy vs Validation Accuracy - {out_prefix}")
    plt.xlabel("Epoca")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"accuracy_{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Loss: entrenamiento y validacion en un mismo grafico.
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, hist.get("loss", []), marker="o", label="Training Loss")
    plt.plot(epochs, hist.get("val_loss", []), marker="o", label="Validation Loss")
    plt.title(f"Training Loss vs Validation Loss - {out_prefix}")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"loss_{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Version conjunta, util para informe general.
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epochs, hist.get("accuracy", []), marker="o", label="Training Accuracy")
    ax[0].plot(epochs, hist.get("val_accuracy", []), marker="o", label="Validation Accuracy")
    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("Epoca")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(epochs, hist.get("loss", []), marker="o", label="Training Loss")
    ax[1].plot(epochs, hist.get("val_loss", []), marker="o", label="Validation Loss")
    ax[1].set_title("Loss")
    ax[1].set_xlabel("Epoca")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"training_curves_{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_and_export(model_name, out_prefix, y_test, y_prob, history=None, extra_config=None):
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_true, y_pred)

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo", average="macro")

    print("\n==============================")
    print(model_name)
    print("==============================")
    print(f"Accuracy:        {acc:.6f}")
    print(f"Precision macro: {prec_macro:.6f}")
    print(f"Recall macro:    {rec_macro:.6f}")
    print(f"F1 macro:        {f1_macro:.6f}")
    print(f"AUC OVR macro:   {auc_ovr:.6f}")
    print(f"AUC OVO macro:   {auc_ovo:.6f}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    pd.DataFrame(
        cm,
        columns=["Pred_GR", "Pred_SD", "Pred_GI"],
        index=["True_GR", "True_SD", "True_GI"],
    ).to_csv(f"confusion_{out_prefix}.csv")

    pd.DataFrame(
        cm_norm,
        columns=["Pred_GR", "Pred_SD", "Pred_GI"],
        index=["True_GR", "True_SD", "True_GI"],
    ).to_csv(f"confusion_{out_prefix}_normalizada.csv")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=CLASS_NAMES)
    disp.plot(cmap=plt.cm.Blues, values_format=".3f")
    plt.title(f"Matriz de confusion normalizada - {model_name}")
    plt.tight_layout()
    plt.savefig(f"confusion_{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 6))
    auc_by_class = {}
    for i, class_name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_prob[:, i])
        auc_i = auc(fpr, tpr)
        auc_by_class[class_name] = auc_i
        plt.plot(fpr, tpr, label=f"{class_name} (AUC={auc_i:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"Curvas ROC - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"roc_{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close()

    if history is not None:
        plot_training_history(history, out_prefix)

    row = {
        "modelo": model_name,
        "features": N_FEATURES,
        "timesteps": TIMESTEPS,
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_w,
        "recall_weighted": rec_w,
        "f1_weighted": f1_w,
        "roc_auc_ovr_macro": auc_ovr,
        "roc_auc_ovo_macro": auc_ovo,
        "auc_GR": auc_by_class["GR"],
        "auc_SD": auc_by_class["SD"],
        "auc_GI": auc_by_class["GI"],
        "cm_norm_00": cm_norm[0, 0],
        "cm_norm_01": cm_norm[0, 1],
        "cm_norm_02": cm_norm[0, 2],
        "cm_norm_10": cm_norm[1, 0],
        "cm_norm_11": cm_norm[1, 1],
        "cm_norm_12": cm_norm[1, 2],
        "cm_norm_20": cm_norm[2, 0],
        "cm_norm_21": cm_norm[2, 1],
        "cm_norm_22": cm_norm[2, 2],
    }

    pd.DataFrame([row]).to_csv(f"metricas_resumen_{out_prefix}.csv", index=False)

    config = {
        "modelo": model_name,
        "datos": {
            "X": X_PATH,
            "Y": Y_PATH,
            "features": N_FEATURES,
            "timesteps": TIMESTEPS,
        },
        "split": {
            "test_size": TEST_SIZE,
            "val_size_from_trainval": VAL_SIZE,
            "seed": SEED,
        },
        "salida": "softmax para tres clases excluyentes",
        "loss": "categorical_crossentropy en redes neuronales",
    }

    if extra_config:
        config.update(extra_config)

    with open(f"config_{out_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print("\nArchivos generados:")
    print(f" - metricas_resumen_{out_prefix}.csv")
    print(f" - confusion_{out_prefix}.png")
    print(f" - confusion_{out_prefix}.csv")
    print(f" - confusion_{out_prefix}_normalizada.csv")
    print(f" - roc_{out_prefix}.png")
    if history is not None:
        print(f" - accuracy_{out_prefix}.png")
        print(f" - loss_{out_prefix}.png")
        print(f" - training_curves_{out_prefix}.png")
        print(f" - history_{out_prefix}.csv")
