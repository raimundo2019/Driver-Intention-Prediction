#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cnn_rnn_9_mejorado_corregido.py

Modelo CNN-RNN mejorado para 9 caracteristicas con entrada temporal real 20 x 9.

Correcciones incorporadas:
    - EarlyStopping con restore_best_weights=True.
    - ModelCheckpoint para guardar el mejor modelo segun val_loss.
    - ReduceLROnPlateau para reducir learning rate cuando val_loss se estanca.
    - Grafico Training Accuracy vs Validation Accuracy en una misma figura.
    - Grafico Training Loss vs Validation Loss en una misma figura.
    - Grafico combinado con Accuracy y Loss en dos paneles.
    - Historial guardado en CSV.

Archivos esperados en la misma carpeta de ejecucion:
    dato_expandido.csv
    salida_expandido.csv
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
    auc
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    SimpleRNN,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# ============================================================
# CONFIGURACION
# ============================================================
SEED = 50
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

X_PATH = "dato_expandido.csv"
Y_PATH = "salida_expandido.csv"

TIMESTEPS = 20
N_FEATURES = 9
N_CLASSES = 3
CLASS_NAMES = ["GR", "SD", "GI"]

TEST_SIZE = 0.20
VAL_SIZE = 0.20

EPOCHS = 180
BATCH_SIZE = 24
LEARNING_RATE = 7e-4
PATIENCE_EARLY_STOPPING = 20

OUT_PREFIX = "cnn_rnn_9_mejorado"


# ============================================================
# FUNCIONES AUXILIARES PARA GRAFICOS
# ============================================================
def guardar_curvas_entrenamiento(history, out_prefix):
    """Guarda curvas de accuracy y loss con entrenamiento y validacion juntas."""
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(f"history_{out_prefix}.csv", index=False)

    epochs_run = len(history.history.get("loss", []))
    epochs_axis = range(1, epochs_run + 1)

    # Accuracy: entrenamiento y validacion en una misma figura
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_axis, history.history.get("accuracy", []), label="Training Accuracy", linewidth=2)
    plt.plot(epochs_axis, history.history.get("val_accuracy", []), label="Validation Accuracy", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy vs Validation Accuracy - CNN-RNN 9")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"accuracy_{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Loss: entrenamiento y validacion en una misma figura
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_axis, history.history.get("loss", []), label="Training Loss", linewidth=2)
    plt.plot(epochs_axis, history.history.get("val_loss", []), label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Validation Loss - CNN-RNN 9")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"loss_{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figura combinada en dos paneles
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    ax[0].plot(epochs_axis, history.history.get("accuracy", []), label="Training Accuracy", linewidth=2)
    ax[0].plot(epochs_axis, history.history.get("val_accuracy", []), label="Validation Accuracy", linewidth=2)
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].set_title("Accuracy")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(epochs_axis, history.history.get("loss", []), label="Training Loss", linewidth=2)
    ax[1].plot(epochs_axis, history.history.get("val_loss", []), label="Validation Loss", linewidth=2)
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].set_title("Loss")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    fig.suptitle(f"Curvas de entrenamiento - CNN-RNN 9 mejorado - {epochs_run} epocas ejecutadas")
    plt.tight_layout()
    plt.savefig(f"training_curves_{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nCurvas de entrenamiento generadas:")
    print(f" - accuracy_{out_prefix}.png")
    print(f" - loss_{out_prefix}.png")
    print(f" - training_curves_{out_prefix}.png")
    print(f" - history_{out_prefix}.csv")


# ============================================================
# CARGA Y RECONSTRUCCION 20 x 9
# ============================================================
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


# ============================================================
# SPLIT TRAIN / VALIDATION / TEST
# ============================================================
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X,
    Y,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=y_labels
)

y_trainval_lbl = np.argmax(y_trainval, axis=1)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval,
    y_trainval,
    test_size=VAL_SIZE,
    random_state=SEED,
    stratify=y_trainval_lbl
)

print("Train:", X_train.shape, y_train.shape)
print("Val:  ", X_val.shape, y_val.shape)
print("Test: ", X_test.shape, y_test.shape)


# ============================================================
# ESCALADO SIN FUGA DE INFORMACION
# ============================================================
scaler = StandardScaler()

train_shape = X_train.shape
val_shape = X_val.shape
test_shape = X_test.shape

X_train = scaler.fit_transform(X_train.reshape(-1, N_FEATURES)).reshape(train_shape)
X_val = scaler.transform(X_val.reshape(-1, N_FEATURES)).reshape(val_shape)
X_test = scaler.transform(X_test.reshape(-1, N_FEATURES)).reshape(test_shape)


# ============================================================
# MODELO CNN-RNN MEJORADO
# ============================================================
model = Sequential([
    Conv1D(
        filters=96,
        kernel_size=3,
        activation="relu",
        padding="same",
        input_shape=(TIMESTEPS, N_FEATURES)
    ),
    BatchNormalization(),
    Dropout(0.15),

    Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu",
        padding="same"
    ),
    BatchNormalization(),

    MaxPooling1D(pool_size=2),
    Dropout(0.15),

    SimpleRNN(
        units=110,
        return_sequences=False,
        dropout=0.10,
        recurrent_dropout=0.05
    ),

    Dense(100, activation="relu"),
    Dropout(0.20),

    Dense(50, activation="relu"),
    Dropout(0.10),

    Dense(N_CLASSES, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ============================================================
# ENTRENAMIENTO CON EARLY STOPPING
# ============================================================
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE_EARLY_STOPPING,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=8,
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

model.save(f"modelo_{OUT_PREFIX}.keras")
guardar_curvas_entrenamiento(history, OUT_PREFIX)


# ============================================================
# PREDICCION
# ============================================================
y_prob = model.predict(X_test, verbose=0)
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_prob, axis=1)


# ============================================================
# METRICAS
# ============================================================
acc = accuracy_score(y_true, y_pred)

prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo", average="macro")

print("\n==============================")
print("CNN-RNN mejorado 9 caracteristicas")
print("==============================")
print(f"Accuracy:        {acc:.6f}")
print(f"Precision macro: {prec_macro:.6f}")
print(f"Recall macro:    {rec_macro:.6f}")
print(f"F1 macro:        {f1_macro:.6f}")
print(f"AUC OVR macro:   {auc_ovr:.6f}")
print(f"AUC OVO macro:   {auc_ovo:.6f}")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


# ============================================================
# MATRIZ DE CONFUSION
# ============================================================
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

pd.DataFrame(
    cm,
    columns=["Pred_GR", "Pred_SD", "Pred_GI"],
    index=["True_GR", "True_SD", "True_GI"]
).to_csv(f"confusion_{OUT_PREFIX}.csv")

pd.DataFrame(
    cm_norm,
    columns=["Pred_GR", "Pred_SD", "Pred_GI"],
    index=["True_GR", "True_SD", "True_GI"]
).to_csv(f"confusion_{OUT_PREFIX}_normalizada.csv")

disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=CLASS_NAMES)
disp.plot(cmap=plt.cm.Blues, values_format=".3f")
plt.title("Matriz de confusion normalizada - CNN-RNN mejorado")
plt.tight_layout()
plt.savefig(f"confusion_{OUT_PREFIX}.png", dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# ROC
# ============================================================
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
plt.title("Curvas ROC - CNN-RNN mejorado")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"roc_{OUT_PREFIX}.png", dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# RESUMEN CSV
# ============================================================
row = {
    "modelo": "CNN-RNN_mejorado_9",
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
    "epochs_configuradas": EPOCHS,
    "epochs_ejecutadas": len(history.history.get("loss", [])),
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "early_stopping": True,
    "patience_early_stopping": PATIENCE_EARLY_STOPPING,
    "loss": "categorical_crossentropy",
    "output_activation": "softmax"
}

pd.DataFrame([row]).to_csv(f"metricas_resumen_{OUT_PREFIX}.csv", index=False)


# ============================================================
# CONFIG
# ============================================================
config = {
    "modelo": "CNN-RNN_mejorado_9",
    "entrada": "20x9",
    "datos": {
        "X": X_PATH,
        "Y": Y_PATH
    },
    "split": {
        "test_size": TEST_SIZE,
        "val_size_from_trainval": VAL_SIZE,
        "seed": SEED
    },
    "arquitectura": [
        "Conv1D(96, kernel_size=3, relu, same)",
        "BatchNormalization",
        "Dropout(0.15)",
        "Conv1D(64, kernel_size=3, relu, same)",
        "BatchNormalization",
        "MaxPooling1D(pool_size=2)",
        "Dropout(0.15)",
        "SimpleRNN(110, dropout=0.10, recurrent_dropout=0.05)",
        "Dense(100, relu)",
        "Dropout(0.20)",
        "Dense(50, relu)",
        "Dropout(0.10)",
        "Dense(3, softmax)"
    ],
    "loss": "categorical_crossentropy",
    "optimizer": "Adam",
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "early_stopping": True,
    "patience_early_stopping": PATIENCE_EARLY_STOPPING,
    "monitor_early_stopping": "val_loss",
    "restore_best_weights": True
}

with open(f"config_{OUT_PREFIX}.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4, ensure_ascii=False)


print("\nArchivos generados:")
print(f" - metricas_resumen_{OUT_PREFIX}.csv")
print(f" - confusion_{OUT_PREFIX}.png")
print(f" - confusion_{OUT_PREFIX}.csv")
print(f" - confusion_{OUT_PREFIX}_normalizada.csv")
print(f" - roc_{OUT_PREFIX}.png")
print(f" - accuracy_{OUT_PREFIX}.png")
print(f" - loss_{OUT_PREFIX}.png")
print(f" - training_curves_{OUT_PREFIX}.png")
print(f" - history_{OUT_PREFIX}.csv")
print(f" - modelo_{OUT_PREFIX}.keras")
print(f" - mejor_modelo_{OUT_PREFIX}.keras")
