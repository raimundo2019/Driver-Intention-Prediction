# =====================================================
# Entrenamiento y evaluación LSTM (9 columnas)
# =====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------------
# Cargar y preparar datos
# ------------------------------
X = np.loadtxt("expanded_data.csv", delimiter=",")
Y = np.loadtxt("out_data.csv", delimiter=",")

n_samples = Y.shape[0]
timesteps = 20
n_features = 9
n_classes = 3

X = X.reshape(n_samples, timesteps, n_features)
Y = Y.reshape(n_samples, n_classes)

# Escalado por feature
scaler = MinMaxScaler()
X = scaler.fit_transform(X.reshape(-1, n_features)).reshape(X.shape)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.5, random_state=30, stratify=np.argmax(Y, axis=1)
)
y_test_labels = np.argmax(y_test, axis=1)

# ------------------------------
# Modelo LSTM
# ------------------------------
model = Sequential([
    LSTM(128, return_sequences=False, input_shape=(timesteps, n_features)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(n_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

# ------------------------------
# Entrenamiento
# ------------------------------
history = model.fit(
    X_train, y_train,
    epochs=60,
    batch_size=32,
    validation_split=0.2,
    callbacks=[es],
    verbose=1
)

model.save("lstm_9cols_model.h5")

# ------------------------------
# Evaluación
# ------------------------------
y_pred_probs = model.predict(X_test)
y_pred_labels = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_test_labels, y_pred_labels)
f1 = f1_score(y_test_labels, y_pred_labels, average="macro")
auc_macro_ovr = roc_auc_score(y_test, y_pred_probs, multi_class="ovr")

print(f"\nAccuracy: {acc:.4f}")
print(f"F1-macro: {f1:.4f}")
print(f"AUC (macro OVR): {auc_macro_ovr:.4f}")
print(classification_report(y_test_labels, y_pred_labels, target_names=["GR", "SD", "GI"]))

# ------------------------------
# Curvas de entrenamiento (opcional útil)
# ------------------------------
plt.figure()
plt.plot(history.history["accuracy"]); plt.plot(history.history["val_accuracy"])
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Curva de entrenamiento LSTM (9 cols)")
plt.legend(["train","val"]); plt.tight_layout()
plt.savefig("training_curves_lstm_9cols.png", dpi=300, bbox_inches="tight"); plt.close()

# ------------------------------
# Matriz de Confusión y CSVs
# ------------------------------
cm = confusion_matrix(y_test_labels, y_pred_labels)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=["GR","SD","GI"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión - LSTM (9 columnas)")
plt.tight_layout()
plt.savefig("confusion_lstm_9cols.png", dpi=300, bbox_inches="tight")
plt.close()

pd.DataFrame(cm, columns=["Pred_GR","Pred_SD","Pred_GI"],
             index=["True_GR","True_SD","True_GI"]).to_csv("confusion_lstm_9cols.csv", index=True)
pd.DataFrame(cm_norm, columns=["Pred_GR","Pred_SD","Pred_GI"],
             index=["True_GR","True_SD","True_GI"]).to_csv("confusion_lstm_9cols_normalized.csv", index=True)

print("\nSalidas LSTM (9 cols):")
print(" - lstm_9cols_model.h5")
print(" - training_curves_lstm_9cols.png")
print(" - confusion_lstm_9cols.png")
print(" - confusion_lstm_9cols.csv")
print(" - confusion_lstm_9cols_normalized.csv")
