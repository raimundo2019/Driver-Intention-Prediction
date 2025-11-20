# =====================================================
# Entrenamiento + ROC + Matriz de Confusión (9 columnas)
# =====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score, f1_score, classification_report
)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------------
# Cargar datos
# ------------------------------
X = np.loadtxt("expanded_data.csv", delimiter=",")
Y = np.loadtxt("out_data.csv", delimiter=",")

n_samples = Y.shape[0]
n_features = 9 * 20
n_classes = 3

X = np.reshape(X, (n_samples, 20, 9))
Y = np.reshape(Y, (n_samples, n_classes))

# ------------------------------
# Crear y entrenar modelo CNN+RNN
# ------------------------------
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(20, 9)),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dense(64, activation='relu'),
    Dense(n_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
model.fit(X, Y, epochs=100, batch_size=32, verbose=1, callbacks=[es])

# Guardar modelo entrenado
model.save("cnn_rnn_9cols_model.h5")
print("Modelo CNN/RNN (9 columnas) guardado como cnn_rnn_9cols_model.h5")

# ------------------------------
# Predicciones y métricas
# ------------------------------
y_pred_probs = model.predict(X)
y_true = np.argmax(Y, axis=1)
y_pred = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='macro')
roc_auc = roc_auc_score(Y, y_pred_probs, multi_class='ovr')

print(f"\nAccuracy: {acc:.4f}")
print(f"F1-macro: {f1:.4f}")
print(f"AUC (macro OVR): {roc_auc:.4f}")
print(classification_report(y_true, y_pred, target_names=['GR', 'SD', 'GI']))

# ------------------------------
# Curvas ROC y Matriz de Confusión
# ------------------------------
fpr, tpr, roc_auc_vals = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y[:, i], y_pred_probs[:, i])
    roc_auc_vals[i] = auc(fpr[i], tpr[i])

plt.figure()
colors = ['darkorange', 'blue', 'green']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color,
             label=f'Clase {i} (área = {roc_auc_vals[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Curvas ROC - CNN/RNN (9 columnas)')
plt.legend(loc='lower right')
plt.savefig("roc_cnn_rnn_9cols.png", dpi=300, bbox_inches='tight')
plt.close()

cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=['GR', 'SD', 'GI']).plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión - CNN/RNN (9 columnas)")
plt.tight_layout()
plt.savefig("confusion_cnn_rnn_9cols.png", dpi=300, bbox_inches='tight')
plt.close()

pd.DataFrame(cm, columns=['Pred_GR','Pred_SD','Pred_GI'],
             index=['True_GR','True_SD','True_GI']).to_csv("confusion_cnn_rnn_9cols.csv", index=True)
pd.DataFrame(cm_norm, columns=['Pred_GR','Pred_SD','Pred_GI'],
             index=['True_GR','True_SD','True_GI']).to_csv("confusion_cnn_rnn_9cols_normalized.csv", index=True)

print("\nMatrices y curva ROC guardadas para modelo de 9 columnas.")
