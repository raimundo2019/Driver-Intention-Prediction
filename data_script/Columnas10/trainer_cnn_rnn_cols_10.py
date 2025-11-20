# =====================================================
# Entrenamiento y evaluación CNN-RNN (10 columnas)
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
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical

# ------------------------------
# Cargar y preparar datos
# ------------------------------
X = np.loadtxt("expanded_data.csv", delimiter=",")
Y = np.loadtxt("out_data.csv", delimiter=",")

n_samples = Y.shape[0]
n_features = 10
timesteps = 20
n_classes = 3

X = np.reshape(X, (n_samples, timesteps, n_features))
Y = np.reshape(Y, (n_samples, n_classes))

# Escalado
scaler = MinMaxScaler()
X = scaler.fit_transform(X.reshape(-1, n_features)).reshape(X.shape)

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=30)
y_train_labels = np.argmax(y_train, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# ------------------------------
# Definición del modelo CNN-RNN
# ------------------------------
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(timesteps, n_features)),
    MaxPooling1D(pool_size=2),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(n_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ------------------------------
# Entrenamiento
# ------------------------------
history = model.fit(X_train, y_train, epochs=30, batch_size=16,
                    validation_split=0.2, verbose=1)

model.save("cnn_rnn_10cols_model.h5")

# ------------------------------
# Evaluación
# ------------------------------
y_pred_probs = model.predict(X_test)
y_pred_labels = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_test_labels, y_pred_labels)
f1 = f1_score(y_test_labels, y_pred_labels, average='macro')
roc_auc = roc_auc_score(y_test, y_pred_probs, multi_class='ovr')

print(f"\nAccuracy: {acc:.4f}")
print(f"F1-macro: {f1:.4f}")
print(f"AUC (macro OVR): {roc_auc:.4f}")
print(classification_report(y_test_labels, y_pred_labels, target_names=['GR', 'SD', 'GI']))

# ------------------------------
# Matriz de Confusión y CSVs
# ------------------------------
cm = confusion_matrix(y_test_labels, y_pred_labels)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm,
                              display_labels=['GR', 'SD', 'GI'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión - CNN/RNN (10 columnas)")
plt.tight_layout()
plt.savefig("confusion_cnn_rnn_10cols.png", dpi=300, bbox_inches='tight')
plt.close()

pd.DataFrame(cm, columns=['Pred_GR','Pred_SD','Pred_GI'],
             index=['True_GR','True_SD','True_GI']).to_csv("confusion_cnn_rnn_10cols.csv", index=True)

pd.DataFrame(cm_norm, columns=['Pred_GR','Pred_SD','Pred_GI'],
             index=['True_GR','True_SD','True_GI']).to_csv("confusion_cnn_rnn_10cols_normalized.csv", index=True)

print("\nMatriz de confusión y CSVs guardados para CNN/RNN (10 columnas).")
