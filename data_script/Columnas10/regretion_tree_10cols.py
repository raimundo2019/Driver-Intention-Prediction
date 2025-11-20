# ===========================================
# Regr. Logística y Árbol de Decisión (10 cols)
# ===========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# ------------------------------
# Cargar datasets (10 columnas)
# ------------------------------
X = np.loadtxt("expanded_data.csv", delimiter=",")
Y = np.loadtxt("out_data.csv", delimiter=",")

# Ajustar dimensiones
n_samples = Y.shape[0]
n_features = 10 * 20   # 20 pasos temporales * 10 features
n_classes = 3

X = np.reshape(X, (n_samples, n_features))
Y = np.reshape(Y, (n_samples, n_classes))

# Normalizar
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# División en entrenamiento / prueba
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=30)
y_true = np.argmax(y_test, axis=1)
y_train_true = np.argmax(y_train, axis=1)

# ------------------------------
# Modelo 1: Regresión Logística (OVR)
# ------------------------------
log_model = LogisticRegression(max_iter=200, multi_class='ovr')
log_model.fit(X_train, y_train_true)
y_pred_log = log_model.predict(X_test)

# ------------------------------
# Modelo 2: Árbol de Decisión
# ------------------------------
tree_model = DecisionTreeClassifier(random_state=30)
tree_model.fit(X_train, y_train_true)
y_pred_tree = tree_model.predict(X_test)

# ------------------------------
# Métricas
# ------------------------------
def evaluar_modelo(nombre, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    report = classification_report(y_true, y_pred, target_names=['GR', 'SD', 'GI'])
    print(f"\n==== {nombre} ====")
    print(report)
    print(f"Accuracy: {acc:.4f} | F1-macro: {f1:.4f}")

evaluar_modelo("Regresión Logística (OVR)", y_true, y_pred_log)
evaluar_modelo("Árbol de Decisión", y_true, y_pred_tree)

# ------------------------------
# Matrices de confusión y CSVs
# ------------------------------
def guardar_matriz(y_true, y_pred, nombre_base, titulo):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=['GR', 'SD', 'GI'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(titulo)
    plt.tight_layout()
    plt.savefig(f"{nombre_base}.png", dpi=300, bbox_inches='tight')
    plt.close()

    pd.DataFrame(cm,
                 columns=['Pred_GR','Pred_SD','Pred_GI'],
                 index=['True_GR','True_SD','True_GI']
                ).to_csv(f"{nombre_base}.csv", index=True)
    pd.DataFrame(cm_norm,
                 columns=['Pred_GR','Pred_SD','Pred_GI'],
                 index=['True_GR','True_SD','True_GI']
                ).to_csv(f"{nombre_base}_normalized.csv", index=True)

guardar_matriz(y_true, y_pred_log,
               "confusion_logistic_10cols",
               "Matriz de confusión - Regresión Logística (10 columnas)")
guardar_matriz(y_true, y_pred_tree,
               "confusion_tree_10cols",
               "Matriz de confusión - Árbol de Decisión (10 columnas)")

print("\nMatrices guardadas en el directorio actual.")
