# Classify the email using the binary classification method. Email Spam detection has two
# states: a) Normal State – Not Spam, b) Abnormal State – Spam. Use K-Nearest Neighbors and
# Support Vector Machine for classification. Analyze their performance.


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, ConfusionMatrixDisplay
)

# --- Load Dataset ---
data = pd.read_csv("C:/Users/mayur/Downloads/emails/emails.csv")

print("\n--- Dataset Info ---")
print(data.info())
print(data.head())

# --- Preprocessing ---

# 1️⃣ Drop any identifier or text columns like 'Email No' or 'Email 3165'
for col in data.columns:
    if data[col].dtype == 'object':
        print(f"Dropping non-numeric column: {col}")
        data.drop(columns=[col], inplace=True)

# 2️⃣ Check for missing values
data.dropna(inplace=True)

# --- Define Features (X) and Target (y) ---
X = data.drop('Prediction', axis=1)
y = data['Prediction']

# 3️⃣ Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# --- K-Nearest Neighbors ---
print("\n--- KNN Model ---")
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

print(classification_report(y_test, y_pred_knn))
cm_knn = confusion_matrix(y_test, y_pred_knn)
print("Confusion Matrix (KNN):\n", cm_knn)

plt.figure(figsize=(5, 5))
ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=[0, 1]).plot(cmap='Blues')
plt.title("KNN Confusion Matrix")
plt.show()

# --- Support Vector Machine ---
print("\n--- SVM Model ---")
svc = SVC(kernel='linear', gamma='auto', random_state=42)
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)

print(classification_report(y_test, y_pred_svc))
cm_svc = confusion_matrix(y_test, y_pred_svc)
print("Confusion Matrix (SVM):\n", cm_svc)

plt.figure(figsize=(5, 5))
ConfusionMatrixDisplay(confusion_matrix=cm_svc, display_labels=[0, 1]).plot(cmap='Greens')
plt.title("SVM Confusion Matrix")
plt.show()

# --- Compare Accuracy ---
print("\nModel Comparison:")
print(f"KNN Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svc):.4f}")
