# Given a bank customer, build a neural network-based classifier that can determine whether
# they will leave or not in the next 6 months.
# Dataset Description: The case study is from an open-source dataset from Kaggle.
# The dataset contains 10,000 sample points with 14 distinct features such as
# CustomerId, CreditScore, Geography, Gender, Age, Tenure, Balance, etc.
# Link to the Kaggle project:
# https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling
# Perform following steps:
# 1. Read the dataset.
# 2. Distinguish the feature and target set and divide the data set into training and test sets.
# 3. Normalize the train and test data.
# 4. Initialize and build the model. Identify the points of improvement and implement the same.
# 5. Print the accuracy score and confusion matrix (5 points).



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the dataset
df = pd.read_csv(r'C:/Users/mayur/Downloads/Churn_Modelling.csv', encoding='latin1')
print(df.head())

# Drop irrelevant columns
# Dropping RowNumber, CustomerId, Surname (as seen in the image output)
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1) 

# --- Encode Categorical Features ---
# Encoding Gender and Geography using LabelEncoder or pd.get_dummies (image suggests label encoding first)

# 1. Label Encoding (for Gender)
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# 2. One-Hot Encoding (for Geography) - The image shows code snippets to handle 'Geography'
df = pd.get_dummies(df, columns=['Geography'], drop_first=True) 

# Define Features (X) and Target (y)
# Assuming 'Exited' is the target variable (based on the context of churn/credit scoring)
X = df.drop('Exited', axis=1) # Features (all columns except 'Exited')
y = df['Exited']             # Target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Normalize Features ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Build Neural Network Classifier (MLP) ---
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), 
                    solver='adam', 
                    max_iter=300, 
                    activation='relu', 
                    random_state=42)

# Train the model
mlp.fit(X_train, y_train)

# Make predictions
y_pred = mlp.predict(X_test)

# --- Model Evaluation ---

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("\n# Accuracy")
print(f"Accuracy: {acc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n# Confusion Matrix")
print("Confusion Matrix: \n", cm)
# Print the example confusion matrix shown in the image output:
# print("\nConfusion Matrix:\n [[1596  87]\n [ 199 208]]") 

# Optional: Heatmap for Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
