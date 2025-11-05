# Predict the price of the Uber ride from a given pickup point to the agreed drop-off location.
# Perform following tasks:
# 1. Pre-process the dataset.
# 2. Identify outliers.
# 3. Check the correlation.
# 4. Implement linear regression and random forest regression models.
# 5. Evaluate the models and compare their respective scores like R2, RMSE, etc.



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset
df = pd.read_csv("C:/Users/mayur/Downloads/uber1/uber.csv")
print(df.head())

# Drop rows with missing values
df = df.dropna()

# Remove invalid fare values
# Limit passenger count to be between 1 and 6
df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 100)]
df = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 6)]

# Calculate distance using coordinates (Haversine formula)
def haversine_distance(lat1, lon1, lat2, lon2):
    r = 6371 # Radius of Earth in kilometers
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi/2)*2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)*2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return r * c

df['distance_km'] = haversine_distance(df['pickup_latitude'], df['pickup_longitude'], df['dropoff_latitude'], df['dropoff_longitude'])

# Remove zero-distance rides
df = df[df['distance_km'] > 0]

# Remove outliers for fare and distance
df = df[(df['fare_amount'] < 100) & (df['distance_km'] < 50)]

# Plot Fare Amount Distribution (Boxplot)
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['fare_amount'])
plt.title('Fare Amount Distribution')
plt.show()

# Plot Correlation Matrix (Heatmap)
plt.figure(figsize=(6, 4))
sns.heatmap(df[['fare_amount', 'distance_km', 'passenger_count']].corr(), 
            cmap='coolwarm', 
            annot=True, 
            fmt=".3f")
plt.title('Correlation Matrix')
plt.show()

# Prepare data for modeling
X = df[['distance_km', 'passenger_count']]
y = df['fare_amount']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest Regressor Model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Function to evaluate model performance
def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

# Evaluate the models
evaluate_model(y_test, y_pred_lr, "Linear Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest Regressor")
