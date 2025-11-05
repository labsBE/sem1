# Implement K-Means clustering/ hierarchical clustering on sales_data_sample.csv dataset.
# Determine the number of clusters using the elbow method.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch

# Load the dataset
df = pd.read_csv('C:/Users/mayur/Downloads/sales_data_sample.csv', encoding='latin1')
print(df.head())

# Data Preprocessing
df = df.dropna()

# Select features for clustering (Quantity Ordered and Sales)
df = df[['QUANTITYORDERED', 'SALES']]

# Scale the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

# --- K-Means Clustering ---

# 1. Elbow Method for optimal K
wcss = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(K, wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# 2. Fit K-Means with optimal K (assuming optimal K = 3 from the plot)
# The image shows the Elbow plot suggesting an elbow at K=3 or K=4, 
# but the code uses a value. Let's assume n_clusters=3 based on typical values/the plot appearance.
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster_KMeans'] = kmeans.fit_predict(data_scaled)

# 3. Plot K-Means Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data_scaled[:, 0], y=data_scaled[:, 1], 
                hue=df['Cluster_KMeans'], 
                palette='viridis')
plt.title('K-Means Clustering')
plt.xlabel('Quantity Ordered (Scaled)')
plt.ylabel('Sales (Scaled)')
plt.show()

# --- Hierarchical Clustering (Agglomerative) ---

# 1. Fit Agglomerative Clustering
hc = AgglomerativeClustering(n_clusters=3) # Assuming n_clusters=3 for comparison
df['Cluster_Hierarchical'] = hc.fit_predict(data_scaled)

# 2. Plot Hierarchical Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data_scaled[:, 0], y=data_scaled[:, 1], 
                hue=df['Cluster_Hierarchical'], 
                palette='Set2')
plt.title('Hierarchical Clustering')
plt.xlabel('Quantity Ordered (Scaled)')
plt.ylabel('Sales (Scaled)')
plt.show()
