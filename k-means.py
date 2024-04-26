import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('data/poker-hand-training-true.data')  # Adjust the path as needed
X = data.iloc[:, :-1]  # Assuming the last column is a label if it exists

# Normalize your data if necessary; it depends on your dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose a range of k values to try
k_values = [2, 3, 5, 8, 10]
silhouette_scores = []

# Perform K-Means clustering with different k values and calculate Silhouette Score
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f'Silhouette Score for k={k}: {silhouette_avg}')

# Plotting the silhouette scores
plt.figure(figsize=(8, 6))
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Number of clusters, k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.grid(True)
plt.show()
