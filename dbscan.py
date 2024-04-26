import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
data_path = 'data/poker-hand-training-true.data'
data = pd.read_csv(data_path, header=None)

# It's a good practice to scale the data for clustering algorithms
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data.iloc[:, :-1])  # Assuming the last column is the label and should be excluded

# DBSCAN clustering experiments with different hyperparameters
def run_dbscan_experiments(X, eps_values, min_samples_values):
    results = []
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(X)
            
            # Silhouette score is only valid if there are more than one cluster identified
            if len(set(labels)) - (1 if -1 in labels else 0) > 1:
                silhouette_avg = silhouette_score(X, labels)
            else:
                silhouette_avg = "N/A - One cluster or noise"
            
            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'silhouette_score': silhouette_avg
            })
    
    return pd.DataFrame(results)

# Define the range of hyperparameters for the experiments
eps_values = [0.3, 0.5, 0.7, 1.0, 1.5]
min_samples_values = [5, 10, 15, 20]

# Run the experiments
experiment_results = run_dbscan_experiments(X_scaled, eps_values, min_samples_values)

# Save the results to a CSV file
experiment_results.to_csv('dbscan_clustering_results.csv', index=False)
