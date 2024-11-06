import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score 
import matplotlib.pyplot as plt
import pickle

current_dir = os.path.join(os.path.dirname('web'))
directory = os.path.join(current_dir, 'zeroShot_vkiit')
key_file_path = os.path.join(current_dir, 'key.txt')
file_list = [f'T{str(i).zfill(4)}.txt' for i in range(1, 1159)]

def load_file(file_path):
    try:
        with open(file_path, 'r') as f:
            data = np.array([float(line.strip()) for line in f])
        return data
    except:
        return None

features = []
file_names = []
for file_name in file_list:
    file_path = os.path.join(directory, file_name)
    data = load_file(file_path)
    if data is not None and len(data) > 0:
        mean = np.mean(data)
        median = np.median(data)
        std = np.std(data)
        data_min = np.min(data)
        data_max = np.max(data)
        features.append([mean, median, std, data_min, data_max])
        file_names.append(file_name)
    else:
        features.append([np.nan, np.nan, np.nan, np.nan, np.nan])
        file_names.append(file_name)

features = np.array(features)

valid_indices = ~np.isnan(features).any(axis=1)
valid_features = features[valid_indices]
valid_file_names = np.array(file_names)[valid_indices]

if valid_features.shape[0] == 0:
    raise ValueError("No valid samples found for scaling and clustering.")

scaler = StandardScaler()
scaled_features = scaler.fit_transform(valid_features)

def load_true_labels(file_path):
    with open(file_path, 'r') as file:
        labels = [float(line.strip()) for line in file.readlines()]
    return labels

true_labels = load_true_labels(key_file_path)

def elbow_method(data, max_k=10):
    distortions = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), distortions, marker='o')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.savefig('elbow_method.png')
    plt.show()

elbow_method(scaled_features, max_k=10)

optimal_clusters = int(input("Enter the number of optimal clusters: "))
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans.fit(scaled_features)
labels = kmeans.labels_

kmeans_model_path = os.path.join(current_dir, f'kmeans_model{optimal_clusters}.pkl')
with open(kmeans_model_path, 'wb') as f:
    pickle.dump(kmeans, f)
print(f"KMeans model saved to {kmeans_model_path}")

silhouette_avg = silhouette_score(scaled_features, labels)
inertia = kmeans.inertia_
davies_bouldin = davies_bouldin_score(scaled_features, labels)
ari = adjusted_rand_score(true_labels, labels)
nmi = normalized_mutual_info_score(true_labels, labels)

print("Silhouette Score:", silhouette_avg)
print("Inertia ", inertia)
print("Davies-Bouldin Index", davies_bouldin)
print("Adjusted Rand Index", ari)
print("Normalized Mutual Information Score", nmi)