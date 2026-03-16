import os
import json
import shutil
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
VECTORS_NPZ  = os.path.join(BASE_DIR, 'data', 'vectors', 'image_vectors.npz')
IMAGES_DIR   = os.path.join(BASE_DIR, 'data', 'auction_images_primary')
CLUSTERS_DIR = os.path.join(BASE_DIR, 'data', 'clusters')
N_CLUSTERS = 15    # number of clusters — adjust to taste

# ============================================================
# LOAD VECTORS
# ============================================================

print("Loading vectors...")
data   = np.load(VECTORS_NPZ)
names  = list(data.files)
matrix = np.stack([data[n] for n in names])   # (N, D)
N, D   = matrix.shape
print(f"  {N} images, {D}-dim embeddings")

# L2-normalise so cosine similarity == dot product
matrix_norm = normalize(matrix, norm='l2')

# ============================================================
# K-MEANS CLUSTERING
# ============================================================

print(f"Clustering into {N_CLUSTERS} groups (K-means)...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
labels = kmeans.fit_predict(matrix_norm)

# Group names by cluster label, sort cluster by distance to centroid
clusters: dict[str, list[str]] = {}
for cluster_id in range(N_CLUSTERS):
    member_idx = np.where(labels == cluster_id)[0]
    centroid   = kmeans.cluster_centers_[cluster_id]
    # Sort members closest-to-centroid first
    dists      = np.linalg.norm(matrix_norm[member_idx] - centroid, axis=1)
    sorted_idx = member_idx[np.argsort(dists)]
    key        = f"cluster_{cluster_id + 1:02d}"
    clusters[key] = [names[i] for i in sorted_idx]

# Report
sizes = sorted([len(v) for v in clusters.values()], reverse=True)
print(f"  Cluster sizes: {sizes}")

# ============================================================
# SAVE RESULTS
# ============================================================

json_path = os.path.join(BASE_DIR, 'data', 'vectors', 'clusters.json')
with open(json_path, 'w') as f:
    json.dump(clusters, f, indent=2)
print(f"Cluster assignments → {json_path}")

# Copy images into per-cluster folders
if os.path.exists(CLUSTERS_DIR):
    shutil.rmtree(CLUSTERS_DIR)
os.makedirs(CLUSTERS_DIR)

for cluster_name, imgs in clusters.items():
    cluster_dir = os.path.join(CLUSTERS_DIR, cluster_name)
    os.makedirs(cluster_dir, exist_ok=True)
    for img_name in imgs:
        src = os.path.join(IMAGES_DIR, img_name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(cluster_dir, img_name))

print(f"Images organised into {CLUSTERS_DIR}")
print("Done.")
