# Assignment

## Instructions

1) K-Means on Iris
- Determine optimal clusters with elbow method

2) DBSCAN on Iris
- Experiment eps and min_samples

## Answer (runnable)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

iris = load_iris()
X = iris.data
X_scaled = StandardScaler().fit_transform(X)

# 1) KMeans Elbow
inertias = []
ks = range(1, 11)
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(7, 4))
plt.plot(list(ks), inertias, marker="o")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("Elbow Method (KMeans on Iris)")
plt.grid(alpha=0.3)
plt.show()

# Optional: silhouette for k>=2
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X_scaled)
    sil = silhouette_score(X_scaled, km.labels_)
    print(f"k={k}, silhouette={sil:.4f}")

# 2) DBSCAN experiments
candidates = [(0.3, 5), (0.4, 5), (0.5, 5), (0.6, 5), (0.5, 3), (0.6, 3)]
for eps, ms in candidates:
    db = DBSCAN(eps=eps, min_samples=ms).fit(X_scaled)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    print(f"DBSCAN eps={eps}, min_samples={ms} -> clusters={n_clusters}, noise={n_noise}")

    # silhouette only valid if >1 cluster
    if n_clusters > 1:
        sil = silhouette_score(X_scaled, labels)
        print("  silhouette:", f"{sil:.4f}")




# Guidance/formatting support: ChatGPT (OpenAI GPT-5).
