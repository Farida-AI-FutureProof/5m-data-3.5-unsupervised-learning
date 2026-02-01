import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def main():
    # Load and scale data
    iris = load_iris()
    X = iris.data
    X_scaled = StandardScaler().fit_transform(X)

    # 1) KMeans – Elbow Method
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

    # Optional: silhouette score for k >= 2
    print("\nKMeans Silhouette Scores:")
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        print(f"k={k}, silhouette={sil:.4f}")

    # 2) DBSCAN experiments
    print("\nDBSCAN Experiments:")
    candidates = [
        (0.3, 5),
        (0.4, 5),
        (0.5, 5),
        (0.6, 5),
        (0.5, 3),
        (0.6, 3),
    ]

    for eps, ms in candidates:
        db = DBSCAN(eps=eps, min_samples=ms)
        labels = db.fit_predict(X_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))

        print(f"eps={eps}, min_samples={ms} -> clusters={n_clusters}, noise={n_noise}")

        if n_clusters > 1:
            sil = silhouette_score(X_scaled, labels)
            print(f"  silhouette={sil:.4f}")


if __name__ == "__main__":
    main()


# Guidance/formatting support: ChatGPT (OpenAI GPT-5)
