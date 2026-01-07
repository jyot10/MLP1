import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn import datasets
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

sns.set(style="whitegrid", context="notebook")


# -------------------------------
# Load datasets
# -------------------------------
def load_datasets():
    iris = datasets.load_iris()
    X_iris = iris.data
    y_iris = iris.target

    X_blobs, y_blobs = make_blobs(
        n_samples=500, centers=4, cluster_std=0.60, random_state=42
    )

    X_moons, y_moons = make_moons(
        n_samples=500, noise=0.08, random_state=42
    )

    return (
        (X_iris, y_iris, "Iris (4D)"),
        (X_blobs, y_blobs, "Blobs (2D)"),
        (X_moons, y_moons, "Moons (2D)")
    )


# -------------------------------
# Standardization
# -------------------------------
def standardize(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


# -------------------------------
# KMeans
# -------------------------------
def run_kmeans(X, k, random_state=42):
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    return labels, centers, kmeans


# -------------------------------
# DBSCAN
# -------------------------------
def run_dbscan(X, eps=0.5, min_samples=5):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    return labels, db


# -------------------------------
# Evaluation
# -------------------------------
def evaluate_clusters(X, labels, name=""):
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1])  # exclude noise

    if n_clusters <= 1:
        sil = -1
        dbi = np.nan
    else:
        sil = silhouette_score(X, labels)
        dbi = davies_bouldin_score(X, labels)

    print(f"{name} -> clusters: {n_clusters}, silhouette: {sil:.4f}, davies-bouldin: {dbi:.4f}")
    return n_clusters, sil, dbi


# -------------------------------
# PCA Plotting
# -------------------------------
def plot_2d(X2, labels, title="Clusters", centers2=None, show_centers=True):
    df = pd.DataFrame(X2, columns=["PC1", "PC2"])
    df["cluster"] = labels.astype(str)

    plt.figure(figsize=(7, 5))

    unique_labels = np.unique(labels)
    palette = sns.color_palette("husl", len(unique_labels))

    if -1 in labels:
        palette_map = {
            str(lab): ("#000000" if lab == -1 else palette[i])
            for i, lab in enumerate(unique_labels)
        }
        sns.scatterplot(
            x="PC1", y="PC2", hue="cluster",
            data=df, palette=palette_map, s=40, legend="full"
        )
    else:
        sns.scatterplot(
            x="PC1", y="PC2", hue="cluster",
            data=df, palette="husl", s=40, legend="full"
        )

    if centers2 is not None and show_centers:
        plt.scatter(
            centers2[:, 0], centers2[:, 1],
            c="red", s=200, marker="X", edgecolor="k", label="Centroids"
        )

    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.tight_layout()
    plt.show()


# -------------------------------
# PCA Transform
# -------------------------------
def pca_transform(X, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X), pca


# -------------------------------
# Main Function
# -------------------------------
def main():
    datasets_list = load_datasets()

    for X, y_true, name in datasets_list:
        print(f"\n=== Dataset: {name}, shape: {X.shape} ===")

        Xs, _ = standardize(X)

        # ---- KMeans ----
        if name.startswith("Iris"):
            k_guess = 3
        elif "Blobs" in name:
            k_guess = 4
        else:
            k_guess = 2

        klabels, centers, _ = run_kmeans(Xs, k_guess)
        evaluate_clusters(Xs, klabels, name + " KMeans")

        X2, pca = pca_transform(Xs)
        centers2 = pca.transform(centers)
        plot_2d(X2, klabels, f"{name} - KMeans (k={k_guess})", centers2)

        # ---- DBSCAN ----
        if name.startswith("Iris"):
            eps, min_samples = 0.9, 5
        elif "Blobs" in name:
            eps, min_samples = 0.5, 5
        else:
            eps, min_samples = 0.2, 5

        dlabels, _ = run_dbscan(Xs, eps, min_samples)
        evaluate_clusters(Xs, dlabels, name + " DBSCAN")

        X2, _ = pca_transform(Xs)
        plot_2d(
            X2, dlabels,
            f"{name} - DBSCAN (eps={eps}, min_samples={min_samples})",
            show_centers=False
        )


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    main()
