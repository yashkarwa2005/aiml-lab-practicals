"""
AIML Laboratory - Semester 5
Assignment 2: Apply Clustering Techniques Using Unsupervised Learning (K-Means)

Aim:
    To apply the K-Means clustering algorithm on unlabeled data and determine
    the optimal number of clusters using the Elbow method.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def main():
    print("=" * 70)
    print("ASSIGNMENT 2: K-Means Clustering & Elbow Method")
    print("=" * 70)

    # 1. Generate synthetic 2D clustering dataset
    X, y_true = make_blobs(
        n_samples=200,
        centers=3,
        cluster_std=1.0,
        random_state=42
    )

    print(f"Dataset generated: {X.shape[0]} samples with {X.shape[1]} features.")

    # 2. Fit K-Means with k = 3
    k_optimal = 3
    model = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    centers = model.cluster_centers_

    print(f"\nK-Means fitted with k = {k_optimal}")
    print("Cluster Centers (Centroids):")
    print(np.array2string(centers, precision=3, suppress_small=True))
    print(f"Model Inertia (WCSS): {model.inertia_:.2f}")

    # 3. Calculate Inertia for Elbow Method across k = 1 to 9
    print("\nCalculating Elbow Method inertia values...")
    k_range = range(1, 10)
    inertia_values = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertia_values.append(km.inertia_)
        print(f"  k = {k}: Inertia = {km.inertia_:.2f}")

    # 4. Plot 2-panel visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Panel 1: Clusters & Centroids
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    for c_idx in range(k_optimal):
        ax1.scatter(
            X[labels == c_idx, 0],
            X[labels == c_idx, 1],
            color=colors[c_idx],
            alpha=0.7,
            s=45,
            label=f"Cluster {c_idx + 1}"
        )

    ax1.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="X",
        s=220,
        c="black",
        edgecolors="white",
        linewidth=2,
        label="Centroids",
        zorder=6
    )
    ax1.set_title("K-Means Clustering (k = 3)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True)

    # Panel 2: Elbow Method Curve
    ax2.plot(k_range, inertia_values, marker="o", color="#023e8a", linewidth=2, markersize=6)
    ax2.axvline(x=k_optimal, color="#e63946", linestyle="--", linewidth=1.5, label=f"Elbow Point (k = {k_optimal})")
    ax2.set_title("Elbow Method for Optimal k", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    ax2.set_xticks(k_range)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True)

    plt.tight_layout()
    plt.savefig("assignment_02_output.png", dpi=300)
    print("\nPlot saved as 'assignment_02_output.png'. Displaying plot...")
    plt.show()


if __name__ == "__main__":
    main()
