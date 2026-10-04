"""
AIML Laboratory - Semester 5
Assignment 3: Dimensionality Reduction Using Principal Component Analysis (PCA)

Aim:
    To reduce the dimensionality of the Iris dataset using Principal Component
    Analysis (PCA) while retaining maximum information (variance).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def main():
    print("=" * 70)
    print("ASSIGNMENT 3: Principal Component Analysis (PCA)")
    print("=" * 70)

    # 1. Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    print(f"Original Dataset Shape: {X.shape} (150 samples, 4 features)")
    print(f"Features: {feature_names}")

    # 2. Standardize features (mean=0, variance=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Apply PCA with 2 components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    var_ratio = pca.explained_variance_ratio_
    total_var = np.sum(var_ratio) * 100

    print(f"\nTransformed Dataset Shape: {X_pca.shape} (150 samples, 2 components)")
    print(f"Explained Variance Ratio per Component:")
    print(f"  - Principal Component 1: {var_ratio[0]:.4f} ({var_ratio[0]*100:.2f}%)")
    print(f"  - Principal Component 2: {var_ratio[1]:.4f} ({var_ratio[1]*100:.2f}%)")
    print(f"Total Variance Explained by PC1 & PC2: {total_var:.2f}%")

    # 4. Visualize 2D PCA projection
    plt.figure(figsize=(7.5, 5), dpi=300)
    colors = ["#e63946", "#2a9d8f", "#457b9d"]

    for idx, target_name in enumerate(target_names):
        plt.scatter(
            X_pca[y == idx, 0],
            X_pca[y == idx, 1],
            color=colors[idx],
            label=target_name.capitalize(),
            alpha=0.85,
            s=55,
            edgecolors="black",
            linewidth=0.5
        )

    plt.title("PCA: 2D Projection of Iris Dataset", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel(f"Principal Component 1 ({var_ratio[0]*100:.2f}% Variance)", fontsize=11)
    plt.ylabel(f"Principal Component 2 ({var_ratio[1]*100:.2f}% Variance)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, title="Iris Class")
    plt.tight_layout()

    # Save figure and display
    plt.savefig("assignment_03_output.png", dpi=300)
    print("\nPlot saved as 'assignment_03_output.png'. Displaying plot...")
    plt.show()


if __name__ == "__main__":
    main()
