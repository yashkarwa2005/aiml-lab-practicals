"""
AIML Laboratory - Semester 5
Assignment 5: K-Nearest Neighbors (KNN) Algorithm for Classification

Aim:
    To implement the K-Nearest Neighbors (KNN) classification algorithm,
    evaluate classification accuracy, and analyze the effect of varying K.
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    print("=" * 70)
    print("ASSIGNMENT 5: K-Nearest Neighbors (KNN) Classification")
    print("=" * 70)

    # 1. Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    # 2. Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Feature Scaling (critical for distance-based algorithms like KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train KNN model with chosen K = 5
    k_chosen = 5
    model = KNeighborsClassifier(n_neighbors=k_chosen)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, predictions)

    print(f"KNN Classifier trained with K = {k_chosen}")
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=target_names))

    # 5. Evaluate Accuracy across different values of K (1 to 15)
    print("Evaluating accuracy across K = 1 to 15:")
    k_range = range(1, 16)
    k_accuracies = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        score = knn.score(X_test_scaled, y_test)
        k_accuracies.append(score)
        print(f"  K = {k:02d} -> Test Accuracy: {score * 100:.2f}%")

    # 6. Plot Accuracy vs K
    plt.figure(figsize=(7, 4.8), dpi=300)
    plt.plot(k_range, k_accuracies, marker="s", color="#2a9d8f", linewidth=2, markersize=7, label="Test Accuracy")
    plt.axvline(x=k_chosen, color="#e76f51", linestyle="--", linewidth=1.5, label=f"Selected K = {k_chosen}")
    plt.title("KNN Classification: Accuracy vs. Value of K", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Number of Neighbors (K)", fontsize=11)
    plt.ylabel("Testing Accuracy", fontsize=11)
    plt.xticks(k_range)
    plt.ylim([0.85, 1.03])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()

    # Save figure and display
    plt.savefig("assignment_05_output.png", dpi=300)
    print("\nPlot saved as 'assignment_05_output.png'. Displaying plot...")
    plt.show()


if __name__ == "__main__":
    main()
