"""
AIML Laboratory - Semester 5
Assignment 4: Naïve Bayes Classification Algorithm

Aim:
    To implement the Gaussian Naïve Bayes classification algorithm on the
    Iris dataset and evaluate its classification performance.
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    print("=" * 70)
    print("ASSIGNMENT 4: Gaussian Naïve Bayes Classification")
    print("=" * 70)

    # 1. Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    # 2. Split dataset into 80% train and 20% test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # 3. Initialize and train Gaussian Naïve Bayes model
    model = GaussianNB()
    model.fit(X_train, y_train)

    # 4. Predictions and metrics
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=target_names))

    print("Confusion Matrix:")
    print(cm)

    # 5. Visualize Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    cax = ax.matshow(cm, cmap="Blues", alpha=0.8)

    # Add numeric labels to cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                x=j, y=i, s=cm[i, j],
                va="center", ha="center",
                size="xx-large", weight="bold", color="black"
            )

    fig.colorbar(cax)
    ax.set_xticks(range(len(target_names)))
    ax.set_yticks(range(len(target_names)))
    ax.set_xticklabels(target_names, fontsize=10)
    ax.set_yticklabels(target_names, fontsize=10)
    ax.set_title("Gaussian Naïve Bayes Confusion Matrix", fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", labelpad=8, fontsize=11)
    ax.set_ylabel("True Label", labelpad=8, fontsize=11)
    plt.tight_layout()

    # Save figure and display
    plt.savefig("assignment_04_output.png", dpi=300)
    print("\nPlot saved as 'assignment_04_output.png'. Displaying plot...")
    plt.show()


if __name__ == "__main__":
    main()
