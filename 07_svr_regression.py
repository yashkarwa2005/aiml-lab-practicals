"""
AIML Laboratory - Semester 5
Assignment 7: Support Vector Regression (SVR) Algorithm

Aim:
    To implement Support Vector Regression (SVR) using an RBF kernel, analyze
    the effect of the epsilon margin, and evaluate regression metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score


def main():
    print("=" * 70)
    print("ASSIGNMENT 7: Support Vector Regression (SVR)")
    print("=" * 70)

    # 1. Dataset
    X = np.array([[1], [2], [3], [4], [5],
                  [6], [7], [8], [9], [10]])
    y = np.array([10, 13, 17, 21, 24,
                  29, 31, 36, 40, 43])

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train SVR with Radial Basis Function (RBF) kernel
    epsilon = 0.1
    model = SVR(kernel="rbf", C=100, gamma="scale", epsilon=epsilon)
    model.fit(X_train_scaled, y_train)

    # 5. Evaluate on test set
    predictions = model.predict(X_test_scaled)
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    print(f"Predictions on Test Set: {np.round(predictions, 2)}")
    print(f"Actual Test Targets:     {y_test}")
    print(f"\nEvaluation Metrics:")
    print(f"  - R² Score: {r2:.4f}")
    print(f"  - MSE:      {mse:.4f}")
    print(f"  - RMSE:     {rmse:.4f}")

    # 6. Smooth prediction curve for visualization
    grid_x = np.linspace(1, 10, 200).reshape(-1, 1)
    grid_x_scaled = scaler.transform(grid_x)
    grid_y = model.predict(grid_x_scaled)

    plt.figure(figsize=(7.2, 5), dpi=300)
    plt.scatter(X, y, color="#264653", s=70, label="Observed Data", zorder=5)
    plt.plot(grid_x, grid_y, color="#e76f51", linewidth=2.5, label="SVR Fit (RBF Kernel)")
    plt.fill_between(
        grid_x.ravel(),
        grid_y - 0.5,
        grid_y + 0.5,
        color="#e76f51",
        alpha=0.18,
        label=f"Epsilon Tube (ε = {epsilon})"
    )

    plt.title("Support Vector Regression (SVR with RBF Kernel)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Independent Variable (X)", fontsize=11)
    plt.ylabel("Dependent Variable (y)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()

    # Save figure and display
    plt.savefig("assignment_07_output.png", dpi=300)
    print("\nPlot saved as 'assignment_07_output.png'. Displaying plot...")
    plt.show()


if __name__ == "__main__":
    main()
