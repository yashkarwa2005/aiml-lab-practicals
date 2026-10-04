"""
AIML Laboratory - Semester 5
Assignment 6: Simple Linear Regression Algorithm

Aim:
    To implement the Simple Linear Regression algorithm using Ordinary Least
    Squares (OLS) and evaluate model performance using regression metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    print("=" * 70)
    print("ASSIGNMENT 6: Simple Linear Regression Algorithm")
    print("=" * 70)

    # 1. Dataset: Input features X and Target values y
    X = np.array([[1], [2], [3], [4], [5],
                  [6], [7], [8], [9], [10]])
    y = np.array([35, 40, 45, 50, 55,
                  60, 65, 70, 75, 80])

    # 2. Split dataset into train (80%) and test (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Fit Ordinary Least Squares Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4. Predict on test set
    predictions = model.predict(X_test)

    # 5. Extract equation parameters and evaluation metrics
    slope = model.coef_[0]
    intercept = model.intercept_
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print(f"Fitted Equation: y = {slope:.2f} * X + {intercept:.2f}")
    print(f"Slope (Coefficient): {slope:.4f}")
    print(f"Intercept: {intercept:.4f}")
    print(f"\nEvaluation Metrics:")
    print(f"  - Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  - Mean Squared Error (MSE):  {mse:.4f}")
    print(f"  - Root Mean Squared Error:   {rmse:.4f}")
    print(f"  - Coefficient of Det. (R²):  {r2:.4f}")

    # 6. Visualize regression line and observations
    plt.figure(figsize=(7, 5), dpi=300)
    plt.scatter(X_train, y_train, color="#1d3557", s=65, label="Train Observations")
    plt.scatter(X_test, y_test, color="#e63946", s=85, marker="^", label="Test Observations")

    x_line = np.linspace(1, 10, 100).reshape(-1, 1)
    y_line = model.predict(x_line)
    plt.plot(x_line, y_line, color="#457b9d", linewidth=2.2, label=f"Fit Line (y = {slope:.1f}x + {intercept:.1f})")

    plt.title("Linear Regression Model Fit", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Independent Variable (X)", fontsize=11)
    plt.ylabel("Dependent Variable (y)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()

    # Save figure and display
    plt.savefig("assignment_06_output.png", dpi=300)
    print("\nPlot saved as 'assignment_06_output.png'. Displaying plot...")
    plt.show()


if __name__ == "__main__":
    main()
