"""
AIML Lab Practical - Assignment 08
Aim: Implementation of Regularization Techniques (Ridge and Lasso Regression)
     to prevent overfitting on high-degree polynomial feature representations.
Author: Semester 5 AI & ML Lab
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

def main():
    print("=" * 65)
    print("      ASSIGNMENT 08: RIDGE & LASSO REGULARIZATION")
    print("=" * 65)

    # 1. Dataset Generation (Nonlinear relationship with noise)
    np.random.seed(42)
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
    # Underlying function: y = x^2 + 2x + 3 + noise
    noise = np.array([0.5, -0.8, 1.2, -1.5, 0.9, -0.4, 1.5, -1.2, 0.7, -0.9])
    y = (X.flatten() ** 2 + 2 * X.flatten() + 3 + noise)

    print(f"Feature (X): {X.flatten()}")
    print(f"Target  (y): {np.round(y, 2)}")
    print("-" * 65)

    # 2. Polynomial Feature Expansion (Degree 5 to induce potential overfitting)
    degree = 5
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(['x'])

    # Standardize polynomial features for stable regularization
    scaler = StandardScaler()
    X_poly_scaled = scaler.fit_transform(X_poly)

    # 3. Model Training
    # (a) Unregularized OLS Linear Regression
    ols = LinearRegression()
    ols.fit(X_poly_scaled, y)
    y_pred_ols = ols.predict(X_poly_scaled)

    # (b) Ridge Regression (L2 Regularization)
    alpha_ridge = 1.0
    ridge = Ridge(alpha=alpha_ridge, random_state=42)
    ridge.fit(X_poly_scaled, y)
    y_pred_ridge = ridge.predict(X_poly_scaled)

    # (c) Lasso Regression (L1 Regularization - Feature Selection / Sparsity)
    alpha_lasso = 0.5
    lasso = Lasso(alpha=alpha_lasso, max_iter=10000, random_state=42)
    lasso.fit(X_poly_scaled, y)
    y_pred_lasso = lasso.predict(X_poly_scaled)

    # 4. Evaluation and Coefficient Comparison
    print(f"Polynomial Degree: {degree}")
    print(f"Number of Features: {X_poly.shape[1]}")
    print("\nModel Coefficients:")
    header = f"{'Feature':<10} | {'OLS':>12} | {'Ridge (alpha=1.0)':>18} | {'Lasso (alpha=0.5)':>18}"
    print(header)
    print("-" * len(header))
    for name, c_ols, c_ridge, c_lasso in zip(feature_names, ols.coef_, ridge.coef_, lasso.coef_):
        print(f"{name:<10} | {c_ols:12.4f} | {c_ridge:18.4f} | {c_lasso:18.4f}")

    print("-" * 65)
    print("Performance Comparison (Training Set):")
    print(f"  OLS      - R2: {r2_score(y, y_pred_ols):.4f} | MSE: {mean_squared_error(y, y_pred_ols):.4f}")
    print(f"  Ridge    - R2: {r2_score(y, y_pred_ridge):.4f} | MSE: {mean_squared_error(y, y_pred_ridge):.4f}")
    print(f"  Lasso    - R2: {r2_score(y, y_pred_lasso):.4f} | MSE: {mean_squared_error(y, y_pred_lasso):.4f}")

    # 5. Prediction for Unseen Sample (x = 11)
    X_new = np.array([[11]])
    X_new_poly = scaler.transform(poly.transform(X_new))
    pred_ols_new = ols.predict(X_new_poly)[0]
    pred_ridge_new = ridge.predict(X_new_poly)[0]
    pred_lasso_new = lasso.predict(X_new_poly)[0]
    true_expected = 11**2 + 2 * 11 + 3  # 146

    print("-" * 65)
    print(f"Out-of-sample Extrapolation at X = 11 (True Expected ~ {true_expected}):")
    print(f"  OLS Prediction   : {pred_ols_new:.2f}")
    print(f"  Ridge Prediction : {pred_ridge_new:.2f}")
    print(f"  Lasso Prediction : {pred_lasso_new:.2f}")

    # 6. Visualization
    X_plot = np.linspace(0.5, 11.5, 300).reshape(-1, 1)
    X_plot_scaled = scaler.transform(poly.transform(X_plot))

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='black', s=60, zorder=5, label='Observed Data Points')
    plt.plot(X_plot, ols.predict(X_plot_scaled), 'r--', linewidth=2, label='OLS (High Variance / Overfitting)')
    plt.plot(X_plot, ridge.predict(X_plot_scaled), 'b-', linewidth=2.5, label='Ridge (L2 Shrinkage)')
    plt.plot(X_plot, lasso.predict(X_plot_scaled), 'g-.', linewidth=2.5, label='Lasso (L1 Sparsity)')
    plt.scatter(X_new, [pred_ridge_new], color='blue', marker='*', s=180, zorder=6, label=f'Ridge Pred (X=11): {pred_ridge_new:.1f}')

    plt.title('Assignment 08: Polynomial Regression with Ridge and Lasso Regularization', fontsize=12, pad=12)
    plt.xlabel('Input Feature (X)', fontsize=11)
    plt.ylabel('Target Value (y)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()

    plt.savefig('output_assignment8.png', dpi=300)
    print("\nPlot saved successfully as 'output_assignment8.png'")
    plt.show()

if __name__ == '__main__':
    main()
