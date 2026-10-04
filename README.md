# AI & Machine Learning Laboratory Practicals (Semester 5)

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Complete](https://img.shields.io/badge/Practicals-1%20to%2010%20Complete-brightgreen.svg)]()

This repository contains the complete, self-contained, and tested implementations of the **10 Artificial Intelligence & Machine Learning Laboratory Practicals** for **Semester 5**. 

Each practical includes the mathematical formulation, algorithmic implementation, evaluation metrics, and automated high-resolution visualization plots.

---

## 📑 Lab Practicals Index

| Assignment | Topic / Aim | Algorithm / Techniques | Script File |
| :---: | :--- | :--- | :--- |
| **01** | **Association Analysis** | Pearson Correlation Coefficient, Covariance, Scatter with OLS Trendline | [`01_association_analysis.py`](./01_association_analysis.py) |
| **02** | **K-Means Clustering** | Unsupervised Centroid Clustering, Inertia, Elbow Method Optimization | [`02_kmeans_clustering.py`](./02_kmeans_clustering.py) |
| **03** | **Dimensionality Reduction** | Principal Component Analysis (PCA), Covariance Eigendecomposition, Iris Projection | [`03_pca_dimensionality_reduction.py`](./03_pca_dimensionality_reduction.py) |
| **04** | **Naïve Bayes Classification** | Gaussian Naïve Bayes, Prior/Posterior Probability, Confusion Matrix Heatmap | [`04_naive_bayes_classification.py`](./04_naive_bayes_classification.py) |
| **05** | **K-Nearest Neighbors (KNN)** | Non-parametric Distance Classification, Euclidean Metric, $K$-hyperparameter Sweep | [`05_knn_classification.py`](./05_knn_classification.py) |
| **06** | **Linear Regression** | Ordinary Least Squares (OLS), Residual Analysis, $R^2$, MAE, MSE, RMSE | [`06_linear_regression.py`](./06_linear_regression.py) |
| **07** | **Support Vector Regression (SVR)** | Non-linear RBF Kernel, Dual Optimization, $\epsilon$-Insensitive Tube Bounds | [`07_svr_regression.py`](./07_svr_regression.py) |
| **08** | **Regularization Techniques** | Ridge ($L_2$ Shrinkage) vs. Lasso ($L_1$ Sparsity), Degree-5 Polynomial Features | [`08_regularization_ridge_lasso.py`](./08_regularization_ridge_lasso.py) |
| **09** | **CCTV Surveillance Object Detection** | YOLOv8 (You Only Look Once), Real-time Bounding Box Inference, Confidence Scoring | [`09_cctv_object_detection.py`](./09_cctv_object_detection.py) |
| **10** | **Multimodal Fake News Detection** | Multimodal Early Feature Fusion (Textual TF-IDF + Visual Channel Moments), Logistic Classifier | [`10_multimodal_fake_news_detection.py`](./10_multimodal_fake_news_detection.py) |

---

## 🛠️ Installation & Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/aiml-lab-assignments.git
cd aiml-lab-assignments
```

### 2. Create and Activate a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Execution Guide

Run each assignment directly using Python:

```bash
# Assignment 1: Association Analysis
python 01_association_analysis.py

# Assignment 2: K-Means Clustering & Elbow Analysis
python 02_kmeans_clustering.py

# Assignment 3: Principal Component Analysis (PCA)
python 03_pca_dimensionality_reduction.py

# Assignment 4: Gaussian Naive Bayes Classification
python 04_naive_bayes_classification.py

# Assignment 5: K-Nearest Neighbors Classification
python 05_knn_classification.py

# Assignment 6: Simple & Multiple Linear Regression
python 06_linear_regression.py

# Assignment 7: Support Vector Regression (SVR)
python 07_svr_regression.py

# Assignment 8: Regularization (Ridge & Lasso Regression)
python 08_regularization_ridge_lasso.py

# Assignment 9: CCTV Surveillance Object Detection (YOLOv8)
# Option A: Run default test surveillance feed
python 09_cctv_object_detection.py
# Option B: Run on a custom video or image
python 09_cctv_object_detection.py --source "path/to/cctv_footage.mp4"
# Option C: Run on live webcam
python 09_cctv_object_detection.py --source 0

# Assignment 10: Multimodal Fake News Detection
python 10_multimodal_fake_news_detection.py
```

---

## 📊 Summary of Practicals & Key Technical Highlights

### Assignment 1: Association Analysis
- Computes Pearson's $r$ correlation coefficient and covariance matrices between paired variables.
- Generates a scatter diagram overlaid with the linear regression trendline.

### Assignment 2: K-Means Clustering & Elbow Method
- Implements unsupervised clustering on multi-dimensional data ($k=3$).
- Identifies cluster centroids and computes total within-cluster sum of squares (Inertia) across $k \in [1, 8]$ to locate the optimal elbow point.

### Assignment 3: Principal Component Analysis (PCA)
- Standardizes feature dimensions and performs covariance matrix eigendecomposition.
- Projects 4-dimensional Fisher's Iris data into a 2D principal subspace retaining **95.81%** cumulative explained variance.

### Assignment 4: Gaussian Naïve Bayes Classification
- Trains a probabilistic classifier using maximum likelihood estimates under class-conditional Gaussian independence assumptions.
- Evaluates confusion matrix and precision/recall/F1 metrics across multiclass flower species.

### Assignment 5: K-Nearest Neighbors (KNN)
- Implements instance-based learning with standard feature scaling.
- Evaluates model performance across odd neighbor counts ($K=1 \dots 15$) to demonstrate the bias-variance tradeoff.

### Assignment 6: Linear Regression
- Fits an Ordinary Least Squares (OLS) regression line.
- Evaluates goodness-of-fit metrics including $R^2$ score, Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

### Assignment 7: Support Vector Regression (SVR)
- Implements $\epsilon$-SVR with a Radial Basis Function (RBF) kernel ($\gamma = 0.1, C = 100, \epsilon = 0.1$).
- Visualizes support vectors, the non-linear regression manifold, and the $\pm\epsilon$ error margin tube.

### Assignment 8: Regularization (Ridge vs. Lasso)
- Compares $L_2$ Tikhonov regularization (Ridge) against $L_1$ penalty (Lasso) on a 5th-degree polynomial feature space.
- Demonstrates how Lasso forces non-informative coefficients to exactly zero (automatic feature selection) while Ridge performs smooth shrinkage.

### Assignment 9: CCTV Surveillance Object Detection
- Leverages the state-of-the-art **YOLOv8** convolutional backbone for real-time bounding-box detection, label identification, and confidence calculation.
- Includes a surveillance HUD overlay and graceful fallback handling for headless or missing video sources.

### Assignment 10: Multimodal Fake News Detection
- Demonstrates **Early Multimodal Feature Fusion** by concatenating TF-IDF linguistic feature vectors with normalized visual color moment statistics.
- Trains a classification model to detect synthetic/misleading digital publications, reporting classification metrics and modality importance weights.

---

## 📂 Repository Structure

```text
aiml-lab-assignments/
│
├── requirements.txt                       # Python dependencies
├── README.md                              # Lab documentation & run instructions
├── .gitignore                             # Git ignore rules
│
├── 01_association_analysis.py            # Assignment 1: Pearson Correlation
├── 02_kmeans_clustering.py               # Assignment 2: K-Means & Elbow Method
├── 03_pca_dimensionality_reduction.py    # Assignment 3: PCA (Iris 2D Projection)
├── 04_naive_bayes_classification.py      # Assignment 4: Gaussian Naive Bayes
├── 05_knn_classification.py              # Assignment 5: K-Nearest Neighbors
├── 06_linear_regression.py               # Assignment 6: OLS Linear Regression
├── 07_svr_regression.py                  # Assignment 7: Support Vector Regression
├── 08_regularization_ridge_lasso.py      # Assignment 8: Ridge & Lasso Regularization
├── 09_cctv_object_detection.py           # Assignment 9: YOLOv8 Object Detection
└── 10_multimodal_fake_news_detection.py  # Assignment 10: Multimodal Fusion Classification
```

---

## 👥 Authors & Academic Attribution
- **Course**: Artificial Intelligence & Machine Learning Lab (Semester 5)
- **Language**: Python 3
- **Primary Libraries**: `numpy`, `pandas`, `scipy`, `matplotlib`, `scikit-learn`, `opencv-python`, `ultralytics`
