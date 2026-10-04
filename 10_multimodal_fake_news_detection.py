"""
AIML Lab Practical - Assignment 10
Aim: Implement Multimodal Fake News Detection System Combining Textual and
     Visual Features Using Feature Fusion and Machine Learning Classification.
Author: Semester 5 AI & ML Lab
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def generate_synthetic_multimodal_dataset(n_samples=200):
    """
    Creates a reproducible multimodal dataset of news headlines and simulated
    visual feature distributions (RGB channel moments, edge density, brightness)
    representing Real (Class 0) vs Fake (Class 1) news articles.
    """
    np.random.seed(42)

    fake_headlines = [
        "Shocking revelation! Secret cure for all diseases discovered overnight!",
        "Alien spacecraft landed in central park, government caught hiding truth!",
        "Earn 10,000 dollars daily from home with this secret trick doctors hate!",
        "Celebrity caught in massive scandal, you won't believe what happened next!",
        "Miracle diet burns 30 pounds in 48 hours without exercise or diet!",
        "Ancient prophecy reveals world will end next Tuesday morning!",
        "Viral video reveals top secret military experiment went wrong!",
        "Billionaire reveals loophole that makes anyone a millionaire instantly!",
        "Exclusive proof that historical moon landing was staged in Hollywood!",
        "Unbelievable discovery: drinking water causes unexpected biological mutation!"
    ]

    real_headlines = [
        "Federal reserve announces adjustment to baseline interest rates following quarterly review.",
        "Scientists publish comprehensive study on atmospheric carbon capture efficiency in Nature journal.",
        "Municipal city council approves funding for public transit expansion and infrastructure repair.",
        "Technology consortium releases open-source specifications for encrypted data exchange protocols.",
        "Agricultural department issues annual harvest yield projection report for major grain crops.",
        "University medical research team conducts phase three clinical trial for respiratory therapeutics.",
        "Global aerospace agency launches environmental observation satellite into sun-synchronous orbit.",
        "Economic bureau reports steady unemployment numbers aligned with market consensus forecasts.",
        "International treaty signed by member delegations to preserve marine conservation zones.",
        "National meteorological agency provides seasonal monsoon forecast and flood mitigation guidelines."
    ]

    headlines = []
    labels = []
    visual_features = []

    for i in range(n_samples):
        is_fake = np.random.rand() > 0.5
        if is_fake:
            base_text = np.random.choice(fake_headlines)
            # Fake news imagery often exhibits higher saturation, heavy editing, or extreme contrast
            # Features: [Mean_R, Mean_G, Mean_B, Std_R, Std_G, Std_B, Edge_Density, Brightness]
            vis = np.random.normal(loc=[180, 110, 130, 65, 70, 75, 0.45, 160],
                                   scale=[15, 15, 15, 8, 8, 8, 0.05, 12], size=8)
            label = 1
        else:
            base_text = np.random.choice(real_headlines)
            # Real news imagery tends to have balanced color distribution and natural lighting
            vis = np.random.normal(loc=[120, 125, 122, 40, 42, 40, 0.25, 125],
                                   scale=[12, 12, 12, 6, 6, 6, 0.04, 10], size=8)
            label = 0

        # Inject minor linguistic variation
        variation = f" (Report ID #{1000 + i})"
        headlines.append(base_text + variation)
        labels.append(label)
        visual_features.append(vis)

    return headlines, np.array(visual_features), np.array(labels)

def main():
    print("=" * 65)
    print("   ASSIGNMENT 10: MULTIMODAL FAKE NEWS DETECTION")
    print("=" * 65)

    # 1. Dataset Acquisition
    print("[1] Generating multimodal dataset (Textual + Visual modalities)...")
    headlines, visual_features, labels = generate_synthetic_multimodal_dataset(n_samples=250)
    print(f"    Total Samples: {len(labels)} (Real: {sum(labels==0)}, Fake: {sum(labels==1)})")
    print(f"    Visual Feature Dimensions: {visual_features.shape[1]}")

    # 2. Train-Test Split (Stratified)
    X_text_tr, X_text_te, X_vis_tr, X_vis_te, y_train, y_test = train_test_split(
        headlines, visual_features, labels, test_size=0.25, random_state=42, stratify=labels
    )

    # 3. Modality 1: Text Feature Extraction via TF-IDF Vectorization
    print("[2] Extracting textual features via TF-IDF...")
    tfidf = TfidfVectorizer(max_features=60, stop_words="english", ngram_range=(1, 2))
    X_text_tr_vec = tfidf.fit_transform(X_text_tr).toarray()
    X_text_te_vec = tfidf.transform(X_text_te).toarray()
    print(f"    Text TF-IDF Features: {X_text_tr_vec.shape[1]}")

    # 4. Modality 2: Visual Feature Normalization
    print("[3] Normalizing visual feature descriptors...")
    scaler = StandardScaler()
    X_vis_tr_scaled = scaler.fit_transform(X_vis_tr)
    X_vis_te_scaled = scaler.transform(X_vis_te)

    # 5. Multimodal Feature Fusion (Early Fusion / Concatenation)
    print("[4] Performing Multimodal Early Fusion (Concatenation)...")
    X_train_fused = np.hstack((X_text_tr_vec, X_vis_tr_scaled))
    X_test_fused = np.hstack((X_text_te_vec, X_vis_te_scaled))
    print(f"    Total Fused Feature Dimension: {X_train_fused.shape[1]}")

    # 6. Classifier Training (Logistic Regression / Fusion Classifier)
    print("[5] Training Multimodal Classification Model...")
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_train_fused, y_train)

    # 7. Model Inference and Evaluation
    y_pred = clf.predict(X_test_fused)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("-" * 65)
    print(f"Model Accuracy on Test Set: {acc * 100:.2f}%\n")
    print("Classification Report:")
    target_names = ["Real News (0)", "Fake News (1)"]
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 8. Visualization of Confusion Matrix and Modality Weights
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: Confusion Matrix Heatmap
    im = axes[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0].set_title(f'Confusion Matrix (Acc: {acc*100:.1f}%)', fontsize=12)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(target_names))
    axes[0].set_xticks(tick_marks)
    axes[0].set_xticklabels(target_names, rotation=20)
    axes[0].set_yticks(tick_marks)
    axes[0].set_yticklabels(target_names)
    axes[0].set_ylabel('True Label', fontsize=10)
    axes[0].set_xlabel('Predicted Label', fontsize=10)

    # Add numeric labels to confusion matrix cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black",
                         fontsize=14, fontweight='bold')

    # Subplot 2: Modality Contribution Comparison
    text_feat_count = X_text_tr_vec.shape[1]
    vis_feat_count = X_vis_tr_scaled.shape[1]
    coefs = clf.coef_[0]

    avg_text_importance = np.mean(np.abs(coefs[:text_feat_count]))
    avg_vis_importance = np.mean(np.abs(coefs[text_feat_count:]))

    modalities = ['Textual (TF-IDF)', 'Visual (Image Descriptors)']
    importances = [avg_text_importance, avg_vis_importance]
    colors = ['#3498db', '#e74c3c']

    axes[1].bar(modalities, importances, color=colors, width=0.45, edgecolor='black', alpha=0.85)
    axes[1].set_title('Mean Feature Importance by Modality', fontsize=12)
    axes[1].set_ylabel('Mean Absolute Coefficient Weight', fontsize=10)
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)
    for i, v in enumerate(importances):
        axes[1].text(i, v + 0.01, f"{v:.4f}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('output_assignment10.png', dpi=300)
    print("[SUCCESS] Evaluation plot saved as 'output_assignment10.png'")
    plt.show()

if __name__ == "__main__":
    main()
