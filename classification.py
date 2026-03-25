"""
Classification Pipeline for IoT Network Intrusion Detection Dataset
====================================================================
Applies 6 classifiers: Decision Tree, Naive Bayes, KNN, SVM, Random Forest, MLP
Handles class imbalance with SMOTE and class_weight, generates full evaluation.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Imbalance handling
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ─── Configuration ───────────────────────────────────────────────────────────
DATA_PATH = Path(r"D:\work\dmProj\AI_Powered_IoT_Network_Intrusion_Detection_Dataset.csv")
OUTPUT_DIR = Path(r"D:\work\dmProj\results")
RANDOM_STATE = 42
TEST_SIZE = 0.2

OUTPUT_DIR.mkdir(exist_ok=True)
print("=" * 70)
print("  IoT Network Intrusion Detection — Classification Pipeline")
print("=" * 70)

# ─── 1. Load & Preprocess ────────────────────────────────────────────────────
print("\n[1/6] Loading and preprocessing data...")
df = pd.read_csv(DATA_PATH)
print(f"  Dataset shape: {df.shape}")
print(f"  Target distribution:\n    {df['intrusion_label'].value_counts().to_dict()}")

imbalance_ratio = df['intrusion_label'].value_counts()[0] / df['intrusion_label'].value_counts()[1]
print(f"  Imbalance ratio (majority/minority): {imbalance_ratio:.1f}:1")
print("  => Applying SMOTE oversampling + class_weight='balanced' to handle imbalance")

# Drop device_id (unique identifier, not a feature)
df = df.drop(columns=["device_id"])

# Encode categorical features
label_encoders = {}
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
print(f"  Categorical columns to encode: {categorical_cols}")

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separate features and target
X = df.drop(columns=["intrusion_label"])
y = df["intrusion_label"]

# Train-test split (stratified to preserve class ratios)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"  Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
print(f"  Train target dist: {dict(y_train.value_counts())}")
print(f"  Test  target dist: {dict(y_test.value_counts())}")

# Apply SMOTE to training data
smote = SMOTE(random_state=RANDOM_STATE)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"  After SMOTE: {X_train_resampled.shape[0]} samples, dist: {dict(pd.Series(y_train_resampled).value_counts())}")

# Scale features (needed for KNN, SVM, MLP)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Also keep unscaled resampled data for tree-based models
X_train_unscaled = X_train_resampled.values if hasattr(X_train_resampled, 'values') else X_train_resampled
X_test_unscaled = X_test.values if hasattr(X_test, 'values') else X_test

# ─── 2. Define Classifiers ───────────────────────────────────────────────────
print("\n[2/6] Setting up classifiers...")

classifiers = {
    "Decision Tree (CART)": {
        "model": DecisionTreeClassifier(
            criterion="gini", max_depth=10, class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "needs_scaling": False,
        "description": "CART algorithm with Gini impurity, max_depth=10, balanced class weights"
    },
    "Decision Tree (Entropy/ID3)": {
        "model": DecisionTreeClassifier(
            criterion="entropy", max_depth=10, class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "needs_scaling": False,
        "description": "ID3-style using entropy/information gain, max_depth=10, balanced class weights"
    },
    "Naive Bayes (Gaussian)": {
        "model": GaussianNB(),
        "needs_scaling": False,
        "description": "Gaussian Naive Bayes — assumes feature independence"
    },
    "KNN (K=3)": {
        "model": KNeighborsClassifier(n_neighbors=3, weights='distance'),
        "needs_scaling": True,
        "description": "K-Nearest Neighbors with K=3, distance-weighted"
    },
    "KNN (K=5)": {
        "model": KNeighborsClassifier(n_neighbors=5, weights='distance'),
        "needs_scaling": True,
        "description": "K-Nearest Neighbors with K=5, distance-weighted"
    },
    "KNN (K=7)": {
        "model": KNeighborsClassifier(n_neighbors=7, weights='distance'),
        "needs_scaling": True,
        "description": "K-Nearest Neighbors with K=7, distance-weighted"
    },
    "KNN (K=11)": {
        "model": KNeighborsClassifier(n_neighbors=11, weights='distance'),
        "needs_scaling": True,
        "description": "K-Nearest Neighbors with K=11, distance-weighted"
    },
    "SVM (Linear)": {
        "model": SVC(kernel="linear", class_weight="balanced",
                     probability=True, random_state=RANDOM_STATE),
        "needs_scaling": True,
        "description": "Support Vector Machine with linear kernel, balanced class weights"
    },
    "SVM (RBF)": {
        "model": SVC(kernel="rbf", class_weight="balanced",
                     probability=True, random_state=RANDOM_STATE),
        "needs_scaling": True,
        "description": "Support Vector Machine with RBF (Gaussian) kernel, balanced class weights"
    },
    "Random Forest": {
        "model": RandomForestClassifier(
            n_estimators=200, max_depth=15, class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "needs_scaling": False,
        "description": "Ensemble of 200 decision trees, max_depth=15, balanced class weights"
    },
    "MLP Neural Network": {
        "model": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            max_iter=500,
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1
        ),
        "needs_scaling": True,
        "description": "Multi-Layer Perceptron with layers (128, 64, 32), ReLU activation"
    },
}

# ─── 3. Train & Evaluate ─────────────────────────────────────────────────────
print("\n[3/6] Training and evaluating classifiers...\n")

results = {}

for name, config in classifiers.items():
    print(f"  Training: {name}...", end=" ", flush=True)
    model = config["model"]
    use_scaled = config["needs_scaling"]

    X_tr = X_train_scaled if use_scaled else X_train_unscaled
    X_te = X_test_scaled if use_scaled else X_test_unscaled

    start_time = time.time()
    model.fit(X_tr, y_train_resampled)
    train_time = time.time() - start_time

    # Predictions
    start_time = time.time()
    y_pred = model.predict(X_te)
    predict_time = time.time() - start_time

    # Probabilities (for ROC-AUC)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_te)[:, 1]
    else:
        y_proba = model.decision_function(X_te)

    # Cross-validation on resampled training data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_tr, y_train_resampled, cv=cv, scoring="f1")

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    results[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "cv_f1_mean": cv_scores.mean(),
        "cv_f1_std": cv_scores.std(),
        "train_time": train_time,
        "predict_time": predict_time,
        "description": config["description"],
        "y_proba": y_proba,
        "y_pred": y_pred,
    }

    print(f"Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  AUC={roc_auc:.4f}  "
          f"(train: {train_time:.3f}s)")

# ─── 4. Generate Plots ───────────────────────────────────────────────────────
print("\n[4/6] Generating plots...")

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# --- 4a. Comparison Bar Chart ---
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

model_names = list(results.keys())
metrics_to_plot = {
    "Accuracy": [results[n]["accuracy"] for n in model_names],
    "Precision": [results[n]["precision"] for n in model_names],
    "Recall": [results[n]["recall"] for n in model_names],
    "F1-Score": [results[n]["f1_score"] for n in model_names],
}

x = np.arange(len(model_names))
width = 0.2
colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

for i, (metric, values) in enumerate(metrics_to_plot.items()):
    axes[0].bar(x + i * width, values, width, label=metric, color=colors[i], alpha=0.85)

axes[0].set_xticks(x + 1.5 * width)
axes[0].set_xticklabels(model_names, rotation=45, ha="right", fontsize=7)
axes[0].set_ylabel("Score")
axes[0].set_title("Classification Metrics Comparison", fontsize=13, fontweight="bold")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].set_ylim(0, 1.15)
axes[0].grid(axis="y", alpha=0.3)

# ROC-AUC comparison
auc_scores = [results[n]["roc_auc"] for n in model_names]
bar_colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(model_names)))
bars = axes[1].barh(model_names, auc_scores, color=bar_colors, alpha=0.85)
axes[1].set_xlabel("ROC-AUC Score")
axes[1].set_title("ROC-AUC Comparison", fontsize=13, fontweight="bold")
axes[1].set_xlim(0.4, 1.05)

for bar, score in zip(bars, auc_scores):
    axes[1].text(score + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{score:.4f}", va="center", fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "metrics_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: metrics_comparison.png")

# --- 4b. ROC Curves ---
fig, ax = plt.subplots(figsize=(10, 8))
cmap = plt.cm.tab20(np.linspace(0, 1, len(model_names)))

for i, name in enumerate(model_names):
    fpr, tpr, _ = roc_curve(y_test, results[name]["y_proba"])
    ax.plot(fpr, tpr, label=f'{name} (AUC={results[name]["roc_auc"]:.4f})',
            color=cmap[i], linewidth=1.8)

ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label="Random Classifier")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — All Classifiers", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: roc_curves.png")

# --- 4c. Confusion Matrices ---
n_classifiers = len(model_names)
n_cols = 4
n_rows = (n_classifiers + n_cols - 1) // n_cols

fig, axes_cm = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.2, n_rows * 3.8))
axes_cm = axes_cm.flatten()

for i, name in enumerate(model_names):
    cm = np.array(results[name]["confusion_matrix"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes_cm[i],
                xticklabels=["Normal (0)", "Intrusion (1)"],
                yticklabels=["Normal (0)", "Intrusion (1)"])
    axes_cm[i].set_title(name, fontsize=9, fontweight="bold")
    axes_cm[i].set_ylabel("Actual" if i % n_cols == 0 else "")
    axes_cm[i].set_xlabel("Predicted")

for j in range(i + 1, len(axes_cm)):
    axes_cm[j].set_visible(False)

plt.suptitle("Confusion Matrices — All Classifiers", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: confusion_matrices.png")

# --- 4d. Cross-Validation F1 Scores ---
fig, ax = plt.subplots(figsize=(13, 5))
cv_means = [results[n]["cv_f1_mean"] for n in model_names]
cv_stds = [results[n]["cv_f1_std"] for n in model_names]

bars = ax.bar(model_names, cv_means, yerr=cv_stds, capsize=5,
              color=plt.cm.plasma(np.linspace(0.2, 0.8, len(model_names))),
              alpha=0.85, edgecolor="white", linewidth=0.8)

for bar, mean in zip(bars, cv_means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
            f"{mean:.4f}", ha="center", va="bottom", fontsize=8)

ax.set_ylabel("F1-Score (5-Fold CV)")
ax.set_title("Cross-Validation F1-Scores (Mean ± Std)", fontsize=13, fontweight="bold")
ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=7)
ax.set_ylim(0, 1.15)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cv_f1_scores.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: cv_f1_scores.png")

# --- 4e. KNN K-value comparison ---
fig, ax = plt.subplots(figsize=(8, 5))
knn_names = [n for n in model_names if n.startswith("KNN")]
knn_k_vals = [3, 5, 7, 11]
knn_metrics = {
    "Accuracy": [results[n]["accuracy"] for n in knn_names],
    "Precision": [results[n]["precision"] for n in knn_names],
    "Recall": [results[n]["recall"] for n in knn_names],
    "F1-Score": [results[n]["f1_score"] for n in knn_names],
    "ROC-AUC": [results[n]["roc_auc"] for n in knn_names],
}

markers = ['o', 's', '^', 'D', 'v']
for (metric, values), marker in zip(knn_metrics.items(), markers):
    ax.plot(knn_k_vals, values, f"{marker}-", label=metric, linewidth=2, markersize=8)

ax.set_xlabel("K Value", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("KNN: Effect of K Value on Performance", fontsize=13, fontweight="bold")
ax.legend()
ax.set_xticks(knn_k_vals)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "knn_k_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: knn_k_comparison.png")

# --- 4f. Training Time Comparison ---
fig, ax = plt.subplots(figsize=(10, 5))
train_times = [results[n]["train_time"] for n in model_names]
bars = ax.barh(model_names, train_times,
               color=plt.cm.coolwarm(np.linspace(0.2, 0.8, len(model_names))),
               alpha=0.85)

for bar, t in zip(bars, train_times):
    ax.text(t + max(train_times) * 0.02, bar.get_y() + bar.get_height() / 2,
            f"{t:.3f}s", va="center", fontsize=8)

ax.set_xlabel("Training Time (seconds)")
ax.set_title("Training Time Comparison", fontsize=13, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_time.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: training_time.png")

# --- 4g. Feature Importance (from Random Forest) ---
fig, ax = plt.subplots(figsize=(10, 6))
rf_model = classifiers["Random Forest"]["model"]
feature_names = X.columns.tolist()
importances = rf_model.feature_importances_
sorted_idx = np.argsort(importances)

ax.barh([feature_names[i] for i in sorted_idx], importances[sorted_idx],
        color=plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_idx))), alpha=0.85)
ax.set_xlabel("Feature Importance")
ax.set_title("Random Forest — Feature Importance", fontsize=13, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: feature_importance.png")

# ─── 5. Save Results JSON ────────────────────────────────────────────────────
print("\n[5/6] Saving results data...")

results_export = {}
for name, data in results.items():
    results_export[name] = {
        k: v for k, v in data.items()
        if k not in ("y_proba", "y_pred")
    }

with open(OUTPUT_DIR / "results.json", "w") as f:
    json.dump(results_export, f, indent=2)
print("  Saved: results.json")

# ─── 6. Summary Table ────────────────────────────────────────────────────────
print("\n[6/6] Final Summary Table\n")

summary_data = []
for name in model_names:
    r = results[name]
    summary_data.append({
        "Classifier": name,
        "Accuracy": f"{r['accuracy']:.4f}",
        "Precision": f"{r['precision']:.4f}",
        "Recall": f"{r['recall']:.4f}",
        "F1-Score": f"{r['f1_score']:.4f}",
        "ROC-AUC": f"{r['roc_auc']:.4f}",
        "CV-F1 (mean +/- std)": f"{r['cv_f1_mean']:.4f} +/- {r['cv_f1_std']:.4f}",
        "Train Time (s)": f"{r['train_time']:.3f}",
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

summary_df.to_csv(OUTPUT_DIR / "summary_table.csv", index=False)
print(f"\n  Saved: summary_table.csv")

# Identify best
best_f1_name = max(results, key=lambda n: results[n]["f1_score"])
best_auc_name = max(results, key=lambda n: results[n]["roc_auc"])
best_rec_name = max(results, key=lambda n: results[n]["recall"])
print(f"\n  Best F1-Score:  {best_f1_name} ({results[best_f1_name]['f1_score']:.4f})")
print(f"  Best ROC-AUC:   {best_auc_name} ({results[best_auc_name]['roc_auc']:.4f})")
print(f"  Best Recall:    {best_rec_name} ({results[best_rec_name]['recall']:.4f})")

print("\n" + "=" * 70)
print("  Pipeline complete! All results saved to:", OUTPUT_DIR)
print("=" * 70)
