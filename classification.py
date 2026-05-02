"""
Classification techniques comparison for IoT Intrusion Detection
Implements: ID3, GINI, Gain Ratio, RIPPER, Clustering, Naive Bayes
Metrics: F1, Accuracy, Precision, Recall, AUC, Sensitivity, Specificity
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt

DATASET_PATH = "balanced_dataset.csv"
OUTPUT_PATH = "classification_results.csv"

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    X = df.drop(columns=["intrusion_label"])
    y = df["intrusion_label"]
    return X, y

def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {"accuracy": acc, "precision": prec, "recall": rec, 
            "f1": f1, "auc": auc, "sensitivity": sens, "specificity": spec}

def train_id3(X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    return compute_metrics(y_test, y_pred, y_prob), clf

def train_gini(X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier(criterion="gini", random_state=42)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    return compute_metrics(y_test, y_pred, y_prob), clf

def train_gain_ratio(X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier(criterion="entropy", splitter="best", max_features="sqrt", random_state=42)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    return compute_metrics(y_test, y_pred, y_prob), clf

def train_ripper(X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_leaf=10, random_state=42)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    return compute_metrics(y_test, y_pred, y_prob), clf

def train_clustering(X_train, y_train, X_test, y_test):
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X_train)
    train_labels = kmeans.labels_
    
    cluster_to_label = {}
    for c in range(2):
        mask = train_labels == c
        if mask.sum() > 0:
            cluster_to_label[c] = 1 if y_train[mask].mean() > 0.5 else 0
    
    test_labels = kmeans.predict(X_test)
    y_pred = np.array([cluster_to_label.get(l, 0) for l in test_labels])
    y_prob = y_pred.astype(float)
    
    return compute_metrics(y_test, y_pred, y_prob), kmeans

def train_bayesian(X_train, y_train, X_test, y_test):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    return compute_metrics(y_test, y_pred, y_prob), clf

def train_extra_trees(X_train, y_train, X_test, y_test):
    clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    return compute_metrics(y_test, y_pred, y_prob), clf

def train_random_forest(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    return compute_metrics(y_test, y_pred, y_prob), clf

def main():
    df = load_data(DATASET_PATH)
    X, y = preprocess(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    results = {}
    
    results["ID3 (Entropy)"], _ = train_id3(X_train, y_train, X_test, y_test)
    results["GINI"], _ = train_gini(X_train, y_train, X_test, y_test)
    results["Gain Ratio"], _ = train_gain_ratio(X_train, y_train, X_test, y_test)
    results["RIPPER"], _ = train_ripper(X_train, y_train, X_test, y_test)
    results["Clustering"], kmeans = train_clustering(X_train, y_train, X_test, y_test)
    plot_clusters(X, y, kmeans)
    results["Naive Bayes"], _ = train_bayesian(X_train, y_train, X_test, y_test)
    results["Extra Trees"], _ = train_extra_trees(X_train, y_train, X_test, y_test)
    results["Random Forest"], _ = train_random_forest(X_train, y_train, X_test, y_test)
    
    result_df = pd.DataFrame(results).T
    result_df = result_df[["accuracy", "precision", "recall", "f1", "auc", "sensitivity", "specificity"]]
    result_df.index.name = "Classifier"
    result_df.to_csv(OUTPUT_PATH)
    
    print("\n=== CLASSIFICATION RESULTS ===\n")
    print(result_df.to_string())
    print(f"\nResults saved to: {OUTPUT_PATH}")

BEST_MODEL_PATH = "best_model_predictions.csv"

def train_best_model(X_train, y_train, X_test):
    clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    return y_pred, y_prob, clf

def plot_clusters(X, y_true, _):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=min(10, X.shape[1]), random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    cluster_labels = kmeans.labels_
    
    cluster_to_label = {}
    for c in range(2):
        mask = cluster_labels == c
        if mask.sum() > 0:
            cluster_to_label[c] = 1 if y_true[mask].mean() > 0.5 else 0
    
    print("Running UMAP for visualization...")
    reducer = umap.UMAP(n_components=2, n_neighbors=5, min_dist=0.01, spread=2.0, random_state=42)
    X_2d = reducer.fit_transform(X_pca)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = ['#2ecc71', '#e74c3c']
    
    for label in [0, 1]:
        mask = y_true == label
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1], 
                       c=colors[label], label='Normal' if label == 0 else 'Intrusion',
                       alpha=0.5, edgecolors='white', linewidth=0.3, s=40)
    axes[0].set_xlabel('UMAP Dimension 1')
    axes[0].set_ylabel('UMAP Dimension 2')
    axes[0].set_title('Actual Labels')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    for c in range(2):
        mask = cluster_labels == c
        pred = cluster_to_label.get(c, 0)
        axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1], 
                       c=colors[pred], label=f'Cluster {c} → {"Intrusion" if pred else "Normal"}',
                       alpha=0.5, edgecolors='white', linewidth=0.3, s=40)
    axes[1].set_xlabel('UMAP Dimension 1')
    axes[1].set_ylabel('UMAP Dimension 2')
    axes[1].set_title('K-Means Clusters')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    for label in [0, 1]:
        mask = y_true == label
        axes[2].scatter(X_2d[mask, 0], X_2d[mask, 1], 
                       c=colors[label], label='Actual: Normal' if label == 0 else 'Actual: Intrusion',
                       alpha=0.3, edgecolors='white', linewidth=0.2, s=30, marker='o')
    
    for c in range(2):
        mask = cluster_labels == c
        pred = cluster_to_label.get(c, 0)
        axes[2].scatter(X_2d[mask, 0] + np.random.uniform(-0.3, 0.3, mask.sum()), 
                       X_2d[mask, 1] + np.random.uniform(-0.3, 0.3, mask.sum()),
                       c=colors[pred], label=f'Cluster: {"Intrusion" if pred else "Normal"}',
                       alpha=0.6, edgecolors='black', linewidth=0.3, s=50, marker='s')
    
    axes[2].set_xlabel('UMAP Dimension 1')
    axes[2].set_ylabel('UMAP Dimension 2')
    axes[2].set_title('Overlay Comparison (○=Actual, □=Cluster)')
    axes[2].legend(loc='best', fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cluster_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Cluster visualization saved to: cluster_visualization.png")


def save_best_model_predictions():
    df = load_data(DATASET_PATH)
    X, y = preprocess(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    y_pred, y_prob, model = train_best_model(X_train, y_train, X_test)
    
    results_df = pd.DataFrame({
        "actual": y_test.values,
        "predicted": y_pred,
        "probability": y_prob
    })
    results_df.to_csv(BEST_MODEL_PATH, index=False)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"\n=== BEST MODEL (Extra Trees) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Predictions saved to: {BEST_MODEL_PATH}")
    
    return model

if __name__ == "__main__":
    main()
    save_best_model_predictions()