# CMPE 462 - Assignment 2: Fruit and Vegetable Classification
# Task 1: Classification & Task 2: Unsupervised Learning

import time
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

# Output to both console and file
class OutputLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

# Start logging
logger = OutputLogger('output.txt')
sys.stdout = logger

# ============================================================
# Load Data
# ============================================================
print("Loading data...")
y = pd.read_csv("labels.csv", index_col=0).to_numpy().flatten()
X = pd.read_csv("fused.csv", index_col=0).to_numpy()  # Using fused features
X_image = pd.read_csv("image.csv", index_col=0).to_numpy()
X_text = pd.read_csv("text.csv", index_col=0).to_numpy()
X_meta = pd.read_csv("meta.csv", index_col=0).to_numpy().astype(float)
X_fused = pd.read_csv("fused.csv", index_col=0).to_numpy()
print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Classes: {np.unique(y)}")
print(f"Samples per class: {pd.Series(y).value_counts().to_dict()}")

# ============================================================
# Feature Set Comparison (to determine best dataset)
# ============================================================
print("\n" + "="*70)
print("FEATURE SET COMPARISON")
print("="*70)

feature_sets = {
    "Image": X_image,
    "Text": X_text,
    "Meta": X_meta,
    "Fused": X_fused
}

print(f"\nFeature dimensions:")
for name, X_feat in feature_sets.items():
    print(f"  {name}: {X_feat.shape[1]} features")

# Compare feature sets using SVM (RBF)
feature_comparison = []
for name, X_feat in feature_sets.items():
    scaler_temp = StandardScaler()
    X_feat_scaled = scaler_temp.fit_transform(X_feat)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_feat_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    svm_temp = SVC(kernel='rbf', C=10, gamma='scale')
    svm_temp.fit(X_tr, y_tr)
    acc = svm_temp.score(X_te, y_te)
    cv_scores = cross_val_score(svm_temp, X_feat_scaled, y, cv=5)
    
    feature_comparison.append({
        "Feature Set": name,
        "Num Features": X_feat.shape[1],
        "Test Accuracy": round(acc, 4),
        "CV Mean": round(cv_scores.mean(), 4),
        "CV Std": round(cv_scores.std(), 4)
    })

feature_comp_df = pd.DataFrame(feature_comparison)
print("\nFeature Set Performance (SVM RBF, C=10):")
print(feature_comp_df.to_string(index=False))

# Determine best feature set
best_feature_set = feature_comp_df.loc[feature_comp_df['CV Mean'].idxmax(), 'Feature Set']
print(f"\n=> Best feature set: {best_feature_set}")

# Early vs Late Fusion Comparison
print("\n" + "-"*70)
print("EARLY vs LATE FUSION")
print("-"*70)

# Early Fusion: Concatenate all features (already done in fused.csv)
scaler_early = StandardScaler()
X_early_scaled = scaler_early.fit_transform(X_fused)
X_tr_early, X_te_early, y_tr, y_te = train_test_split(
    X_early_scaled, y, test_size=0.2, random_state=42, stratify=y
)
svm_early = SVC(kernel='rbf', C=10, gamma='scale')
svm_early.fit(X_tr_early, y_tr)
early_acc = svm_early.score(X_te_early, y_te)
print(f"Early Fusion (concatenated features) Accuracy: {early_acc:.4f}")

# Late Fusion: Train separate models and combine predictions
from sklearn.base import BaseEstimator, ClassifierMixin

class LateFusionClassifier(BaseEstimator, ClassifierMixin):
    """Late fusion: train separate classifiers, average probabilities."""
    def __init__(self):
        self.classifiers = []
        self.scalers = []
        
    def fit(self, X_list, y):
        self.classifiers = []
        self.scalers = []
        for X in X_list:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
            clf.fit(X_scaled, y)
            self.classifiers.append(clf)
            self.scalers.append(scaler)
        self.classes_ = self.classifiers[0].classes_
        return self
    
    def predict(self, X_list):
        probs = np.zeros((X_list[0].shape[0], len(self.classes_)))
        for X, clf, scaler in zip(X_list, self.classifiers, self.scalers):
            X_scaled = scaler.transform(X)
            probs += clf.predict_proba(X_scaled)
        probs /= len(self.classifiers)
        return self.classes_[np.argmax(probs, axis=1)]

# Split each feature set with same random state
X_tr_img, X_te_img, y_tr, y_te = train_test_split(X_image, y, test_size=0.2, random_state=42, stratify=y)
X_tr_txt, X_te_txt, _, _ = train_test_split(X_text, y, test_size=0.2, random_state=42, stratify=y)
X_tr_meta, X_te_meta, _, _ = train_test_split(X_meta, y, test_size=0.2, random_state=42, stratify=y)

late_clf = LateFusionClassifier()
late_clf.fit([X_tr_img, X_tr_txt, X_tr_meta], y_tr)
late_preds = late_clf.predict([X_te_img, X_te_txt, X_te_meta])
late_acc = accuracy_score(y_te, late_preds)
print(f"Late Fusion (probability averaging) Accuracy: {late_acc:.4f}")

if early_acc > late_acc:
    print(f"\n=> Early fusion performs better (+{early_acc-late_acc:.4f})")
else:
    print(f"\n=> Late fusion performs better (+{late_acc-early_acc:.4f})")

# Visualization: Feature Set Comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar plot for feature set accuracy
colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
ax1 = axes[0]
bars = ax1.bar(feature_comp_df['Feature Set'], feature_comp_df['CV Mean'], 
               yerr=feature_comp_df['CV Std'], capsize=5, color=colors, edgecolor='black')
ax1.set_ylabel('Cross-Validation Accuracy')
ax1.set_title('Feature Set Comparison')
ax1.set_ylim(0, 1)
for i, (bar, acc) in enumerate(zip(bars, feature_comp_df['CV Mean'])):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, 
             f'{acc:.3f}', ha='center', fontsize=10)

# Bar plot for early vs late fusion
ax2 = axes[1]
fusion_names = ['Early Fusion\n(Concatenation)', 'Late Fusion\n(Prob. Averaging)']
fusion_accs = [early_acc, late_acc]
bars2 = ax2.bar(fusion_names, fusion_accs, color=['#3498db', '#e74c3c'], edgecolor='black')
ax2.set_ylabel('Test Accuracy')
ax2.set_title('Early vs Late Fusion')
ax2.set_ylim(0, 1)
for bar, acc in zip(bars2, fusion_accs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{acc:.3f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('feature_comparison.png', dpi=150)
print("\nSaved feature comparison to 'feature_comparison.png'")

# Use the best performing feature set for remaining tasks
X_best = feature_sets["Image"]
print(f"\nUsing 'Image' features for remaining tasks.")

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_best)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# TASK 1.1: Benchmarking Classifiers
# ============================================================
print("\n" + "="*70)
print("TASK 1.1: Benchmarking Classifiers")
print("="*70)

# Define classifiers with hyperparameter grids
# Hyperparameters determined via 5-fold cross-validation GridSearch
classifiers = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=2000, random_state=42),
        "params": {"C": [0.001, 0.01, 0.1, 1, 10, 100]},
        "description": "L2 regularization, C controls inverse regularization strength"
    },
    "LogReg + Poly(degree=2)": {
        "model": LogisticRegression(max_iter=2000, random_state=42),
        "params": {"C": [0.001, 0.01, 0.1, 1, 10]},
        "poly_degree": 2,
        "description": "Polynomial features (degree=2) for non-linear transformation"
    },
    "SVM (Linear)": {
        "model": SVC(kernel='linear', random_state=42),
        "params": {"C": [0.001, 0.01, 0.1, 1, 10, 100]},
        "description": "Soft-margin linear SVM, C controls margin softness"
    },
    "SVM (RBF Kernel)": {
        "model": SVC(kernel='rbf', random_state=42),
        "params": {"C": [0.1, 1, 10, 100], "gamma": ['scale', 'auto', 0.01, 0.1]},
        "description": "RBF kernel for non-linear decision boundary"
    },
    "k-NN": {
        "model": KNeighborsClassifier(),
        "params": {"n_neighbors": [3, 5, 7, 9, 11, 15], "weights": ['uniform', 'distance']},
        "description": "k determined by CV, distance weighting option"
    },
    "Naive Bayes": {
        "model": GaussianNB(),
        "params": {"var_smoothing": [1e-9, 1e-8, 1e-7]},
        "description": "Gaussian NB with variance smoothing"
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20, 30]},
        "description": "Ensemble of decision trees"
    }
}

results = []
trained_models = {}

for name, config in classifiers.items():
    print(f"\nTraining {name}...")
    
    # Handle polynomial features
    if "poly_degree" in config:
        poly = PolynomialFeatures(degree=config["poly_degree"], include_bias=False)
        X_tr = poly.fit_transform(X_train)
        X_te = poly.transform(X_test)
    else:
        X_tr, X_te = X_train, X_test
    
    # GridSearchCV for hyperparameter tuning
    start_time = time.time()
    grid_search = GridSearchCV(config["model"], config["params"], cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_tr, y_train)
    train_time = time.time() - start_time
    
    # Evaluate
    y_pred = grid_search.predict(X_te)
    
    results.append({
        "Classifier": name,
        "Best Params": str(grid_search.best_params_),
        "CV Accuracy": round(grid_search.best_score_, 4),
        "Test Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Training Time (s)": round(train_time, 4)
    })
    trained_models[name] = grid_search.best_estimator_

# Display results table
results_df = pd.DataFrame(results)
print("\n" + "-"*70)
print("CLASSIFIER BENCHMARK RESULTS")
print("-"*70)
print(results_df.to_string(index=False))

# Hyperparameter choices table
print("\n" + "-"*70)
print("HYPERPARAMETER CHOICES")
print("-"*70)
for name, config in classifiers.items():
    print(f"{name}: {config['description']}")
    print(f"  Search space: {config['params']}")

# Visualization: Classifier Benchmark
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot for accuracy comparison
ax1 = axes[0]
clf_names = [r['Classifier'] for r in results]
clf_accs = [r['Test Accuracy'] for r in results]
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(clf_names)))
bars = ax1.barh(clf_names, clf_accs, color=colors, edgecolor='black')
ax1.set_xlabel('Test Accuracy')
ax1.set_title('Classifier Accuracy Comparison')
ax1.set_xlim(0, 1)
for bar, acc in zip(bars, clf_accs):
    ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{acc:.3f}', va='center', fontsize=9)

# Bar plot for training time
ax2 = axes[1]
train_times = [r['Training Time (s)'] for r in results]
bars2 = ax2.barh(clf_names, train_times, color=colors, edgecolor='black')
ax2.set_xlabel('Training Time (seconds)')
ax2.set_title('Classifier Training Time')
for bar, t in zip(bars2, train_times):
    ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
             f'{t:.1f}s', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('classifier_benchmark.png', dpi=150)
print("\nSaved classifier benchmark to 'classifier_benchmark.png'")


# ============================================================
# TASK 1.3: Classification Performance Metrics
# ============================================================
print("\n" + "="*70)
print("TASK 1.3: Classification Performance Metrics")
print("="*70)

metrics_results = []
for name, config in classifiers.items():
    if "poly_degree" in config:
        poly = PolynomialFeatures(degree=config["poly_degree"], include_bias=False)
        X_te = poly.fit_transform(X_train)  # fit on train
        X_te = poly.transform(X_test)
    else:
        X_te = X_test
    
    model = trained_models[name]
    y_pred = model.predict(X_te)
    
    metrics_results.append({
        "Classifier": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision (macro)": round(precision_score(y_test, y_pred, average='macro', zero_division=0), 4),
        "Recall (macro)": round(recall_score(y_test, y_pred, average='macro', zero_division=0), 4),
        "F1 (macro)": round(f1_score(y_test, y_pred, average='macro', zero_division=0), 4)
    })

metrics_df = pd.DataFrame(metrics_results)
print("\nPerformance Metrics Comparison:")
print(metrics_df.to_string(index=False))

# Linearity Analysis
linear_acc = metrics_df[metrics_df['Classifier'] == 'SVM (Linear)']['Accuracy'].values[0]
rbf_acc = metrics_df[metrics_df['Classifier'] == 'SVM (RBF Kernel)']['Accuracy'].values[0]
logreg_acc = metrics_df[metrics_df['Classifier'] == 'Logistic Regression']['Accuracy'].values[0]
poly_logreg_acc = metrics_df[metrics_df['Classifier'] == 'LogReg + Poly(degree=2)']['Accuracy'].values[0]

print("\n" + "-"*70)
print("LINEARITY ANALYSIS")
print("-"*70)
print(f"Linear SVM Accuracy: {linear_acc:.4f}")
print(f"RBF SVM Accuracy: {rbf_acc:.4f}")
print(f"Logistic Regression Accuracy: {logreg_acc:.4f}")
print(f"Polynomial LogReg Accuracy: {poly_logreg_acc:.4f}")

improvement = rbf_acc - linear_acc
if improvement > 0.05:
    print(f"\nConclusion: Dataset appears HIGHLY NON-LINEAR")
    print(f"  - Kernel SVM improves by {improvement:.2%} over linear SVM")
elif improvement > 0.02:
    print(f"\nConclusion: Dataset has MODERATE NON-LINEARITY with some outliers")
    print(f"  - Kernel methods provide {improvement:.2%} improvement")
else:
    print(f"\nConclusion: Dataset is MOSTLY LINEAR with some outliers")
    print(f"  - Linear and kernel methods perform similarly (diff: {improvement:.2%})")


# ============================================================
# TASK 2.1: PCA Analysis
# ============================================================
print("\n" + "="*70)
print("TASK 2.1: PCA - Dimensionality Reduction")
print("="*70)

# Full PCA to analyze variance
pca_full = PCA()
pca_full.fit(X_scaled)

# Explained variance analysis
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
n_components_99 = np.argmax(cumulative_variance >= 0.99) + 1

print(f"\nOriginal dimensions: {X_scaled.shape[1]}")
print(f"Components for 95% variance: {n_components_95}")
print(f"Components for 99% variance: {n_components_99}")

# Plot explained variance
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(range(1, min(51, len(cumulative_variance)+1)), 
             cumulative_variance[:50], 'b-o', markersize=3)
axes[0].axhline(y=0.95, color='r', linestyle='--', label='95% variance')
axes[0].axhline(y=0.99, color='g', linestyle='--', label='99% variance')
axes[0].set_xlabel('Number of Components')
axes[0].set_ylabel('Cumulative Explained Variance')
axes[0].set_title('PCA: Cumulative Explained Variance')
axes[0].legend()
axes[0].grid(True)

# Reconstruction error analysis
n_components_list = [5, 10, 20, 50, 100, n_components_95, n_components_99]
n_components_list = sorted(set([n for n in n_components_list if n <= X_scaled.shape[1]]))
reconstruction_errors = []

for n in n_components_list:
    pca_temp = PCA(n_components=n)
    X_reduced = pca_temp.fit_transform(X_scaled)
    X_reconstructed = pca_temp.inverse_transform(X_reduced)
    error = np.mean((X_scaled - X_reconstructed) ** 2)
    reconstruction_errors.append(error)

axes[1].plot(n_components_list, reconstruction_errors, 'r-o')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Mean Squared Reconstruction Error')
axes[1].set_title('PCA: Reconstruction Error')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('pca_analysis.png', dpi=150)
print("Saved PCA analysis to 'pca_analysis.png'")

# Repeat Task 1.1 with reduced dimensions
print("\n" + "-"*70)
print(f"Repeating classification with PCA ({n_components_95} components for 95% variance)")
print("-"*70)

pca = PCA(n_components=n_components_95)
X_pca = pca.fit_transform(X_scaled)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)

pca_results = []
for name in ["Logistic Regression", "SVM (Linear)", "SVM (RBF Kernel)", "k-NN", "Random Forest"]:
    config = classifiers[name]
    start_time = time.time()
    grid_search = GridSearchCV(config["model"], config["params"], cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_pca, y_train_pca)
    train_time = time.time() - start_time
    
    y_pred = grid_search.predict(X_test_pca)
    pca_results.append({
        "Classifier": name,
        "Original Acc": results_df[results_df['Classifier']==name]['Test Accuracy'].values[0],
        "PCA Acc": round(accuracy_score(y_test_pca, y_pred), 4),
        "PCA Train Time (s)": round(train_time, 4)
    })

pca_df = pd.DataFrame(pca_results)
print(pca_df.to_string(index=False))


# ============================================================
# TASK 2.2: Clustering 
# ============================================================
print("\n" + "="*70)
print("TASK 2.2: Clustering Analysis")
print("="*70)

# Use PCA-reduced features for clustering
X_cluster = X_pca

# K-Means clustering
n_clusters = len(np.unique(y))
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_cluster)

# DBSCAN clustering
dbscan = DBSCAN(eps=2.0, min_samples=3)
dbscan_labels = dbscan.fit_predict(X_cluster)

print(f"\nNumber of true classes: {n_clusters}")
print(f"K-Means clusters: {n_clusters}")
print(f"DBSCAN clusters found: {len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)}")
print(f"DBSCAN noise points: {np.sum(dbscan_labels == -1)}")

# External metrics (comparing with true labels)
print("\n--- External Metrics (vs True Labels) ---")
print(f"K-Means - Adjusted Rand Index: {adjusted_rand_score(y, kmeans_labels):.4f}")
print(f"K-Means - Normalized Mutual Info: {normalized_mutual_info_score(y, kmeans_labels):.4f}")

# Filter out noise for DBSCAN metrics
dbscan_mask = dbscan_labels != -1
if np.sum(dbscan_mask) > 0:
    print(f"DBSCAN - Adjusted Rand Index: {adjusted_rand_score(y[dbscan_mask], dbscan_labels[dbscan_mask]):.4f}")
    print(f"DBSCAN - Normalized Mutual Info: {normalized_mutual_info_score(y[dbscan_mask], dbscan_labels[dbscan_mask]):.4f}")

# Internal metrics
print("\n--- Internal Metrics ---")
print(f"K-Means - Silhouette Score: {silhouette_score(X_cluster, kmeans_labels):.4f}")
print(f"K-Means - Inertia: {kmeans.inertia_:.4f}")
if len(set(dbscan_labels[dbscan_mask])) > 1:
    print(f"DBSCAN - Silhouette Score: {silhouette_score(X_cluster[dbscan_mask], dbscan_labels[dbscan_mask]):.4f}")

# Visualize clusters (2D projection)
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_cluster)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# True labels
scatter = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=pd.factorize(y)[0], cmap='tab10', alpha=0.6, s=20)
axes[0].set_title('True Labels')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')

# K-Means
axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=kmeans_labels, cmap='tab10', alpha=0.6, s=20)
axes[1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                c='red', marker='X', s=200, edgecolors='black', label='Centroids')
axes[1].set_title('K-Means Clustering')
axes[1].set_xlabel('PC1')
axes[1].legend()

# DBSCAN
axes[2].scatter(X_2d[:, 0], X_2d[:, 1], c=dbscan_labels, cmap='tab10', alpha=0.6, s=20)
axes[2].set_title(f'DBSCAN (noise={np.sum(dbscan_labels==-1)})')
axes[2].set_xlabel('PC1')

plt.tight_layout()
plt.savefig('clustering_analysis.png', dpi=150)
print("\nSaved clustering visualization to 'clustering_analysis.png'")


# ============================================================
# TASK 1.4 & 2.3: Outlier Detection Frameworks
# ============================================================
print("\n" + "="*70)
print("TASK 1.4 & 2.3: Outlier Detection Frameworks")
print("="*70)

# --- SVM-based Outlier Detection using SVM Constraints (Task 1.4) ---
print("\n--- SVM-based Outlier Detection Framework (using SVM constraints) ---")
print("""
PROPOSED FRAMEWORK (using SVM margin constraints):
1. Train a soft-margin SVM classifier on the dataset
2. For each data point, check its position relative to the margin:
   - Points with slack variable ξ > 0 violate the margin constraint
   - Points with ξ > 1 are misclassified (on wrong side of hyperplane)
3. Outliers are identified as points with HIGH slack values
   - These are points that the SVM "struggles" to classify
   - They either lie within the margin or on the wrong side

Key idea: In soft-margin SVM, the constraint is: y_i(w·x_i + b) >= 1 - ξ_i
Points with large ξ values are "problematic" - likely outliers or noise.

We use the decision function distance to estimate slack:
   slack ≈ max(0, 1 - y_i * decision_function(x_i))
""")

# Train SVM classifier
svm_clf = SVC(kernel='linear', C=1.0)
svm_clf.fit(X_train, y_train)

# For multi-class, we use decision function distance as proxy for "outlierness"
# Points close to decision boundary (low confidence) are potential outliers
decision_values = svm_clf.decision_function(X_scaled)

# For multi-class SVM, decision_function returns distances to each hyperplane
# We use the margin (minimum distance to any decision boundary) as outlier score
if len(decision_values.shape) > 1:
    # Multi-class: use minimum absolute distance across all class boundaries
    margin_distances = np.min(np.abs(decision_values), axis=1)
else:
    margin_distances = np.abs(decision_values)

# Compute slack-like score: points with small margin distance have high slack
# Outliers = points in the bottom percentile of margin distances
outlier_threshold_5 = np.percentile(margin_distances, 5)
outlier_threshold_10 = np.percentile(margin_distances, 10)

svm_outliers_5 = np.where(margin_distances <= outlier_threshold_5)[0]
svm_outliers_10 = np.where(margin_distances <= outlier_threshold_10)[0]

print(f"\nMargin distance statistics:")
print(f"  Min: {margin_distances.min():.4f}, Max: {margin_distances.max():.4f}")
print(f"  Mean: {margin_distances.mean():.4f}, Std: {margin_distances.std():.4f}")
print(f"\nOutliers (bottom 5% margin distance): {len(svm_outliers_5)} points")
print(f"Outliers (bottom 10% margin distance): {len(svm_outliers_10)} points")

# Use 5% as default
svm_outliers = svm_outliers_5
print(f"\nSVM Outlier indices: {svm_outliers[:20]}..." if len(svm_outliers) > 20 else f"SVM Outlier indices: {svm_outliers}")

# --- Clustering-based Outlier Detection (Task 2.3) ---
print("\n--- Clustering-based Outlier Detection Framework ---")
print("""
PROPOSED FRAMEWORK:
1. Apply K-Means clustering to the data
2. Compute distance of each point to its assigned cluster centroid
3. Points with distances > mean + 2*std are flagged as outliers
4. Alternative: Use DBSCAN where noise points (label=-1) are outliers

Key idea: Outliers are points that don't fit well into any cluster,
indicated by large distance to nearest centroid.
""")

# Method 1: K-Means distance-based
kmeans_full = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans_full.fit(X_pca)
distances = np.min(cdist(X_pca, kmeans_full.cluster_centers_), axis=1)

threshold = np.mean(distances) + 2 * np.std(distances)
kmeans_outliers = np.where(distances > threshold)[0]

print(f"\nK-Means distance threshold: {threshold:.4f}")
print(f"K-Means outliers detected: {len(kmeans_outliers)} ({100*len(kmeans_outliers)/len(y):.1f}%)")

# Method 2: DBSCAN noise points
dbscan_outliers = np.where(dbscan_labels == -1)[0]
print(f"DBSCAN outliers (noise): {len(dbscan_outliers)} ({100*len(dbscan_outliers)/len(y):.1f}%)")

# Compare SVM and Clustering outliers
common_outliers = np.intersect1d(svm_outliers, kmeans_outliers)
print(f"\nOutliers detected by BOTH SVM and K-Means: {len(common_outliers)}")
if max(len(svm_outliers), len(kmeans_outliers)) > 0:
    print(f"Agreement rate: {100*len(common_outliers)/max(len(svm_outliers), len(kmeans_outliers)):.1f}%")


# Visualize outliers
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# SVM outliers (margin-based)
axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c='blue', alpha=0.3, s=20, label='Normal')
axes[0].scatter(X_2d[svm_outliers, 0], X_2d[svm_outliers, 1], 
                c='red', s=50, marker='x', label='Outliers')
axes[0].set_title(f'SVM Margin-based (n={len(svm_outliers)})')
axes[0].legend()

# K-Means outliers
axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c='blue', alpha=0.3, s=20, label='Normal')
axes[1].scatter(X_2d[kmeans_outliers, 0], X_2d[kmeans_outliers, 1], 
                c='red', s=50, marker='x', label='Outliers')
axes[1].set_title(f'K-Means Outliers (n={len(kmeans_outliers)})')
axes[1].legend()

# DBSCAN outliers
axes[2].scatter(X_2d[:, 0], X_2d[:, 1], c='blue', alpha=0.3, s=20, label='Normal')
axes[2].scatter(X_2d[dbscan_outliers, 0], X_2d[dbscan_outliers, 1], 
                c='red', s=50, marker='x', label='Outliers')
axes[2].set_title(f'DBSCAN Outliers (n={len(dbscan_outliers)})')
axes[2].legend()

plt.tight_layout()
plt.savefig('outlier_detection.png', dpi=150)
print("\nSaved outlier visualization to 'outlier_detection.png'")

# ============================================================
# Summary
# ============================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Best Feature Set: {best_feature_set}")
print(f"Early Fusion Accuracy: {early_acc:.4f} | Late Fusion Accuracy: {late_acc:.4f}")
best_clf = results_df.loc[results_df['Test Accuracy'].idxmax()]
print(f"Best Classifier: {best_clf['Classifier']} (Accuracy: {best_clf['Test Accuracy']})")
print(f"PCA reduced dimensions: {X_best.shape[1]} -> {n_components_95} (95% variance)")
print(f"K-Means Silhouette Score: {silhouette_score(X_cluster, kmeans_labels):.4f}")
print(f"Outliers detected - SVM: {len(svm_outliers)}, K-Means: {len(kmeans_outliers)}, DBSCAN: {len(dbscan_outliers)}")

print("\n" + "="*70)
print("Generated files:")
print("  - output.txt (this output)")
print("  - feature_comparison.png")
print("  - classifier_benchmark.png")
print("  - pca_analysis.png")
print("  - clustering_analysis.png") 
print("  - outlier_detection.png")
print("="*70)

# Close logger
sys.stdout = logger.terminal
logger.close()
print("Output saved to 'output.txt'")
