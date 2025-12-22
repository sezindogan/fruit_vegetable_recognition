"""
CMPE 462 - Assignment 2, Task 1.2
Soft-Margin Linear SVM from Scratch using QP Solver

Implementation follows the lecture slides exactly:
- Primal formulation with slack variables
- Constructs Q, p, A, c matrices as shown in slides
- Uses cvxopt QP solver
- Finds support vectors and analyzes them
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from scipy.spatial.distance import cdist
import sys

# Suppress cvxopt output
solvers.options['show_progress'] = False

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
logger = OutputLogger('svm_output.txt')
sys.stdout = logger

class SoftMarginSVMScratch:
    """
    Soft-Margin Linear SVM from Scratch using QP Solver.
    
    Primal Problem (from slides):
    min_(b,w,ε) 1/2 w^T w + C Σε_n
    subject to: y_n(w^T x_n + b) ≥ 1 - ε_n, ∀n
                ε_n ≥ 0, ∀n
    
    QP Formulation:
    min_u 1/2 u^T Q u + p^T u
    subject to: A u ≥ c
    
    where u = [b, w, ε]^T ∈ R^(d+1+N)
    """
    
    def __init__(self, C=1.0):
        """
        Parameters:
        -----------
        C : float
            Regularization parameter. Large C = care more about margin violations
            (closer to hard-margin). Small C = care less about violations.
        """
        self.C = C
        self.w = None
        self.b = None
        self.support_vectors = None
        self.support_vector_indices = None
        self.support_vector_labels = None
        self.slack_variables = None
        
    def fit(self, X, y):
        """
        Train the soft-margin SVM using QP solver.
        
        Parameters:
        -----------
        X : ndarray, shape (N, d)
            Training data
        y : ndarray, shape (N,)
            Labels (+1 or -1)
        """
        N, d = X.shape
        
        # Convert labels to {-1, +1} if not already
        unique_labels = np.unique(y)
        if set(unique_labels) == {-1.0, 1.0} or set(unique_labels) == {-1, 1}:
            y_binary = y.astype(float)
        else:
            # Map: first unique label -> -1, second -> +1
            y_binary = np.where(y == unique_labels[0], -1, 1).astype(float)
        
        print(f"\n{'='*70}")
        print("SOFT-MARGIN LINEAR SVM FROM SCRATCH")
        print(f"{'='*70}")
        print(f"Training samples: N = {N}")
        print(f"Feature dimension: d = {d}")
        print(f"Regularization parameter: C = {self.C}")
        
        # =====================================================================
        # Construct QP Problem: min 1/2 u^T Q u + p^T u, subject to A u ≥ c
        # u = [b, w_1, ..., w_d, ε_1, ..., ε_N]^T  (dimension: 1 + d + N)
        # =====================================================================
        
        # --- Construct Q matrix (from slides) ---
        # Q has structure:
        #     [0    0^T_d      0^T_N  ]
        #     [0_d  I_d       0_{d×N}]
        #     [0_N  0_{N×d}   0_{N×N}]
        # This corresponds to: 1/2 w^T w (only w terms contribute)
        
        Q = np.zeros((1 + d + N, 1 + d + N))
        Q[1:d+1, 1:d+1] = np.eye(d)  # I_d for w^T w term
        
        # --- Construct p vector (from slides) ---
        # p = [0, 0, ..., 0, C, C, ..., C]^T
        # First (1+d) entries are 0, last N entries are C (penalty for slack)
        p = np.zeros(1 + d + N)
        p[1+d:] = self.C  # C for each ε_n
        
        # --- Construct constraint matrix A and vector c ---
        # We have two sets of constraints:
        # 1) y_n(w^T x_n + b) ≥ 1 - ε_n  =>  y_n b + y_n w^T x_n + ε_n ≥ 1
        #    Rewrite as: [y_n, y_n x_n^T, e_n^T] u ≥ 1
        #    where e_n is the n-th standard basis vector in R^N
        # 2) ε_n ≥ 0  =>  [0, 0^T, e_n^T] u ≥ 0
        
        # Total constraints: N (margin constraints) + N (non-negativity) = 2N
        A = np.zeros((2*N, 1 + d + N))
        c = np.zeros(2*N)
        
        # First N rows: margin constraints y_n(w^T x_n + b) ≥ 1 - ε_n
        for n in range(N):
            A[n, 0] = y_binary[n]           # coefficient for b
            A[n, 1:d+1] = y_binary[n] * X[n]  # coefficients for w
            A[n, 1+d+n] = 1.0               # coefficient for ε_n
            c[n] = 1.0
        
        # Next N rows: non-negativity constraints ε_n ≥ 0
        for n in range(N):
            A[N+n, 1+d+n] = 1.0  # ε_n ≥ 0
            c[N+n] = 0.0
        
        print(f"\nQP Problem dimensions:")
        print(f"  Q: {Q.shape} (objective quadratic term)")
        print(f"  p: {p.shape} (objective linear term)")
        print(f"  A: {A.shape} (constraint matrix, {2*N} constraints)")
        print(f"  c: {c.shape} (constraint vector)")
        
        # =====================================================================
        # Solve QP Problem using cvxopt
        # =====================================================================
        
        print(f"\nSolving QP problem with cvxopt...")
        
        # Convert to cvxopt matrix format
        Q_cvx = matrix(Q)
        p_cvx = matrix(p)
        # cvxopt uses G u <= h, but we have A u >= c
        # So we use: -A u <= -c
        G_cvx = matrix(-A)
        h_cvx = matrix(-c)
        
        # Solve
        sol = solvers.qp(Q_cvx, p_cvx, G_cvx, h_cvx)
        
        if sol['status'] != 'optimal':
            print(f"Warning: QP solver status: {sol['status']}")
        
        # Extract solution
        u_opt = np.array(sol['x']).flatten()
        
        self.b = u_opt[0]
        self.w = u_opt[1:d+1]
        self.slack_variables = u_opt[1+d:]
        
        print(f"\nOptimization completed!")
        print(f"  Status: {sol['status']}")
        print(f"  Optimal value: {sol['primal objective']:.6f}")
        
        # =====================================================================
        # Find Support Vectors
        # =====================================================================
        
        # Support vectors are points where:
        # - α_n > 0 (in dual formulation), or equivalently
        # - Points that are on the margin or violate it
        # - Points where y_n(w^T x_n + b) ≤ 1 (with some tolerance)
        
        margins = y_binary * (X @ self.w + self.b)
        
        # Support vectors: points on margin (margin = 1) or with slack (margin < 1)
        # Use tolerance for numerical stability
        tolerance = 1e-4
        sv_mask = (margins <= 1 + tolerance) | (self.slack_variables > tolerance)
        
        self.support_vector_indices = np.where(sv_mask)[0]
        self.support_vectors = X[sv_mask]
        self.support_vector_labels = y[sv_mask]
        
        print(f"\n{'='*70}")
        print("SOLUTION SUMMARY")
        print(f"{'='*70}")
        print(f"Bias term: b* = {self.b:.6f}")
        print(f"Weight vector: w* = {self.w}")
        print(f"  ||w*|| = {np.linalg.norm(self.w):.6f}")
        print(f"  Margin = 1/||w*|| = {1/np.linalg.norm(self.w):.6f}")
        
        print(f"\nSlack variables (ε):")
        print(f"  Total violation: Σε = {np.sum(self.slack_variables):.6f}")
        print(f"  Max slack: {np.max(self.slack_variables):.6f}")
        print(f"  Points with slack > 0: {np.sum(self.slack_variables > tolerance)}")
        
        print(f"\nSupport Vectors:")
        print(f"  Total support vectors: {len(self.support_vector_indices)}")
        print(f"  Percentage: {100*len(self.support_vector_indices)/N:.1f}%")
        print(f"  Indices: {self.support_vector_indices[:20]}{'...' if len(self.support_vector_indices) > 20 else ''}")
        
        # Find points on exact margin (y(w^T x + b) = 1)
        on_margin = np.abs(margins - 1) < tolerance
        print(f"  Points exactly on margin: {np.sum(on_margin)}")
        
        # Find misclassified points (y(w^T x + b) < 0)
        misclassified = margins < 0
        print(f"  Misclassified points: {np.sum(misclassified)}")
        
        return self
    
    def predict(self, X):
        """Predict class labels."""
        return np.sign(X @ self.w + self.b)
    
    def decision_function(self, X):
        """Compute decision function values."""
        return X @ self.w + self.b
    
    def get_farthest_points(self, X, y):
        """
        Find the data points that are farthest from the hyperplane in each category.
        
        Returns:
        --------
        farthest_indices : dict
            Dictionary mapping class label to index of farthest point
        farthest_distances : dict
            Dictionary mapping class label to distance of farthest point
        """
        # Distance to hyperplane: |w^T x + b| / ||w||
        distances = np.abs(self.decision_function(X)) / np.linalg.norm(self.w)
        
        farthest_indices = {}
        farthest_distances = {}
        
        for label in np.unique(y):
            mask = (y == label)
            class_distances = distances[mask]
            class_indices = np.where(mask)[0]
            
            farthest_idx_in_class = np.argmax(class_distances)
            farthest_indices[label] = class_indices[farthest_idx_in_class]
            farthest_distances[label] = class_distances[farthest_idx_in_class]
        
        return farthest_indices, farthest_distances


# ============================================================================
# LOAD DATA AND TRAIN SVM
# ============================================================================
from itertools import combinations
from sklearn.preprocessing import StandardScaler

print("Loading data...")
y = pd.read_csv("labels.csv", index_col=0).to_numpy().flatten()
X_image = pd.read_csv("image.csv", index_col=0).to_numpy()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_image)

print(f"Total samples: {len(X_scaled)}")
all_classes = np.unique(y)
print(f"All classes ({len(all_classes)}): {all_classes}")

# ============================================================================
# TRAIN SVM FOR ALL CLASS PAIRS
# ============================================================================

class_pairs = list(combinations(all_classes, 2))
print(f"\nTotal class pairs to compare: {len(class_pairs)}")

# Store results for all pairs
all_pair_results = []
all_pair_svms = {}

C_default = 1.0  # Use C=1.0 for all pair comparisons

for pair_idx, (class_a, class_b) in enumerate(class_pairs):
    print(f"\n{'#'*70}")
    print(f"CLASS PAIR {pair_idx + 1}/{len(class_pairs)}: {class_a} vs {class_b}")
    print(f"{'#'*70}")
    
    # Filter data for this pair
    binary_mask = np.isin(y, [class_a, class_b])
    X_pair = X_scaled[binary_mask]
    y_pair = y[binary_mask]
    
    # Create numeric labels: class_a -> +1, class_b -> -1
    y_pair_numeric = np.where(y_pair == class_a, 1.0, -1.0)
    
    print(f"  Class +1: {class_a} ({np.sum(y_pair == class_a)} samples)")
    print(f"  Class -1: {class_b} ({np.sum(y_pair == class_b)} samples)")
    
    # Train SVM
    svm = SoftMarginSVMScratch(C=C_default)
    svm.fit(X_pair, y_pair_numeric)
    
    # Compute accuracy
    y_pred = svm.predict(X_pair)
    accuracy = np.mean(y_pred == y_pair_numeric)
    
    # Store results
    all_pair_results.append({
        'Class_A': class_a,
        'Class_B': class_b,
        'Samples_A': np.sum(y_pair == class_a),
        'Samples_B': np.sum(y_pair == class_b),
        'Num_SV': len(svm.support_vector_indices),
        'SV_Percent': 100 * len(svm.support_vector_indices) / len(y_pair),
        'Total_Slack': np.sum(svm.slack_variables),
        'Margin': 1 / np.linalg.norm(svm.w),
        'Accuracy': accuracy
    })
    
    all_pair_svms[(class_a, class_b)] = {
        'svm': svm,
        'X': X_pair,
        'y': y_pair,
        'y_numeric': y_pair_numeric
    }

# Summary table
pair_results_df = pd.DataFrame(all_pair_results)
print(f"\n{'='*70}")
print("SUMMARY: ALL CLASS PAIR COMPARISONS (C=1.0)")
print(f"{'='*70}")
print(pair_results_df.to_string(index=False))

# Find easiest and hardest pairs to separate
print(f"\n{'='*70}")
print("ANALYSIS: EASIEST AND HARDEST CLASS PAIRS")
print(f"{'='*70}")

sorted_by_accuracy = pair_results_df.sort_values('Accuracy', ascending=False)
print("\nTop 5 EASIEST pairs to separate (highest accuracy):")
print(sorted_by_accuracy.head(5)[['Class_A', 'Class_B', 'Accuracy', 'Margin', 'Num_SV']].to_string(index=False))

print("\nTop 5 HARDEST pairs to separate (lowest accuracy):")
print(sorted_by_accuracy.tail(5)[['Class_A', 'Class_B', 'Accuracy', 'Margin', 'Num_SV']].to_string(index=False))

sorted_by_margin = pair_results_df.sort_values('Margin', ascending=False)
print("\nTop 5 pairs with LARGEST margin:")
print(sorted_by_margin.head(5)[['Class_A', 'Class_B', 'Margin', 'Accuracy', 'SV_Percent']].to_string(index=False))

print("\nTop 5 pairs with SMALLEST margin:")
print(sorted_by_margin.tail(5)[['Class_A', 'Class_B', 'Margin', 'Accuracy', 'SV_Percent']].to_string(index=False))

# ============================================================================
# DETAILED ANALYSIS FOR TOP 2 CLASSES (for visualization)
# ============================================================================

print(f"\n{'='*70}")
print("DETAILED ANALYSIS: TOP 2 CLASSES BY FREQUENCY")
print(f"{'='*70}")

class_counts = pd.Series(y).value_counts()
top_2_classes = class_counts.head(2).index.tolist()
pos_class, neg_class = top_2_classes[0], top_2_classes[1]

# Get the SVM for this pair
pair_key = (pos_class, neg_class) if (pos_class, neg_class) in all_pair_svms else (neg_class, pos_class)
pair_data = all_pair_svms[pair_key]
svm_main = pair_data['svm']
X_binary = pair_data['X']
y_binary = pair_data['y']
y_binary_numeric = pair_data['y_numeric']

# Also run C value comparison for this pair
print(f"\nC-value comparison for {pos_class} vs {neg_class}:")
C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
results = []
svms = {}

for C in C_values:
    svm = SoftMarginSVMScratch(C=C)
    svm.fit(X_binary, y_binary_numeric)
    y_pred = svm.predict(X_binary)
    accuracy = np.mean(y_pred == y_binary_numeric)
    
    results.append({
        'C': C,
        'Num_SV': len(svm.support_vector_indices),
        'Total_Slack': np.sum(svm.slack_variables),
        'Margin': 1/np.linalg.norm(svm.w),
        'Accuracy': accuracy
    })
    svms[C] = svm

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))


# ============================================================================
# TASK 1.2(a): Visual Inspection of Support Vectors
# ============================================================================

print(f"\n{'='*70}")
print("TASK 1.2(a): Visual Inspection of Support Vectors")
print(f"{'='*70}")

# Use SVM with C=1.0 for detailed analysis
svm_main = svms[1.0]

# Get farthest points
farthest_indices, farthest_distances = svm_main.get_farthest_points(X_binary, y_binary_numeric)

print("\nFarthest points from hyperplane in each category:")
for label, idx in farthest_indices.items():
    class_name = pos_class if label == 1.0 else neg_class
    print(f"  Class {class_name} ({label}): Index {idx}, Distance = {farthest_distances[label]:.4f}")

# Use PCA to visualize in 2D
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_binary)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Support vectors vs. all points
ax1 = axes[0]
colors = ['red' if y_binary_numeric[i] == 1.0 else 'blue' for i in range(len(y_binary_numeric))]

# 1. TRAINING POINTS: Slightly smaller (s=20) to separate them better
ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, alpha=0.3, s=20, label='Training points')

# Highlight support vectors
sv_2d = X_2d[svm_main.support_vector_indices]

# 2. SUPPORT VECTORS
ax1.scatter(sv_2d[:, 0], sv_2d[:, 1], 
            facecolors='none', 
            edgecolors='black', 
            s=80,             # Much smaller
            linewidths=0.6,   # Much thinner
            alpha=0.6,        # Semi-transparent edges
            label=f'Support Vectors (n={len(svm_main.support_vector_indices)})')

# Highlight farthest points
for label, idx in farthest_indices.items():
    class_name = pos_class if label == 1.0 else neg_class
    # 3. FARTHEST POINTS
    ax1.scatter(X_2d[idx, 0], X_2d[idx, 1], 
                marker='*', s=200, c='gold', edgecolors='black', linewidths=1.5,
                label=f'Farthest ({class_name})')

ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_title(f'Support Vectors vs Farthest Points ({pos_class} vs {neg_class})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Slack variables
ax2 = axes[1]
slack_sorted_idx = np.argsort(svm_main.slack_variables)[::-1]
ax2.bar(range(len(svm_main.slack_variables)), 
        svm_main.slack_variables[slack_sorted_idx],
        color='coral', edgecolor='black', alpha=0.7)
ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='ε=1 (misclassified)')
ax2.set_xlabel('Data Point (sorted by slack)')
ax2.set_ylabel('Slack Variable ε')
ax2.set_title('Margin Violations (Slack Variables)')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('svm_scratch_analysis.png', dpi=150)
print("\nSaved visualization to 'svm_scratch_analysis.png'")


# ============================================================================
# TASK 1.2(b): Support Vector Similarity Analysis
# ============================================================================

print(f"\n{'='*70}")
print("TASK 1.2(b): Cross-Class Support Vector Similarity")
print(f"{'='*70}")

# Get support vectors by class
sv_class1_mask = y_binary_numeric[svm_main.support_vector_indices] == 1.0
sv_class2_mask = y_binary_numeric[svm_main.support_vector_indices] == -1.0

sv_class1 = svm_main.support_vectors[sv_class1_mask]
sv_class2 = svm_main.support_vectors[sv_class2_mask]

print(f"\nSupport vectors per class:")
print(f"  Class {pos_class} (+1): {len(sv_class1)} support vectors")
print(f"  Class {neg_class} (-1): {len(sv_class2)} support vectors")

if len(sv_class1) > 0 and len(sv_class2) > 0:
    # Compute pairwise distances between support vectors across classes
    cross_class_distances = cdist(sv_class1, sv_class2, metric='euclidean')
    
    print(f"\nCross-class support vector distances:")
    print(f"  Min distance: {cross_class_distances.min():.4f}")
    print(f"  Mean distance: {cross_class_distances.mean():.4f}")
    print(f"  Max distance: {cross_class_distances.max():.4f}")
    
    # Find closest pairs
    min_dist_idx = np.unravel_index(cross_class_distances.argmin(), 
                                     cross_class_distances.shape)
    print(f"\n  Closest SV pair:")
    print(f"    Class {pos_class} SV index: {min_dist_idx[0]}")
    print(f"    Class {neg_class} SV index: {min_dist_idx[1]}")
    print(f"    Distance: {cross_class_distances[min_dist_idx]:.4f}")
    
    # Compute confusion matrix to check if closest SVs correspond to confused classes
    from sklearn.metrics import confusion_matrix
    y_pred = svm_main.predict(X_binary)
    cm = confusion_matrix(y_binary_numeric, y_pred)
    
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"  Off-diagonal confusion: {cm[0,1] + cm[1,0]} misclassifications")
    
    print(f"\nInterpretation:")
    print(f"  The support vectors closest across classes are the 'boundary points'")
    print(f"  that define the margin. These points are the most difficult to separate")
    print(f"  and are often the ones that get confused when the classifier makes errors.")

print(f"\n{'='*70}")
print("SVM FROM SCRATCH IMPLEMENTATION COMPLETE")
print(f"{'='*70}")
print(f"\nOutput saved to 'svm_output.txt'")

# Close logger
sys.stdout = logger.terminal
logger.close()
