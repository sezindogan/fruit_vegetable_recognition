# Task 1 – Part 1: Benchmarking Classifiers
# Assumes X (features) and y (labels) are already loaded

import time
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier


y = pd.read_csv("labels.csv", index_col=0).to_numpy().flatten()
X = pd.read_csv("image.csv", index_col=0).to_numpy()  # Using combined features for benchmarking

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define classifiers and hyperparameter grids for tuning 
classifiers = {
    "Logistic Regression": (LogisticRegression(max_iter=1000), {
        'C': [10**i for i in range(-5,5)] # Regularization parameter 
    }),
    "Soft-margin SVM (Linear)": (SVC(kernel='linear'), {
        'C': [10**i for i in range(-5,5)] 
    }),
    "Soft-margin SVM (Kernel)": (SVC(), {
        'kernel': ['rbf', 'poly'],
        'C': [10**i for i in range(-5,5)],
        'gamma': ['scale', 'auto']
    }),
    "k-NN": (KNeighborsClassifier(), {
        'n_neighbors': [2*i+1 for i in range(10)],  # Odd values from 1 to 19
        'weights': ['uniform', 'distance']
    }),
    "Naive Bayes": (GaussianNB(), {}),
    "Random Forest": (RandomForestClassifier(), {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20]
    })
}

results = []

for name, (model, params) in classifiers.items():
    start_time = time.time()
    
    # GridSearch handles hyperparameter determination 
    clf = GridSearchCV(model, params, cv=5)
    clf.fit(X_train, y_train)
    
    end_time = time.time()
    train_time = end_time - start_time # Report training times 
    
    # Evaluate [cite: 23]
    score = clf.score(X_test, y_test)
    results.append({
        "Classifier": name,
        "Best Params": clf.best_params_,
        "Training Time (s)": round(train_time, 4),
        "Test Accuracy": round(score, 4)
    })

# Logistic Regression with Non-linear Transformation [cite: 11]
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

log_reg_poly = LogisticRegression(max_iter=1000)
param_grid_poly = {'C': [10**i for i in range(-5,5)]}

start_time = time.time()
clf_poly = GridSearchCV(log_reg_poly, param_grid_poly, cv=5)
clf_poly.fit(X_train_poly, y_train)
train_time_poly = time.time() - start_time

results.append({
    "Classifier": "LogReg (Non-linear Poly)",
    "Best Params": clf_poly.best_params_,
    "Training Time (s)": round(train_time_poly, 4),
    "Test Accuracy": round(clf_poly.score(X_test_poly, y_test), 4)
})

# 2. Present results in a table 
results_df = pd.DataFrame(results)
print(results_df)
