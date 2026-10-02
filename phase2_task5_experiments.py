# ENGE707 Phase II
# Task 5 - Experiments and Statistical Analysis

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# Reproducibility
RANDOM_STATE = 42

print("=" * 60)
print("TASK 5 - EXPERIMENTS AND STATISTICAL ANALYSIS")
print("=" * 60)
print(f"Random state: {RANDOM_STATE}")
# ------------------------------------------------------------
# 1. LOAD PREPROCESSED TRAIN/TEST DATA FROM TASK 3
# ------------------------------------------------------------

TARGET = "hospdead"

train_data = pd.read_csv(
    "data/processed/support2_train_preprocessed.csv"
)

test_data = pd.read_csv(
    "data/processed/support2_test_preprocessed.csv"
)

print("\n--- Experimental Data Setup ---")
print("Training data shape:", train_data.shape)
print("Testing data shape:", test_data.shape)

# Separate predictors and target
X_train = train_data.drop(columns=[TARGET])
y_train = train_data[TARGET]

X_test = test_data.drop(columns=[TARGET])
y_test = test_data[TARGET]

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())
print(y_train.value_counts(normalize=True).mul(100).round(2))

print("\nTesting target distribution:")
print(y_test.value_counts())
print(y_test.value_counts(normalize=True).mul(100).round(2))

# Basic validation checks
assert list(X_train.columns) == list(X_test.columns), \
    "ERROR: Train and test feature columns do not match."

assert X_train.isna().sum().sum() == 0, \
    "ERROR: Missing values found in training predictors."

assert X_test.isna().sum().sum() == 0, \
    "ERROR: Missing values found in testing predictors."

print("\nTrain/test feature columns match.")
print("No missing predictor values detected.")
print("Experimental dataset is ready.")
# ------------------------------------------------------------
# 2. MODEL EVALUATION FUNCTION
# ------------------------------------------------------------

def evaluate_model(model_name, y_true, y_pred, y_proba=None):
    """
    Calculate classification metrics for a model.

    Parameters
    ----------
    model_name : str
        Name of the model.
    y_true : array-like
        Actual target values.
    y_pred : array-like
        Predicted class labels.
    y_proba : array-like, optional
        Predicted probability for class 1.
    """

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    results = {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Balanced Accuracy": balanced_accuracy,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }

    # Probability-based metrics require predicted probabilities.
    if y_proba is not None:
        results["ROC-AUC"] = roc_auc_score(y_true, y_proba)
        results["PR-AUC"] = average_precision_score(y_true, y_proba)
    else:
        results["ROC-AUC"] = np.nan
        results["PR-AUC"] = np.nan

    return results


print("\nModel evaluation function created successfully.")