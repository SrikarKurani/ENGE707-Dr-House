# ENGE707 Phase II
# Task 5 - Experiments and Statistical Analysis

import numpy as np
import pandas as pd

from scipy.stats import binomtest

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
# ------------------------------------------------------------
# 3. EXACT MCNEMAR STATISTICAL TEST
# ------------------------------------------------------------

def mcnemar_exact(y_true, pred_a, pred_b):
    """
    Perform an exact McNemar test on two classifiers evaluated
    on the same test observations.

    Returns the number of discordant prediction pairs and
    the exact two-sided p-value.
    """

    correct_a = np.asarray(pred_a) == np.asarray(y_true)
    correct_b = np.asarray(pred_b) == np.asarray(y_true)

    # Model A correct, Model B incorrect
    a_only_correct = int(np.sum(correct_a & ~correct_b))

    # Model A incorrect, Model B correct
    b_only_correct = int(np.sum(~correct_a & correct_b))

    discordant = a_only_correct + b_only_correct

    if discordant == 0:
        p_value = 1.0
    else:
        p_value = binomtest(
            min(a_only_correct, b_only_correct),
            n=discordant,
            p=0.5,
            alternative="two-sided"
        ).pvalue

    return {
        "A only correct": a_only_correct,
        "B only correct": b_only_correct,
        "Discordant pairs": discordant,
        "p-value": p_value
    }


print("Exact McNemar test function created successfully.")
# ------------------------------------------------------------
# 4. FUNCTION VALIDATION CHECK
# ------------------------------------------------------------

# Simple controlled example to verify the evaluation code
test_y_true = np.array([0, 0, 0, 1, 1, 1])
test_y_pred = np.array([0, 0, 1, 1, 0, 1])
test_y_proba = np.array([0.10, 0.20, 0.70, 0.80, 0.40, 0.90])

validation_results = evaluate_model(
    "Validation Example",
    test_y_true,
    test_y_pred,
    test_y_proba
)

print("\n--- Evaluation Function Validation ---")
for metric, value in validation_results.items():
    print(f"{metric}: {value}")

validation_mcnemar = mcnemar_exact(
    test_y_true,
    test_y_pred,
    np.array([0, 1, 0, 1, 1, 0])
)

print("\n--- McNemar Function Validation ---")
for item, value in validation_mcnemar.items():
    print(f"{item}: {value}")

print("\nTask 5 functions validated successfully.")
# ------------------------------------------------------------
# 5. LOAD AND VALIDATE EXPERIMENT RESULTS
# ------------------------------------------------------------

RESULTS_PATH = "results/task6_results.csv"

experiment_results = pd.read_csv(RESULTS_PATH)

required_metrics = [
    "Acc",
    "BalAcc",
    "Prec",
    "Recall",
    "F1",
    "ROC-AUC",
    "PR-AUC",
    "TN",
    "FP",
    "FN",
    "TP"
]

missing_metrics = [
    metric for metric in required_metrics
    if metric not in experiment_results.columns
]

print("\n--- Actual Experiment Results Check ---")
print("Number of evaluated models:", len(experiment_results))
print("Models:")
for model in experiment_results["Model"]:
    print(" -", model)

if missing_metrics:
    raise ValueError(
        f"Missing required evaluation metrics: {missing_metrics}"
    )

print("\nAll required classification metrics are present.")

# Verify confusion-matrix totals equal the test-set size
experiment_results["CM_Total"] = (
    experiment_results["TN"]
    + experiment_results["FP"]
    + experiment_results["FN"]
    + experiment_results["TP"]
)

expected_test_size = len(y_test)

if not (experiment_results["CM_Total"] == expected_test_size).all():
    raise ValueError(
        "At least one model's confusion matrix does not match "
        "the Task 5 test-set size."
    )

print(
    f"All confusion matrices contain exactly "
    f"{expected_test_size} test observations."
)

print("\nActual experiment results validated successfully.")
# ------------------------------------------------------------
# 6. LOAD AND VALIDATE STATISTICAL TEST RESULTS
# ------------------------------------------------------------

MCNEMAR_PATH = "results/task6_mcnemar.csv"

mcnemar_results = pd.read_csv(MCNEMAR_PATH)

required_mcnemar_columns = [
    "Compared with",
    "improved_only_correct",
    "other_only_correct",
    "p_raw",
    "p_holm"
]

missing_mcnemar_columns = [
    column for column in required_mcnemar_columns
    if column not in mcnemar_results.columns
]

print("\n--- Statistical Test Results Check ---")

if missing_mcnemar_columns:
    raise ValueError(
        f"Missing McNemar result columns: {missing_mcnemar_columns}"
    )

print("Number of model comparisons:", len(mcnemar_results))

# Check that p-values are valid probabilities
if not mcnemar_results["p_raw"].between(0, 1).all():
    raise ValueError("Invalid raw McNemar p-value detected.")

if not mcnemar_results["p_holm"].between(0, 1).all():
    raise ValueError("Invalid Holm-adjusted p-value detected.")

# Holm-adjusted p-values should never be smaller than raw p-values
if not (
    mcnemar_results["p_holm"] >= mcnemar_results["p_raw"]
).all():
    raise ValueError(
        "A Holm-adjusted p-value is smaller than its raw p-value."
    )

print("All raw McNemar p-values are valid.")
print("All Holm-adjusted p-values are valid.")
print("Holm correction consistency check passed.")

print("\nCompared models:")
for model in mcnemar_results["Compared with"]:
    print(" -", model)

print("\nStatistical test results validated successfully.")