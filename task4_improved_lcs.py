import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from skeLCS import eLCS

# Task 4: Improved LCS System
# Use the preprocessed data from Task 3

TARGET = 'hospdead'

train = pd.read_csv('data/processed/support2_train_preprocessed.csv')
test = pd.read_csv('data/processed/support2_test_preprocessed.csv')

X_train = train.drop(columns=[TARGET]).values.astype(float)
y_train = train[TARGET].values
X_test = test.drop(columns=[TARGET]).values.astype(float)
y_test = test[TARGET].values

print("Train:", X_train.shape, "Test:", X_test.shape)


def evaluate(model, X, y):
    # Get the model predictions and probabilities.
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    return {
        'Acc': accuracy_score(y, y_pred),
        'BalAcc': balanced_accuracy_score(y, y_pred),
        'F1': f1_score(y, y_pred),
        'AUC': roc_auc_score(y, y_proba),
    }


# Stage 1: Original eLCS on the Task 3 data
# Use the same settings as the Task 2 baseline so the results 
# can be compared after preprocessing.

start = time.time()

original = eLCS(learning_iterations=5000, random_state=42)
original.fit(X_train, y_train)

orig_time = time.time() - start
orig_scores = evaluate(original, X_test, y_test)

print(f"\nOriginal eLCS on Task 3 data (trained in {orig_time:.1f}s):")
print(" ".join(f"{k}={v:.4f}" for k, v in orig_scores.items()))

# Stage 2: Choose the improved settings using cross-validation
# Use 3-fold cross-validation on the training data to compare
# different eLCS settings. The test data is not used here.
# p_spec controls how specific the rules are for each feature.
# Lower values can create more general rules, so p_spec=0.2
# is also tested.

configs = [
    dict(learning_iterations=5000, N=1000),
    dict(learning_iterations=10000, N=1000),
    dict(learning_iterations=10000, N=1000, p_spec=0.2),
    dict(learning_iterations=10000, N=2000, p_spec=0.2),
]

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cv_rows = []

for cfg in configs:
    fold_scores = []
    start = time.time()

    for fit_idx, val_idx in skf.split(X_train, y_train):
        m = eLCS(random_state=42, **cfg)
        m.fit(X_train[fit_idx], y_train[fit_idx])

        fold_scores.append(
            balanced_accuracy_score(
                y_train[val_idx],
                m.predict(X_train[val_idx])
            )
        )

    cv_rows.append({
        'config': cfg,
        'cv_BalAcc': np.mean(fold_scores),
        'cv_std': np.std(fold_scores),
        'secs': time.time() - start,
    })

    print(
        f"{cfg}: CV BalAcc={np.mean(fold_scores):.4f} "
        f"(+/- {np.std(fold_scores):.4f}), "
        f"{time.time() - start:.0f}s"
    )

best = max(cv_rows, key=lambda r: r['cv_BalAcc'])
best_cfg = best['config']

print("\nSelected config:", best_cfg)

# Stage 3: Train the final improved model and test it once
start = time.time()

improved = eLCS(random_state=42, **best_cfg)
improved.fit(X_train, y_train)

imp_time = time.time() - start
imp_scores = evaluate(improved, X_test, y_test)

print(f"\nImproved eLCS on Task 3 data (trained in {imp_time:.1f}s):")
print(" ".join(f"{k}={v:.4f}" for k, v in imp_scores.items()))


# Add the results from Task 2 so all the models can be compared.
# The same train/test split was used, so the results are comparable.

rows = [
    ("Task 2: raw data, leakage kept", 0.9248, 0.9045, 0.8559, 0.9585),
    ("Task 2: raw data, leakage removed", 0.8781, 0.7917, 0.7225, 0.8896),
    ("Original eLCS, Task 3 data", *orig_scores.values()),
    ("Improved eLCS, Task 3 data", *imp_scores.values()),
]

lines = [
    "| Model | Accuracy | Balanced Acc | F1 | ROC-AUC |",
    "|---|---|---|---|---|"
]

for name, a, b, f, u in rows:
    lines.append(
        f"| {name} | {a:.4f} | {b:.4f} | {f:.4f} | {u:.4f} |"
    )

table = "\n".join(lines)

print("\n" + table)


# Save the results to a markdown file.
with open('task4_results.md', 'w') as f:
    f.write("# Task 4 results\n\n")
    f.write(f"Selected config (3-fold CV on training data): `{best_cfg}`\n\n")
    f.write(table + "\n")

print("\nSaved task4_results.md")

