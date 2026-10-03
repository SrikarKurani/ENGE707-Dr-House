import time
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from skeLCS import eLCS

# Task 6: model comparison. Same 80/20 stratified split (random_state=42) everywhere,
# so every model is scored on exactly the same test rows.
TARGET = 'hospdead'
SEED = 42

# ---------- data ----------
# (a) raw / minimally processed (same as Task 2, leakage columns KEPT)
raw = pd.read_csv('data/raw/support2.csv')
for c in ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca', 'dnr', 'sfdm2']:
    raw[c] = LabelEncoder().fit_transform(raw[c].astype(str))
X_raw = raw.drop(columns=[TARGET]).values
y = raw[TARGET].values

# (b) cleaned data from Task 3
clean = pd.read_csv('data/processed/support2_cleaned.csv')
y_clean = clean[TARGET].values
feature_names = [c for c in clean.columns if c != TARGET]
X_clean = clean[feature_names].astype(float).values   # bool dummies -> float

assert (y == y_clean).all(), "raw and cleaned rows are not aligned"

idx = np.arange(len(y))
tr, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
y_tr, y_te = y[tr], y[te]

# ---------- models ----------
experiments = {
    'eLCS original (raw data)':
        (eLCS(learning_iterations=5000, random_state=SEED), X_raw),
    'eLCS original (cleaned data)':
        (eLCS(learning_iterations=5000, random_state=SEED), X_clean),
    'eLCS improved (cleaned data)':
        (eLCS(learning_iterations=20000, N=2000, random_state=SEED), X_clean),
    'Random Forest':
        (RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1), X_clean),
    'Logistic Regression':
        (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X_clean),
    'SVM (RBF)':
        (make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED)), X_clean),
    'Decision Tree':
        (DecisionTreeClassifier(random_state=SEED), X_clean),
}

rows, preds, fitted = [], {}, {}
for name, (model, X) in experiments.items():
    t0 = time.time()
    model.fit(X[tr], y_tr)
    secs = time.time() - t0
    pred = model.predict(X[te])
    proba = model.predict_proba(X[te])[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    rows.append({
        'Model': name,
        'Acc': accuracy_score(y_te, pred),
        'BalAcc': balanced_accuracy_score(y_te, pred),
        'Prec': precision_score(y_te, pred, zero_division=0),
        'Recall': recall_score(y_te, pred),
        'F1': f1_score(y_te, pred),
        'ROC-AUC': roc_auc_score(y_te, proba),
        'PR-AUC': average_precision_score(y_te, proba),
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
        'Train s': round(secs, 1),
    })
    preds[name] = pred
    fitted[name] = model
    print(f'done: {name} ({secs:.1f}s)')

results = pd.DataFrame(rows).set_index('Model')
print(results.round(4).to_string())
results.round(4).to_csv('task6_results.csv')

# ---------- statistical test: exact McNemar, improved LCS vs every other model ----------
def mcnemar_exact(a_pred, b_pred, y_true):
    a_ok, b_ok = a_pred == y_true, b_pred == y_true
    n01 = int((a_ok & ~b_ok).sum())   # A right, B wrong
    n10 = int((~a_ok & b_ok).sum())   # A wrong, B right
    if n01 + n10 == 0:
        return n01, n10, 1.0
    return n01, n10, binomtest(min(n01, n10), n01 + n10, 0.5).pvalue

ref = 'eLCS improved (cleaned data)'
print(f'\nMcNemar (exact) vs {ref}')
for name in preds:
    if name == ref:
        continue
    n01, n10, p = mcnemar_exact(preds[ref], preds[name], y_te)
    print(f'{name:32s} improved-only-right={n01:4d}  other-only-right={n10:4d}  p={p:.4g}')

# ---------- Task 7 helper: dump rules from the improved eLCS ----------
# Attribute names below are from skeLCS's Classifier class; if one errors,
# run dir(model.population.popSet[0]) and adjust.
lcs = fitted[ref]
pop = lcs.population.popSet
print(f'\nPopulation: {len(pop)} classifiers (macro), '
      f'{sum(cl.numerosity for cl in pop)} (micro)')

def rule_to_text(cl):
    parts = []
    for att, cond in zip(cl.specifiedAttList, cl.condition):
        name = feature_names[att]
        if isinstance(cond, (list, tuple, np.ndarray)):   # continuous attribute -> [low, high]
            parts.append(f'{cond[0]:.3g} <= {name} <= {cond[1]:.3g}')
        else:
            parts.append(f'{name} == {cond}')
    return ' AND '.join(parts) + f'  ->  hospdead={cl.phenotype}'

top = sorted(pop, key=lambda c: c.numerosity * c.fitness, reverse=True)[:20]
with open('task7_top_rules.txt', 'w') as f:
    for i, cl in enumerate(top, 1):
        line = (f'[{i}] {rule_to_text(cl)} | numerosity={cl.numerosity} '
                f'fitness={cl.fitness:.3f} accuracy={cl.accuracy:.3f}')
        print(line)
        f.write(line + '\n')
