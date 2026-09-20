import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from skeLCS import eLCS

# Task 2: Original LCS System on Raw Dataset
# Run eLCS on the original SUPPORT2 dataset without applying the full
# preprocessing from Task 3. The leakage columns are kept in this version
# because Task 2 is meant to provide a baseline using the original data.

TARGET = 'hospdead'

df = pd.read_csv('data/raw/support2.csv')
print("Raw dataset shape:", df.shape)

# Minimal processing required for eLCS
# eLCS requires the input data to be numeric, so the categorical columns
# are label encoded. This is only done so the data can be used by eLCS.
cat_cols = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca', 'dnr', 'sfdm2']
df_raw = df.copy()
for c in cat_cols:
    df_raw[c] = LabelEncoder().fit_transform(df_raw[c].astype(str))

# Missing values are not filled in here because eLCS can handle missing
# values using its built-in matching method.
print("Missing values remaining (untouched, eLCS handles these natively):",
      df_raw.drop(columns=[TARGET]).isnull().sum().sum())

X = df_raw.drop(columns=[TARGET]).values
y = df_raw[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# eLCS parameters
# Only learning_iterations and random_state are set manually.
# The other parameters use the library's default values.
params = dict(learning_iterations=5000, random_state=42)
print("\neLCS parameters used:", params)

start = time.time()
model = eLCS(**params)
model.fit(X_train, y_train)
elapsed = time.time() - start
print(f"Training time: {elapsed:.1f}s")

y_pred = model.predict(X_test)

print("\n--- Baseline Results (raw dataset, including leakage columns) ---")
print(f"Accuracy:          {accuracy_score(y_test, y_pred):.4f}")
print(f"Balanced Accuracy:  {balanced_accuracy_score(y_test, y_pred):.4f}")
print(f"F1-score:           {f1_score(y_test, y_pred):.4f}")
try:
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"ROC-AUC:            {roc_auc_score(y_test, y_proba):.4f}")
except Exception as e:
    print("ROC-AUC not available:", e)

print("""
Note: This baseline still includes the leakage columns.
They are kept for Task 2 so the original dataset can be tested.
The cleaned version will be used in Task 3 for comparison.
""")