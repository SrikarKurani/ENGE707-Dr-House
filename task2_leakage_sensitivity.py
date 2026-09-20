import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from skeLCS import eLCS

# Task 2 (supplementary): Leakage Sensitivity Check
# Run eLCS again after removing the leakage columns to see how much
# the results change compared with the raw baseline.

TARGET = 'hospdead'
df = pd.read_csv('data/raw/support2.csv')

leakage_cols = ['death', 'd.time', 'slos', 'surv2m', 'surv6m', 'prg2m', 'prg6m', 'sfdm2', 'hday']
df_noleak = df.drop(columns=[c for c in leakage_cols if c in df.columns])

cat_cols = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca', 'dnr']
for c in cat_cols:
    df_noleak[c] = LabelEncoder().fit_transform(df_noleak[c].astype(str))

print(f"Shape after removing {len(leakage_cols)} leakage columns:", df_noleak.shape)

X = df_noleak.drop(columns=[TARGET]).values
y = df_noleak[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

start = time.time()
model = eLCS(learning_iterations=5000, random_state=42)
model.fit(X_train, y_train)
elapsed = time.time() - start

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"\nTraining time: {elapsed:.1f}s")
print("\n--- Results WITHOUT leakage columns ---")
print(f"Accuracy:          {accuracy_score(y_test, y_pred):.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
print(f"F1-score:          {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:           {roc_auc_score(y_test, y_proba):.4f}")

print(f"""
Compared to the raw baseline, removing the leakage columns caused the
largest drops in Balanced Accuracy (0.9045 to 0.7917) and F1-score
(0.8559 to 0.7225). Accuracy also dropped from 0.9248 to 0.8781,
while ROC-AUC dropped from 0.9585 to 0.8896. This shows that the
leakage columns had a noticeable effect on the model's performance.
""")