import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from skeLCS import eLCS

# Task 4: Improved LCS System (preliminary tuning)
# Testing different eLCS settings to see if the model can improve
# on the Task 2 baseline. The leakage columns have been removed
# for now, and the cleaned dataset from Task 3 will be used later.

TARGET = 'hospdead'
df = pd.read_csv('data/raw/support2.csv')

# Remove the columns identified as data leakage in Task 2
leakage_cols = [
    'death', 'd.time', 'slos', 'surv2m', 'surv6m',
    'prg2m', 'prg6m', 'sfdm2', 'hday'
]

df_noleak = df.drop(
    columns=[c for c in leakage_cols if c in df.columns]
)

# Convert categorical columns into numerical values
cat_cols = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca', 'dnr']

for c in cat_cols:
    df_noleak[c] = LabelEncoder().fit_transform(
        df_noleak[c].astype(str)
    )

# Separate the features and target variable
X = df_noleak.drop(columns=[TARGET]).values
y = df_noleak[TARGET].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Check that the baseline model still produces the expected results
# before making any changes to the eLCS settings.
model = eLCS(
    learning_iterations=5000,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Baseline check:")
print(
    f"Acc={accuracy_score(y_test, y_pred):.4f}  "
    f"BalAcc={balanced_accuracy_score(y_test, y_pred):.4f}  "
    f"F1={f1_score(y_test, y_pred):.4f}  "
    f"AUC={roc_auc_score(y_test, y_proba):.4f}"
)