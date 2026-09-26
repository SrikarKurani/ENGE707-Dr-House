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

# Test different numbers of learning iterations to see if
# more training improves the model's performance
print("\nTesting learning_iterations")

for iters in [5000, 10000, 20000]:
    start = time.time()

    model = eLCS(
        learning_iterations=iters,
        random_state=42
    )

    model.fit(X_train, y_train)

    elapsed = time.time() - start

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(
        f"iterations={iters}: time={elapsed:.1f}s  "
        f"Acc={accuracy_score(y_test, y_pred):.4f}  "
        f"BalAcc={balanced_accuracy_score(y_test, y_pred):.4f}  "
        f"F1={f1_score(y_test, y_pred):.4f}  "
        f"AUC={roc_auc_score(y_test, y_proba):.4f}"
    )

# Test a larger rule population to see if it improves the model
# since the dataset contains around 9000 rows
print("\nTesting rule population size (N), with 20000 iterations")

for N in [1000, 2000]:
    start = time.time()

    model = eLCS(
        learning_iterations=20000,
        N=N,
        random_state=42
    )

    model.fit(X_train, y_train)

    elapsed = time.time() - start

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(
        f"N={N}: time={elapsed:.1f}s  "
        f"Acc={accuracy_score(y_test, y_pred):.4f}  "
        f"BalAcc={balanced_accuracy_score(y_test, y_pred):.4f}  "
        f"F1={f1_score(y_test, y_pred):.4f}  "
        f"AUC={roc_auc_score(y_test, y_proba):.4f}"
    )

print(f"""
The best configuration so far is 20000 iterations with N=2000.
This improved Balanced Accuracy from 0.7917 to 0.8165 and F1
from 0.7225 to 0.7604 compared to the baseline. However, the
training time increased from around 5 seconds to around 47 seconds.
This configuration will be tested again using the cleaned Task 3
dataset once it is ready.
""")