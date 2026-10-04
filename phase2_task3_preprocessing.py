# ============================================================
# ENGE707 - Project Phase II
# Task 3: Data Preprocessing and Feature Engineering
# Dataset: SUPPORT2
# Target: hospdead
# ============================================================

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

print("Phase II Task 3 started successfully.")
# ============================================================
# 1. LOAD AND INSPECT THE RAW SUPPORT2 DATASET
# ============================================================

# Load the original dataset
df = pd.read_csv("data/raw/support2.csv")

print("\n--- RAW DATASET CHECK ---")

# Dataset dimensions
print("Dataset shape:", df.shape)
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# Display column names
print("\nColumn names:")
print(df.columns.tolist())

# Display first 5 records
print("\nFirst 5 rows:")
print(df.head())

# Check data types
print("\nData types:")
print(df.dtypes)

# Check duplicate records
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

# Check missing values
print("\nMissing values per column:")
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100

missing_table = pd.DataFrame({
    "Missing Count": missing,
    "Missing Percent": missing_percent
})

print(missing_table[missing_table["Missing Count"] > 0]
      .sort_values("Missing Percent", ascending=False))

# Check the target variable
print("\nTarget variable (hospdead) distribution:")
print(df["hospdead"].value_counts(dropna=False))

print("\nTarget percentages:")
print(df["hospdead"].value_counts(normalize=True, dropna=False) * 100)
# ============================================================
# TASK 3 - FURTHER DATA QUALITY CHECKS
# ============================================================

print("\n" + "=" * 60)
print("TASK 3 - DATA QUALITY CHECKS")
print("=" * 60)

# ------------------------------------------------------------
# 1. Check duplicate records
# ------------------------------------------------------------

duplicate_count = df.duplicated().sum()

print("\nNumber of duplicate rows:")
print(duplicate_count)


# ------------------------------------------------------------
# 2. Check data types
# ------------------------------------------------------------

print("\nData types:")
print(df.dtypes)

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()

print("\nNumber of numerical columns:", len(numeric_columns))
print("Numerical columns:")
print(numeric_columns)

print("\nNumber of categorical columns:", len(categorical_columns))
print("Categorical columns:")
print(categorical_columns)


# ------------------------------------------------------------
# 3. Inspect categorical values
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CATEGORICAL VARIABLE CHECK")
print("=" * 60)

for column in categorical_columns:

    print(f"\nColumn: {column}")

    print("Unique values:")
    print(df[column].value_counts(dropna=False).head(20))


# ------------------------------------------------------------
# 4. Check target for missing or invalid values
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TARGET CHECK")
print("=" * 60)

print("\nTarget unique values:")
print(df["hospdead"].value_counts(dropna=False))

print("\nMissing target values:")
print(df["hospdead"].isnull().sum())


# ------------------------------------------------------------
# 5. Identify possible target leakage variables
# ------------------------------------------------------------

# These variables may contain information that occurs after,
# or is directly related to, the hospital mortality outcome.
# They are flagged here for investigation before modelling.

possible_leakage_columns = [
    "death",
    "d.time",
    "slos"
]

print("\n" + "=" * 60)
print("POSSIBLE TARGET LEAKAGE VARIABLES")
print("=" * 60)

for column in possible_leakage_columns:

    if column in df.columns:

        print(f"\n{column}")
        print(df[column].describe())


# ------------------------------------------------------------
# 6. Numerical summary
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("NUMERICAL SUMMARY")
print("=" * 60)

print(df[numeric_columns].describe().T)


print("\nTask 3 data-quality investigation completed.")
# ============================================================
# TASK 3 - CREATE PREPROCESSING DATASET
# ============================================================

print("\n" + "=" * 60)
print("CREATING PREPROCESSING DATASET")
print("=" * 60)

# Keep the original dataframe unchanged
df_processed = df.copy()

print("\nOriginal shape:")
print(df_processed.shape)


# ------------------------------------------------------------
# 1. REMOVE TARGET LEAKAGE / OUTCOME-RELATED VARIABLES
# ------------------------------------------------------------

# These variables were identified during Phase I as variables
# that should not be used as predictors for hospdead.

leakage_columns = [
    "death",
    "d.time",
    "slos",
    "surv2m",
    "surv6m",
    "prg2m",
    "prg6m",
    "sfdm2",
    "hday"
]

existing_leakage_columns = [
    column for column in leakage_columns
    if column in df_processed.columns
]

df_processed = df_processed.drop(
    columns=existing_leakage_columns
)

print("\nRemoved leakage/outcome-related columns:")
print(existing_leakage_columns)

print("\nShape after leakage removal:")
print(df_processed.shape)


# ------------------------------------------------------------
# 2. REMOVE DUPLICATE RECORDS
# ------------------------------------------------------------

rows_before = len(df_processed)

df_processed = df_processed.drop_duplicates()

rows_after = len(df_processed)

print("\nDuplicate rows removed:")
print(rows_before - rows_after)


# ------------------------------------------------------------
# 3. HANDLE INVALID PHYSIOLOGICAL ZERO VALUES
# ------------------------------------------------------------

# Phase I identified zero values in these measurements as
# invalid/unrecorded physiological measurements.
# Convert these values to NaN so they can later be imputed.

invalid_zero_columns = [
    "meanbp",
    "hrt",
    "resp"
]

print("\nInvalid zero values converted to NaN:")

for column in invalid_zero_columns:

    if column in df_processed.columns:

        invalid_count = (
            df_processed[column] == 0
        ).sum()

        print(
            f"{column}: {invalid_count}"
        )

        df_processed.loc[
            df_processed[column] == 0,
            column
        ] = np.nan


# ------------------------------------------------------------
# 4. HANDLE INVALID NEGATIVE VALUES
# ------------------------------------------------------------

# Convert invalid negative values to NaN so they can later
# be handled by numerical median imputation.

invalid_negative_columns = ["dnrday", "totmcst"]

print("\nInvalid negative values converted to NaN:")

for column in invalid_negative_columns:

    negative_count = (df_processed[column] < 0).sum()

    print(f"{column}: {negative_count}")

    df_processed.loc[
        df_processed[column] < 0,
        column
    ] = np.nan


# ------------------------------------------------------------
# 5. CREATE MISSINGNESS INDICATORS
# ------------------------------------------------------------

# Preserve information about whether highly-missing variables
# were originally recorded or missing.

missing_percent = (
    df_processed.isnull().mean() * 100
)

high_missing_columns = (
    missing_percent[
        missing_percent > 40
    ]
    .index
    .tolist()
)

print("\nColumns with more than 40% missing data:")
print(high_missing_columns)

for column in high_missing_columns:

    indicator_name = (
        column + "_was_missing"
    )

    df_processed[indicator_name] = (
        df_processed[column]
        .isnull()
        .astype(int)
    )

print("\nMissingness indicators created:")
print(
    [
        column + "_was_missing"
        for column in high_missing_columns
    ]
)


# ------------------------------------------------------------
# 6. VERIFY TARGET
# ------------------------------------------------------------

print("\nTarget variable check:")

print(
    df_processed["hospdead"]
    .value_counts()
)

print("\nMissing target values:")

print(
    df_processed["hospdead"]
    .isnull()
    .sum()
)


# ------------------------------------------------------------
# 7. PREPROCESSING SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PREPROCESSING SUMMARY")
print("=" * 60)

print(
    "Original dataset shape:",
    df.shape
)

print(
    "Current processed shape:",
    df_processed.shape
)

print(
    "Remaining missing values:",
    df_processed.isnull().sum().sum()
)

print(
    "Remaining duplicate rows:",
    df_processed.duplicated().sum()
)

print(
    "\nInitial preprocessing stage completed."
)
# ============================================================
# TASK 3 - TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

TARGET = "hospdead"

# Separate predictors and target
X = df_processed.drop(columns=[TARGET])
y = df_processed[TARGET]

# Use the same split settings as the Task 2 baseline
# so later model comparisons use the same test proportion
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining set shape:")
print(X_train.shape)

print("\nTest set shape:")
print(X_test.shape)

print("\nTraining target distribution (%):")
print(y_train.value_counts(normalize=True) * 100)

print("\nTest target distribution (%):")
print(y_test.value_counts(normalize=True) * 100)


# ============================================================
# IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

categorical_features = (
    X_train.select_dtypes(exclude=np.number)
    .columns
    .tolist()
)

numerical_features = (
    X_train.select_dtypes(include=np.number)
    .columns
    .tolist()
)

print("\nCategorical features:")
print(categorical_features)

print("\nNumber of categorical features:")
print(len(categorical_features))

print("\nNumber of numerical features:")
print(len(numerical_features))


# ============================================================
# CHECK MISSING VALUES AFTER SPLIT
# ============================================================

print("\nMissing values in training predictors:")
print(X_train.isnull().sum().sum())

print("\nMissing values in test predictors:")
print(X_test.isnull().sum().sum())

print("\nTrain/test split completed successfully.")
# ============================================================
# TASK 3 - IMPUTATION AND CATEGORICAL ENCODING
# ============================================================

print("\n" + "=" * 60)
print("IMPUTATION AND CATEGORICAL ENCODING")
print("=" * 60)


# ------------------------------------------------------------
# 1. NUMERICAL IMPUTATION
# ------------------------------------------------------------

# Use the median of each numerical feature to replace missing
# numerical values.
#
# The imputer is fitted ONLY on the training data.
# The same fitted imputer is then applied to the test data.
# This prevents information from the test set leaking into training.

numeric_imputer = SimpleImputer(strategy="median")

X_train_numeric = numeric_imputer.fit_transform(
    X_train[numerical_features]
)

X_test_numeric = numeric_imputer.transform(
    X_test[numerical_features]
)

# Convert the arrays back into DataFrames so the feature
# names and row indices are retained.

X_train_numeric = pd.DataFrame(
    X_train_numeric,
    columns=numerical_features,
    index=X_train.index
)

X_test_numeric = pd.DataFrame(
    X_test_numeric,
    columns=numerical_features,
    index=X_test.index
)

print("\nNumerical imputation completed.")

print(
    "Missing numerical values in training set:",
    X_train_numeric.isnull().sum().sum()
)

print(
    "Missing numerical values in test set:",
    X_test_numeric.isnull().sum().sum()
)


# ------------------------------------------------------------
# 2. CATEGORICAL IMPUTATION
# ------------------------------------------------------------

# Missing categorical values are represented using a separate
# "Missing" category instead of deleting the affected patients.

categorical_imputer = SimpleImputer(
    strategy="constant",
    fill_value="Missing"
)

X_train_categorical = categorical_imputer.fit_transform(
    X_train[categorical_features]
)

X_test_categorical = categorical_imputer.transform(
    X_test[categorical_features]
)

X_train_categorical = pd.DataFrame(
    X_train_categorical,
    columns=categorical_features,
    index=X_train.index
)

X_test_categorical = pd.DataFrame(
    X_test_categorical,
    columns=categorical_features,
    index=X_test.index
)

print("\nCategorical imputation completed.")

print(
    "Missing categorical values in training set:",
    X_train_categorical.isnull().sum().sum()
)

print(
    "Missing categorical values in test set:",
    X_test_categorical.isnull().sum().sum()
)


# ------------------------------------------------------------
# 3. ONE-HOT ENCODING
# ------------------------------------------------------------

# Convert categorical variables into numeric binary features.
# handle_unknown="ignore" prevents an error if a category appears
# in the test set that was not present in the training set.

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

X_train_encoded = encoder.fit_transform(
    X_train_categorical
)

X_test_encoded = encoder.transform(
    X_test_categorical
)

encoded_feature_names = encoder.get_feature_names_out(
    categorical_features
)

X_train_encoded = pd.DataFrame(
    X_train_encoded,
    columns=encoded_feature_names,
    index=X_train.index
)

X_test_encoded = pd.DataFrame(
    X_test_encoded,
    columns=encoded_feature_names,
    index=X_test.index
)

print("\nCategorical encoding completed.")

print(
    "Number of one-hot encoded categorical features:",
    len(encoded_feature_names)
)


# ------------------------------------------------------------
# 4. COMBINE NUMERICAL AND CATEGORICAL FEATURES
# ------------------------------------------------------------

X_train_processed = pd.concat(
    [
        X_train_numeric,
        X_train_encoded
    ],
    axis=1
)

X_test_processed = pd.concat(
    [
        X_test_numeric,
        X_test_encoded
    ],
    axis=1
)


# ------------------------------------------------------------
# 5. VALIDATE THE PROCESSED DATA
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PROCESSED DATA VALIDATION")
print("=" * 60)

print(
    "\nProcessed training shape:",
    X_train_processed.shape
)

print(
    "Processed test shape:",
    X_test_processed.shape
)

print(
    "\nMissing values in processed training data:",
    X_train_processed.isnull().sum().sum()
)

print(
    "Missing values in processed test data:",
    X_test_processed.isnull().sum().sum()
)

print(
    "\nInfinite values in processed training data:",
    np.isinf(X_train_processed.to_numpy()).sum()
)

print(
    "Infinite values in processed test data:",
    np.isinf(X_test_processed.to_numpy()).sum()
)

print(
    "\nTask 3 imputation and categorical encoding completed successfully."
)
# ============================================================
# TASK 3 - NUMERICAL FEATURE SCALING
# ============================================================

print("\n" + "=" * 60)
print("NUMERICAL FEATURE SCALING")
print("=" * 60)


# ------------------------------------------------------------
# 1. INSPECT FEATURE RANGES BEFORE SCALING
# ------------------------------------------------------------

print("\nExample numerical feature ranges before scaling:")

range_summary = pd.DataFrame({
    "Min": X_train_numeric.min(),
    "Max": X_train_numeric.max(),
    "Mean": X_train_numeric.mean(),
    "Std": X_train_numeric.std()
})

print(range_summary)


# ------------------------------------------------------------
# 2. STANDARDISE NUMERICAL FEATURES
# ------------------------------------------------------------

# StandardScaler is fitted ONLY on the training data.
# The same scaler is then used to transform the test data.
# This avoids information leakage from the test set.

scaler = StandardScaler()

X_train_numeric_scaled = scaler.fit_transform(
    X_train_numeric
)

X_test_numeric_scaled = scaler.transform(
    X_test_numeric
)

X_train_numeric_scaled = pd.DataFrame(
    X_train_numeric_scaled,
    columns=numerical_features,
    index=X_train.index
)

X_test_numeric_scaled = pd.DataFrame(
    X_test_numeric_scaled,
    columns=numerical_features,
    index=X_test.index
)


# ------------------------------------------------------------
# 3. COMBINE SCALED NUMERICAL + ENCODED CATEGORICAL FEATURES
# ------------------------------------------------------------

X_train_final = pd.concat(
    [
        X_train_numeric_scaled,
        X_train_encoded
    ],
    axis=1
)

X_test_final = pd.concat(
    [
        X_test_numeric_scaled,
        X_test_encoded
    ],
    axis=1
)


# ------------------------------------------------------------
# 4. VALIDATE FINAL FEATURE MATRICES
# ------------------------------------------------------------

print("\nFinal training shape:")
print(X_train_final.shape)

print("\nFinal test shape:")
print(X_test_final.shape)

print("\nMissing values in final training data:")
print(X_train_final.isnull().sum().sum())

print("\nMissing values in final test data:")
print(X_test_final.isnull().sum().sum())

print("\nInfinite values in final training data:")
print(np.isinf(X_train_final.to_numpy()).sum())

print("\nInfinite values in final test data:")
print(np.isinf(X_test_final.to_numpy()).sum())


# Check the mean and standard deviation of the scaled
# numerical training variables.

print("\nScaled numerical training means (first 10):")
print(
    X_train_numeric_scaled.mean()
    .head(10)
    .round(4)
)

print("\nScaled numerical training standard deviations (first 10):")
print(
    X_train_numeric_scaled.std()
    .head(10)
    .round(4)
)


print("\nNumerical feature scaling completed successfully.")
# ------------------------------------------------------------
# FINAL TASK 3 OUTPUT FILES
# ------------------------------------------------------------

import os

print("\n" + "=" * 60)
print("SAVING TASK 3 PROCESSED DATA")
print("=" * 60)

# Create processed-data folder if it does not already exist
os.makedirs("data/processed", exist_ok=True)

# Add target back to each dataset
train_unscaled = X_train_processed.copy()
test_unscaled = X_test_processed.copy()

train_scaled = X_train_final.copy()
test_scaled = X_test_final.copy()

train_unscaled[TARGET] = y_train.values
test_unscaled[TARGET] = y_test.values

train_scaled[TARGET] = y_train.values
test_scaled[TARGET] = y_test.values

# Save unscaled processed data
train_unscaled.to_csv(
    "data/processed/support2_train_preprocessed.csv",
    index=False
)

test_unscaled.to_csv(
    "data/processed/support2_test_preprocessed.csv",
    index=False
)

# Save scaled version separately
train_scaled.to_csv(
    "data/processed/support2_train_scaled.csv",
    index=False
)

test_scaled.to_csv(
    "data/processed/support2_test_scaled.csv",
    index=False
)

print("\nFiles saved successfully:")
print("data/processed/support2_train_preprocessed.csv")
print("data/processed/support2_test_preprocessed.csv")
print("data/processed/support2_train_scaled.csv")
print("data/processed/support2_test_scaled.csv")

print("\nFinal saved dataset shapes:")
print("Train unscaled:", train_unscaled.shape)
print("Test unscaled:", test_unscaled.shape)
print("Train scaled:", train_scaled.shape)
print("Test scaled:", test_scaled.shape)

print("\nTarget counts in saved training data:")
print(train_unscaled[TARGET].value_counts())

print("\nTarget counts in saved test data:")
print(test_unscaled[TARGET].value_counts())

print("\nTASK 3 PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY.")

# TASK 3 - OUTLIER AND CLASS IMBALANCE CHECKS
# These checks will print information. 

print("\n" + "=" * 60)
print("OUTLIER CHECK (IQR rule, training data only)")
print("=" * 60)

# Only check continuous numerical features for outliers.
continuous_features = [
    column for column in numerical_features
    if X_train[column].nunique() > 10
]

q1 = X_train[continuous_features].quantile(0.25)
q3 = X_train[continuous_features].quantile(0.75)
iqr = q3 - q1

lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr

outlier_counts = (
    (X_train[continuous_features] < lower_limit) |
    (X_train[continuous_features] > upper_limit)
).sum()

outlier_table = pd.DataFrame({
    "Outliers": outlier_counts,
    "Percent of training rows": (outlier_counts / len(X_train) * 100).round(1)
}).sort_values("Outliers", ascending=False)

print("\nFeatures with the most IQR outliers:")
print(outlier_table.head(10))

print(
    "\nDecision: outliers are kept. Extreme values like very high hospital"
    "\ncharges or creatinine are believable for critically ill patients, so"
    "\nthey are not data entry mistakes (the impossible zero and negative"
    "\nvalues were already fixed above). eLCS rules use intervals instead of"
    "\ndistances, so extreme values should affect them less than they would"
    "\naffect a distance-based or linear model."
)

print("\n" + "=" * 60)
print("CLASS IMBALANCE CHECK")
print("=" * 60)

train_counts = y_train.value_counts()
imbalance_ratio = train_counts[0] / train_counts[1]

print("\nTraining class counts:")
print(train_counts)
print(f"\nMajority:minority ratio = {imbalance_ratio:.2f} : 1")

print(
    "\nDecision: no resampling. The imbalance is only moderate (about 26%"
    "\npositive), eLCS has no class weight option, and resampling would make"
    "\nit harder to compare directly with the Task 2 baselines and the other"
    "\nmodels. Instead the results are judged with balanced accuracy, F1 and"
    "\nPR-AUC as well as accuracy, so the death class doesn't get hidden."
)

print("\nTask 3 outlier and class imbalance checks completed.")