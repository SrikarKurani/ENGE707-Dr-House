# Task 2 - Data Inspection
# Use the support2 dataset and checking its structure, missing values, 
# duplicates, and other statistics before we can start cleaning

import pandas as pd

pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 150)

# Load the dataset
df = pd.read_csv("data/raw/support2.csv")

print("Number of rows and columns:", df.shape)
print()

# Check data types for each column
print("Data types:")
print(df.dtypes)
print()

# Check for duplicate rows
duplicates = df.duplicated().sum()
print("Number of duplicate rows:", duplicates)
print()

# check missing values (amount and percentage for each column)
missing_count = df.isnull().sum()
missing_percent = (missing_count / len(df) * 100).round(2)

missing_summary = pd.DataFrame({
    "missing_count": missing_count,
    "missing_percent": missing_percent
})

# Show columns that actually have missing values, sorted from worst to best
missing_summary = missing_summary[missing_summary["missing_count"] > 0]
missing_summary = missing_summary.sort_values("missing_percent", ascending=False)

print("Missing values per column:")
print(missing_summary)
print()

# Look at the categorical (text) columns to check for weird values/typos
cat_columns = df.select_dtypes(include="object").columns

for col in cat_columns:
    print(f"{col} - unique values:")
    print(df[col].value_counts(dropna=False))
    print()

# Summary statistics for numeric columns (checking ranges, possible outliers)
num_columns = df.select_dtypes(include="number").columns
print("Numeric column summary:")
print(df[num_columns].describe().T)
print()

# Checking the target variables we might use for the ML model later
print("death value counts:")
print(df["death"].value_counts())
print()

print("hospdead value counts:")
print(df["hospdead"].value_counts())