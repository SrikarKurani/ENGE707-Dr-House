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
df = pd.read_csv("support2.csv")

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