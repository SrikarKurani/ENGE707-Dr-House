import pandas as pd

# Increase the display limits so pandas does not cut off large outputs (aka datatypes etc)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 150)

# Load the raw SUPPORT2 dataset into a DataFrame
df = pd.read_csv("data/raw/support2.csv")

# Check the size of the dataset to make sure it has the expected number of rows and columns
print("Number of rows and columns:", df.shape)
print()

# Check the data type of each column. This helps identify any columns
# that may have been incorrectly loaded as text instead of numeric data
print("Data types:")
print(df.dtypes)
print()

# Check whether there are any completely duplicated rows
duplicates = df.duplicated().sum()
print("Number of duplicate rows:", duplicates)
print()

# Calculate the number and percentage of missing values in each column
missing_count = df.isnull().sum()
missing_percent = (missing_count / len(df) * 100).round(2)

# Combine the missing value counts and percentages into one table
missing_summary = pd.DataFrame({
    "missing_count": missing_count,
    "missing_percent": missing_percent
})

# Only show columns that contain missing values and sort them so that
# columns with the most missing data appear first
missing_summary = missing_summary[missing_summary["missing_count"] > 0]
missing_summary = missing_summary.sort_values("missing_percent", ascending=False)

print("Missing values per column:")
print(missing_summary)
print()

# Find the columns containing categorical/text data
cat_columns = df.select_dtypes(include=["object", "str"]).columns
print("Categorical columns detected:", list(cat_columns))
print()

# Display the different values in each categorical column.
# This helps identify unexpected values or inconsistent categories
for col in cat_columns:
    print(f"{col} - unique values:")
    print(df[col].value_counts(dropna=False))
    print()

# Find the numeric columns and generate summary statistics.
# These can be used to get an overview of the data and identify
# unusual values or possible outliers
num_columns = df.select_dtypes(include="number").columns
print("Numeric column summary:")
print(df[num_columns].describe().T)
print()

# Check the distribution of the two possible target variables.
# This helps compare the number of patients in each class before
# selecting the target variable for the machine learning model
print("death value counts:")
print(df["death"].value_counts())
print()

print("hospdead value counts:")
print(df["hospdead"].value_counts())