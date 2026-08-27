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



# Dictionary explaining what each column means
data_dictionary = {
    'age': 'Patient age in years',
    'death': 'Died within ~6-month study follow-up (1=yes, 0=no)',
    'sex': 'Patient sex',
    'hospdead': 'Died during this hospital admission (1=yes, 0=no)',
    'slos': 'Length of hospital stay (days)',
    'd.time': 'Days of follow-up until death or censoring',
    'dzgroup': 'Primary disease group',
    'dzclass': 'Primary disease class (broader grouping of dzgroup)',
    'num.co': 'Number of comorbidities',
    'edu': 'Years of education',
    'income': 'Income bracket',
    'scoma': 'Coma/reactivity score at admission (SUPPORT physiology score)',
    'charges': 'Total hospital charges (USD)',
    'totcst': 'Total ratio-of-cost-to-charge cost estimate (USD)',
    'totmcst': 'Total micro-cost estimate (USD)',
    'avtisst': 'Average TISS (Therapeutic Intervention Scoring System) severity score',
    'race': 'Patient race',
    'sps': 'SUPPORT physiology score',
    'aps': 'APACHE III physiology score',
    # These two are predictions from the original study, not raw measurements
    'surv2m': "Model-estimated probability of surviving 2 months",
    'surv6m': "Model-estimated probability of surviving 6 months",
    'hday': 'Day of study entry (administrative index)',
    'diabetes': 'Diabetes present (1=yes, 0=no)',
    'dementia': 'Dementia present (1=yes, 0=no)',
    'ca': 'Cancer status (no / yes / metastatic)',
    'prg2m': "Physician-estimated probability of surviving 2 months",
    'prg6m': "Physician-estimated probability of surviving 6 months",
    'dnr': 'Do-not-resuscitate status',
    'dnrday': 'Study day on which DNR order was written',
    'meanbp': 'Mean arterial blood pressure (mmHg)',
    'wblc': 'White blood cell count (1000s/mm3)',
    'hrt': 'Heart rate (beats/min)',
    'resp': 'Respiration rate (breaths/min)',
    'temp': 'Body temperature (Celsius)',
    'pafi': 'PaO2/FiO2 ratio (oxygenation measure)',
    'alb': 'Serum albumin (g/dL)',
    'bili': 'Serum bilirubin (mg/dL)',
    'crea': 'Serum creatinine (mg/dL)',
    'sod': 'Serum sodium (mEq/L)',
    'ph': 'Arterial blood pH',
    'glucose': 'Blood glucose (mg/dL)',
    'bun': 'Blood urea nitrogen (mg/dL)',
    'urine': 'Urine output (mL/day)',
    'adlp': 'Activities of Daily Living score, patient-reported',
    'adls': 'Activities of Daily Living score, surrogate-reported',
    'sfdm2': 'Functional disability status at follow-up',
    'adlsc': 'Activities of Daily Living score, imputed/combined',
}

# Turns it into a table
dd = pd.DataFrame.from_dict(data_dictionary, orient='index', columns=['description'])
dd.index.name = 'variable'

# Check nothing was mistyped/missed
print(f"\nData dictionary covers {len(dd)} of {df.shape[1]} columns")
print(dd)
