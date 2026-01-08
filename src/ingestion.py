import pandas as pd
import os

# ---- Paths ----
RAW_DATA_PATH = os.path.join("..", "data", "raw", "Life Expectancy Data.csv")
PROCESSED_DATA_PATH = os.path.join("..", "data", "processed", "cleaned_life_expectancy.csv")

# ---- Step 1: Read CSV ----
print(f"Using CSV: {RAW_DATA_PATH}")
df = pd.read_csv(RAW_DATA_PATH)
print(df.head())

# ---- Step 2: Check for missing values ----
print("\nMissing values in each column:")
print(df.isnull().sum())

# ---- Step 3: Handle missing values ----
# Fill numeric columns with mean
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

# Fill categorical columns with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# ---- Step 4: Encode categorical variables ----
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# ---- Step 5: Save cleaned data ----
os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
df.to_csv(PROCESSED_DATA_PATH, index=False)
print(f"\nCleaned data saved successfully at: {PROCESSED_DATA_PATH}")
