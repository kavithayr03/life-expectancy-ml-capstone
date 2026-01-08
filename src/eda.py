# eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Step 1: Load cleaned data ----
df = pd.read_csv("../data/processed/cleaned_life_expectancy.csv")
print(df.head())
print(df.info())
print(df.describe())

# ---- Step 2: Correlation heatmap ----
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()  # <-- make sure this is here

# ---- Step 3: Feature distributions ----
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols].hist(figsize=(15,12))
plt.tight_layout()
plt.show()  # <-- this is required
