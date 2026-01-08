import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Paths
BASE_DIR = os.path.abspath(".")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_life_expectancy.csv")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# Load processed dataset
df = pd.read_csv(DATA_PATH)

# Select feature columns (exclude target)
feature_cols = df.drop("Life expectancy ", axis=1).columns

# Create and fit scaler
scaler = StandardScaler()
scaler.fit(df[feature_cols])

# Save scaler
joblib.dump(scaler, SCALER_PATH)

print(f"scaler.pkl created successfully at {SCALER_PATH}")
