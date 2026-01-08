# predict.py

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import os

# ---- Step 1: Load saved Random Forest model ----
MODEL_PATH = "../models/Random_Forest_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run modeling.py first!")

model = joblib.load(MODEL_PATH)
print("Model loaded successfully!")

# ---- Step 2: Load feature columns from cleaned data ----
CLEANED_DATA_PATH = "../data/processed/cleaned_life_expectancy.csv"
df_cleaned = pd.read_csv(CLEANED_DATA_PATH)
df_cleaned.columns = df_cleaned.columns.str.strip()  # remove spaces

feature_cols = df_cleaned.drop("Life expectancy", axis=1).columns.tolist()

# ---- Step 3: Prepare new input ----
# Example: only provide a few numeric features; the rest will be 0
input_data = {
    "Year": 2016,
    "Adult Mortality": 200,
    "infant deaths": 20,
    "Alcohol": 4.0,
    "percentage expenditure": 5.0,
    "Hepatitis B": 90,
    "Measles": 50,
    " BMI": 25.0,
    "under-five deaths": 25,
    "Polio": 90,
    "Total expenditure": 5.0,
    "Diphtheria": 90,
    " HIV/AIDS": 0.1,
    "GDP": 5000,
    "Population": 1000000,
    " thinness  1-19 years": 10,
    " thinness 5-9 years": 9,
    "Income composition of resources": 0.5,
    "Schooling": 12.0,
    # Do not include country/status columns; script will handle them automatically
}

# Create DataFrame with all features set to 0
new_data = pd.DataFrame(0, index=[0], columns=feature_cols)

# Fill the input values
for key, value in input_data.items():
    if key in new_data.columns:
        new_data[key] = value

# ---- Step 4: Scale features ----
scaler = StandardScaler()
new_data_scaled = scaler.fit_transform(new_data)

# ---- Step 5: Predict ----
pred = model.predict(new_data_scaled)
print("\nPredicted Life Expectancy:", round(pred[0], 2))
