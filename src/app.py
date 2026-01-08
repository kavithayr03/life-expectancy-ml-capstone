# app.py

import streamlit as st
import pandas as pd
import joblib
import os

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "Random_Forest_model.pkl")
FEATURE_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_life_expectancy.csv")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# ---- Load files ----
model = joblib.load(MODEL_PATH)
df_cleaned = pd.read_csv(FEATURE_DATA_PATH)
df_cleaned.columns = df_cleaned.columns.str.strip()
feature_cols = df_cleaned.drop("Life expectancy", axis=1).columns.tolist()
scaler = joblib.load(SCALER_PATH)

# ---- UI ----
st.title("Life Expectancy Prediction App")
st.markdown("Enter the values for key features to predict Life Expectancy:")

# ---- Inputs ----
input_data = {}
input_data["Year"] = st.number_input("Year", min_value=2000, max_value=2030, value=2016)
input_data["Adult Mortality"] = st.number_input("Adult Mortality", value=200)
input_data["infant deaths"] = st.number_input("Infant Deaths", value=20)
input_data["Alcohol"] = st.number_input("Alcohol consumption", value=4.0)
input_data["BMI"] = st.number_input("BMI", value=25.0)
input_data["Hepatitis B"] = st.number_input("Hepatitis B immunization", value=90)
input_data["Schooling"] = st.number_input("Years of Schooling", value=12.0)
input_data["GDP"] = st.number_input("GDP", value=5000)
input_data["Population"] = st.number_input("Population", value=1000000)

# ---- Build dataframe ----
new_data = pd.DataFrame(0, index=[0], columns=feature_cols)

for key, value in input_data.items():
    if key in new_data.columns:
        new_data[key] = value

# ---- Predict ----
if st.button("Predict Life Expectancy"):
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)
    st.success(f"Predicted Life Expectancy: {round(prediction[0], 2)} years")
