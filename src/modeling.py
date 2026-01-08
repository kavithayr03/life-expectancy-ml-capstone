# modeling.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# ---- Step 1: Load cleaned data ----
df = pd.read_csv("../data/processed/cleaned_life_expectancy.csv")

# Strip spaces from column names (important)
df.columns = df.columns.str.strip()

print("Data loaded successfully!")
print(df.head())

# ---- Step 2: Split features and target ----
X = df.drop("Life expectancy", axis=1)
y = df["Life expectancy"]

# ---- Step 3: Train-test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- Step 4: Feature scaling ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- Step 5: Define models ----
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, eval_metric="rmse")
}

results = {}

# ---- Step 6: Train and evaluate ----
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    results[name] = {"R2": r2, "MSE": mse}

    print(f"\n{name} Performance:")
    print(f"R2 Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")

# ---- Step 7: Feature importance plots ----
for name in ["Random Forest", "XGBoost"]:
    model = models[name]
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        features = X.columns

        imp_df = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False).head(20)

        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=imp_df)
        plt.title(f"Top 20 Important Features - {name}")
        plt.tight_layout()
        plt.show()

# ---- Step 8: Create models folder & save feature columns ----
os.makedirs("../models", exist_ok=True)
joblib.dump(X.columns.tolist(), "../models/feature_columns.pkl")

# ---- Step 9: Save best model ----
best_model_name = max(results, key=lambda x: results[x]["R2"])
best_model = models[best_model_name]

joblib.dump(best_model, f"../models/{best_model_name.replace(' ', '_')}_model.pkl")

print(f"\nBest model '{best_model_name}' saved successfully!")
